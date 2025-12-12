from flask import Flask, Response, render_template_string
import cv2
import serial
import threading
import time
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Serial connection
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)

# Global states
status = "SCANNING"
last_event_time = 0
hold_mode = False
co2_values = []  # Store CO2 history for graph
current_co2 = 400

# Camera (NO PAGE REFRESH - realtime MJPEG)
camera = cv2.VideoCapture(0)

def read_serial():
    global status, last_event_time, hold_mode, current_co2, co2_values
    
    while True:
        now = time.time()
        
        # Your 5-second hold logic
        if hold_mode and (now - last_event_time >= 5):
            status = "SCANNING"
            hold_mode = False
        
        try:
            line = ser.readline().decode().strip()
            if line and ',' in line:
                parts = line.split(',')
                if len(parts) == 2:
                    human_status = parts[0]
                    co2_str = parts[1]
                    
                    # Your original human logic
                    if human_status == "HUMAN":
                        status = "HUMAN"
                        last_event_time = now
                        hold_mode = True
                    elif human_status == "NO HUMAN":
                        status = "NO HUMAN"
                        last_event_time = now
                        hold_mode = True
                    
                    # CO2 value
                    try:
                        current_co2 = int(co2_str)
                        co2_values.append(current_co2)
                        # Keep last 50 readings for graph
                        if len(co2_values) > 50:
                            co2_values.pop(0)
                    except:
                        pass
                        
        except:
            pass
        
        time.sleep(0.1)

threading.Thread(target=read_serial, daemon=True).start()

# Realtime camera feed (MJPEG stream)
def generate_camera():
    while True:
        success, frame = camera.read()
        if not success:
            continue
        
        # Add status overlay
        color = (0, 255, 0) if status == "HUMAN" else (0, 0, 255) if status == "NO HUMAN" else (255, 255, 0)
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"CO2: {current_co2} ppm", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# CO2 graph generator (realtime updating)
def generate_co2_graph():
    while True:
        plt.ioff()
        fig, ax = plt.subplots(figsize=(8, 3))
        
        if co2_values:
            ax.plot(co2_values, 'g-', linewidth=2)
            ax.fill_between(range(len(co2_values)), co2_values, alpha=0.3, color='green')
            ax.set_ylim(300, max(co2_values + [2000]))
        
        ax.set_title("CO2 Levels (ppm)", fontsize=14)
        ax.set_ylabel("ppm", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=80)
        plt.close(fig)
        buf.seek(0)
        img_bytes = buf.read()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + img_bytes + b'\r\n')
        
        time.sleep(1)  # Update graph every second

@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Human Detection Dashboard</title>
        <style>
            body { font-family: Arial; margin: 20px; }
            .container { display: flex; flex-direction: column; align-items: center; }
            .status { font-size: 24px; padding: 10px; margin: 10px; }
            .human { color: green; }
            .nohuman { color: red; }
            .scanning { color: orange; }
            .camera { margin: 20px 0; }
            .graph { margin: 20px 0; width: 800px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Human Detection System</h1>
            <div class="status" id="status">Status: SCANNING</div>
            <div class="camera">
                <h2>Live Camera</h2>
                <img src="/camera_feed" width="640">
            </div>
            <div class="graph">
                <h2>CO2 Levels</h2>
                <img src="/co2_graph" width="800">
            </div>
        </div>
        
        <script>
            // Update status without refreshing page
            function updateStatus() {
                fetch('/get_status')
                    .then(response => response.json())
                    .then(data => {
                        const statusDiv = document.getElementById('status');
                        statusDiv.textContent = `Status: ${data.status}`;
                        statusDiv.className = 'status ' + 
                            (data.status === 'HUMAN' ? 'human' : 
                             data.status === 'NO HUMAN' ? 'nohuman' : 'scanning');
                    });
            }
            
            // Update every 500ms
            setInterval(updateStatus, 500);
            updateStatus();
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/camera_feed')
def camera_feed():
    return Response(generate_camera(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/co2_graph')
def co2_graph():
    return Response(generate_co2_graph(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    return {'status': status, 'co2': current_co2}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
from flask import Flask, Response, render_template_string
import cv2
import serial
import threading
import time

app = Flask(__name__)

# Serial connection to ESP32
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)

# Global states
status = "SCANNING"
last_event_time = 0
hold_mode = False
current_co2 = 400

# Camera
camera = cv2.VideoCapture(0)

def read_serial():
    global status, last_event_time, hold_mode, current_co2
    
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
                    except:
                        pass
                        
        except:
            pass
        
        time.sleep(0.1)

threading.Thread(target=read_serial, daemon=True).start()

def generate_camera():
    while True:
        success, frame = camera.read()
        if not success:
            continue
        
        # Add status text to camera feed
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"CO2: {current_co2} ppm", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    html = '''
    <html>
    <head>
        <title>Human + CO2 Dashboard</title>
        <meta http-equiv="refresh" content="1">
    </head>
    <body>
        <h1>Status: <span style="color:red">STATUS_PLACEHOLDER</span></h1>
        <h2>CO2: <span style="color:blue">CO2_PLACEHOLDER</span> ppm</h2>
        <img src="/camera" width="640"/>
    </body>
    </html>
    '''
    html = html.replace('STATUS_PLACEHOLDER', status)
    html = html.replace('CO2_PLACEHOLDER', str(current_co2))
    return render_template_string(html)

@app.route('/camera')
def camera_feed():
    return Response(generate_camera(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
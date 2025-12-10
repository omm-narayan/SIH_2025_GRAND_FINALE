from flask import Flask, Response, render_template_string
import cv2
import serial
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import time

# ====== Flask App ======
app = Flask(__name__)

# ====== Serial Setup ======
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)  # wait for ESP32

# ====== Global data ======
co2_values = []
distance_values = []
presence_values = []

# ====== Camera Setup ======
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ====== Sensor Smoothing Parameters ======
PRESENCE_THRESHOLD = 30   # minimum value to count as actual presence
SMOOTH_WINDOW = 5         # number of readings for smoothing

# ====== Serial Reading Thread ======
def read_serial():
    global co2_values, distance_values, presence_values
    buffer = []
    while True:
        try:
            line = ser.readline().decode().strip()
            if line and ',' in line:
                parts = line.split(',')
                if len(parts) == 3:  # Now expecting 3 values
                    co2, distance, presence = parts
                    co2_values.append(float(co2))
                    distance_values.append(float(distance))
                    
                    # Smooth presence detection
                    val = float(presence)
                    buffer.append(val)
                    if len(buffer) > SMOOTH_WINDOW:
                        buffer.pop(0)
                    
                    # rolling average
                    avg_val = sum(buffer)/len(buffer)
                    
                    # threshold after smoothing
                    presence_values.append(avg_val if avg_val >= PRESENCE_THRESHOLD else 0)
                    
                    # keep last 50 points
                    co2_values = co2_values[-50:]
                    distance_values = distance_values[-50:]
                    presence_values = presence_values[-50:]
            
        except Exception as e:
            print(f"Serial error: {e}")
            pass

threading.Thread(target=read_serial, daemon=True).start()

# ====== Camera Frame Generator ======
def generate_camera():
    while True:
        success, frame = camera.read()
        if not success:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ====== Graph Generators ======
def generate_co2_graph():
    plt.ioff()
    fig, ax = plt.subplots(figsize=(5,2))
    while True:
        ax.clear()
        if co2_values:
            ax.plot(co2_values, color='green')
            ax.set_ylim(0, max(co2_values + [500]))
        ax.set_title("CO2 Levels")
        ax.set_ylabel("ppm")
        ax.set_xlabel("Samples")
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_bytes = buf.read()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + img_bytes + b'\r\n')
        time.sleep(0.5)

def generate_distance_graph():
    plt.ioff()
    fig, ax = plt.subplots(figsize=(5,2))
    while True:
        ax.clear()
        if distance_values:
            ax.plot(distance_values, color='blue')
            ax.set_ylim(0, max(distance_values + [10]))
        ax.set_title("Distance Measurement")
        ax.set_ylabel("cm")
        ax.set_xlabel("Samples")
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_bytes = buf.read()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + img_bytes + b'\r\n')
        time.sleep(0.5)

def generate_presence_graph():
    plt.ioff()
    fig, ax = plt.subplots(figsize=(5,2))
    while True:
        ax.clear()
        if presence_values:
            ax.plot(presence_values, color='red')
        ax.set_ylim(0, 120)
        ax.set_title("Presence Detection")
        ax.set_ylabel("%")
        ax.set_xlabel("Samples")
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_bytes = buf.read()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + img_bytes + b'\r\n')
        time.sleep(0.5)

# ====== Routes ======
@app.route('/')
def index():
    html = '''
    <html>
    <head>
        <title>ESP32 + Camera Dashboard</title>
    </head>
    <body>
        <h2>Live Camera</h2>
        <img src="/camera" width="640"/>
        <h2>CO2 Graph</h2>
        <img src="/co2_graph" width="640"/>
        <h2>Distance Graph</h2>
        <img src="/distance_graph" width="640"/>
    

    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/camera')
def camera_feed():
    return Response(generate_camera(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/co2_graph')
def co2_graph_feed():
    return Response(generate_co2_graph(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/distance_graph')
def distance_graph_feed():
    return Response(generate_distance_graph(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/presence_graph')
def presence_graph_feed():
    return Response(generate_presence_graph(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ====== Run App ======
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)

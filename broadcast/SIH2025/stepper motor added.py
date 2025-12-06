from flask import Flask, Response, render_template_string, request, redirect
import cv2
import serial
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import time
import RPi.GPIO as GPIO

# ================= Flask =================
app = Flask(__name__)

# ================= Serial =================
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)

co2_values = []
presence_values = []

# ================= Camera =================
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

PRESENCE_THRESHOLD = 30
SMOOTH_WINDOW = 5

# ================= Stepper GPIO =================
STEP = 20
DIR  = 21
EN   = 16

GPIO.setmode(GPIO.BCM)
GPIO.setup([STEP, DIR, EN], GPIO.OUT)
GPIO.output(EN, GPIO.LOW)   # enable driver

STEPS_PER_REV = 200   # change if microstepping
STEP_DELAY = 0.001

# ================= Serial Read =================
def read_serial():
    buffer = []
    while True:
        try:
            line = ser.readline().decode().strip()
            if ',' in line:
                co2, presence = line.split(',')
                co2_values.append(float(co2))

                buffer.append(float(presence))
                if len(buffer) > SMOOTH_WINDOW:
                    buffer.pop(0)

                avg = sum(buffer) / len(buffer)
                presence_values.append(avg if avg >= PRESENCE_THRESHOLD else 0)

                co2_values[:] = co2_values[-50:]
                presence_values[:] = presence_values[-50:]
        except:
            pass

threading.Thread(target=read_serial, daemon=True).start()

# ================= Stepper Function =================
def rotate_stepper(rotations, clockwise=True):
    GPIO.output(DIR, GPIO.HIGH if clockwise else GPIO.LOW)
    steps = int(rotations * STEPS_PER_REV)

    for _ in range(steps):
        GPIO.output(STEP, GPIO.HIGH)
        time.sleep(STEP_DELAY)
        GPIO.output(STEP, GPIO.LOW)
        time.sleep(STEP_DELAY)

# ================= Camera Stream =================
def generate_camera():
    while True:
        ret, frame = camera.read()
        if not ret:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

# ================= Graphs =================
def plot_stream(data, title, ymax):
    fig, ax = plt.subplots(figsize=(5,2))
    while True:
        ax.clear()
        ax.plot(data)
        ax.set_ylim(0, ymax)
        ax.set_title(title)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        yield (b'--frame\r\nContent-Type: image/png\r\n\r\n' +
               buf.read() + b'\r\n')
        time.sleep(0.5)

# ================= Routes =================
@app.route('/')
def index():
    return render_template_string("""
<html>
<head>
<title>Dashboard</title>
</head>
<body>

<h2>Live Camera</h2>
<img src="/camera" width="640">

<h2>CO2</h2>
<img src="/co2">

<h2>Presence</h2>
<img src="/presence">

<h2>Stepper Motor Control</h2>
<form action="/rotate" method="post">
  Rotations:
  <input type="number" step="0.1" name="rotations" value="1" required>
  <br><br>
  <button name="direction" value="cw">Rotate Clockwise</button>
  <button name="direction" value="ccw">Rotate Anti-Clockwise</button>
</form>

</body>
</html>
""")

@app.route('/rotate', methods=['POST'])
def rotate():
    rotations = float(request.form['rotations'])
    direction = request.form['direction']
    rotate_stepper(rotations, direction == 'cw')
    return redirect('/')

@app.route('/camera')
def camera_feed():
    return Response(generate_camera(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/co2')
def co2_graph():
    return Response(plot_stream(co2_values, "CO2 ppm", 5000),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/presence')
def presence_graph():
    return Response(plot_stream(presence_values, "Presence", 120),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ================= Run =================
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8080, threaded=True)
    finally:
        GPIO.cleanup()

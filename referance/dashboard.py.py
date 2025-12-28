[file name]: dashboard_v2.py
[file content begin]
from flask import Flask, Response, render_template_string
import cv2
import serial
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import io
import time
from datetime import datetime

# ====== Flask App ======
app = Flask(__name__)

# ====== Serial Setup ======
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)  # Wait for ESP32

# ====== Global Data ======
co2_values = []  # Store CO2 values
human_status = False  # Current human detection status
last_human_change = 0  # Time of last status change
status_history = []  # History of status for debouncing

# ====== Camera Setup ======
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ====== Parameters ======
MAX_DATA_POINTS = 100  # Maximum points to display
STATUS_DEBOUNCE = 5    # Number of consistent readings to change status

# ====== Serial Reading Thread ======
def read_serial():
    global co2_values, human_status, status_history
    
    while True:
        try:
            line = ser.readline().decode().strip()
            
            if line and ',' in line:
                parts = line.split(',')
                if len(parts) == 2:
                    # Parse CO2 value
                    try:
                        co2_val = float(parts[0])
                        co2_values.append(co2_val)
                        co2_values = co2_values[-MAX_DATA_POINTS:]  # Keep last N points
                    except ValueError:
                        continue
                    
                    # Parse human status
                    new_status = parts[1].strip() == "1"
                    
                    # Debouncing logic
                    status_history.append(new_status)
                    if len(status_history) > STATUS_DEBOUNCE:
                        status_history.pop(0)
                    
                    # Check if all recent readings are consistent
                    if len(status_history) == STATUS_DEBOUNCE:
                        all_same = all(s == status_history[0] for s in status_history)
                        if all_same and status_history[0] != human_status:
                            human_status = status_history[0]
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Human detected: {human_status}")
                    
        except Exception as e:
            print(f"Serial read error: {e}")
            time.sleep(0.1)

# Start serial reading thread
threading.Thread(target=read_serial, daemon=True).start()

# ====== Camera Frame Generator ======
def generate_camera():
    while True:
        success, frame = camera.read()
        if not success:
            continue
        
        # Add timestamp to frame
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ====== CO2 Graph Generator ======
def generate_co2_graph():
    plt.ioff()
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#f0f0f0')
    
    while True:
        ax.clear()
        
        if co2_values:
            # Plot CO2 data
            x_values = list(range(len(co2_values)))
            ax.plot(x_values, co2_values, color='green', linewidth=2, label='CO2 Level')
            
            # Add current value annotation
            last_value = co2_values[-1] if co2_values else 0
            ax.annotate(f'{last_value:.1f} ppm', 
                       xy=(len(co2_values)-1, last_value),
                       xytext=(len(co2_values)-15, last_value + 50),
                       arrowprops=dict(arrowstyle='->', color='green'),
                       fontsize=10, color='green')
            
            # Add threshold lines
            ax.axhline(y=1000, color='orange', linestyle='--', alpha=0.5, label='Warning (1000 ppm)')
            ax.axhline(y=2000, color='red', linestyle='--', alpha=0.5, label='Danger (2000 ppm)')
        
        # Chart formatting
        ax.set_facecolor('white')
        ax.set_title('CO2 Levels - Real Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Number', fontsize=10)
        ax.set_ylabel('CO2 (ppm)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Set y-axis limits
        if co2_values:
            max_val = max(co2_values) if max(co2_values) > 500 else 500
            ax.set_ylim(0, max_val * 1.1)
        else:
            ax.set_ylim(0, 500)
        
        ax.set_xlim(0, max(len(co2_values), 50))
        
        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0)
        img_bytes = buf.read()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + img_bytes + b'\r\n')
        
        time.sleep(0.5)  # Update every 500ms

# ====== Human Status Image Generator ======
def generate_human_status():
    plt.ioff()
    
    while True:
        # Create figure for status display
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor('#f0f0f0')
        ax.axis('off')  # Hide axes
        
        if human_status:
            # Human Detected - Green background
            ax.set_facecolor('#d4edda')
            status_text = "HUMAN DETECTED"
            text_color = '#155724'
            border_color = 'green'
            border_width = 3
            font_size = 36
        else:
            # No Human - Red background
            ax.set_facecolor('#f8d7da')
            status_text = "NO HUMAN"
            text_color = '#721c24'
            border_color = 'red'
            border_width = 2
            font_size = 32
        
        # Add border
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(border_width)
        
        # Add status text
        ax.text(0.5, 0.5, status_text, 
                ha='center', va='center', 
                fontsize=font_size, fontweight='bold',
                color=text_color,
                transform=ax.transAxes)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        ax.text(0.02, 0.02, f"Last update: {timestamp}", 
                fontsize=8, color='gray',
                transform=ax.transAxes)
        
        # Add breathing detection info
        info_text = "Breathing pattern monitoring active"
        ax.text(0.5, 0.1, info_text, 
                ha='center', va='center',
                fontsize=10, style='italic', color='gray',
                transform=ax.transAxes)
        
        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_bytes = buf.read()
        plt.close(fig)  # Close figure to free memory
        
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + img_bytes + b'\r\n')
        
        time.sleep(0.5)  # Update every 500ms

# ====== Routes ======
@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ESP32 Monitoring Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .header {
                text-align: center;
                margin-bottom: 20px;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .container {
                display: grid;
                grid-template-columns: 1fr;
                gap: 20px;
                max-width: 1200px;
                margin: 0 auto;
            }
            .card {
                background: white;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .card h2 {
                margin-top: 0;
                color: #333;
                border-bottom: 2px solid #f0f0f0;
                padding-bottom: 10px;
            }
            .status-container {
                text-align: center;
                padding: 20px;
            }
            img {
                max-width: 100%;
                height: auto;
                border-radius: 5px;
                display: block;
                margin: 0 auto;
            }
            .footer {
                text-align: center;
                margin-top: 20px;
                padding: 15px;
                color: #666;
                font-size: 14px;
            }
            @media (min-width: 768px) {
                .container {
                    grid-template-columns: repeat(2, 1fr);
                }
                .camera-card {
                    grid-column: span 2;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏠 ESP32 Monitoring Dashboard</h1>
            <p>Real-time CO2 Monitoring & Human Presence Detection</p>
        </div>
        
        <div class="container">
            <div class="card camera-card">
                <h2>📷 Live Camera Feed</h2>
                <img src="/camera_feed" alt="Live Camera">
            </div>
            
            <div class="card">
                <h2>📊 CO2 Levels</h2>
                <img src="/co2_graph" alt="CO2 Graph">
            </div>
            
            <div class="card">
                <h2>👤 Human Presence Status</h2>
                <div class="status-container">
                    <img src="/human_status" alt="Human Status">
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>System Status: <span style="color: green;">●</span> Active | 
               Last Updated: <span id="time">Loading...</span></p>
            <p>CO2 Sensor + Radar Breathing Detection System</p>
        </div>
        
        <script>
            // Update time every second
            function updateTime() {
                const now = new Date();
                document.getElementById('time').textContent = 
                    now.toLocaleTimeString();
            }
            setInterval(updateTime, 1000);
            updateTime();
            
            // Auto-refresh images to prevent caching issues
            function refreshImages() {
                const images = document.querySelectorAll('img[src*="feed"], img[src*="graph"], img[src*="status"]');
                images.forEach(img => {
                    const src = img.src;
                    img.src = '';
                    setTimeout(() => { img.src = src; }, 100);
                });
            }
            setInterval(refreshImages, 10000); // Refresh every 10 seconds
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

@app.route('/human_status')
def human_status_feed():
    return Response(generate_human_status(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ====== Run App ======
if __name__ == '__main__':
    print("Starting ESP32 Monitoring Dashboard...")
    print("Open your browser and navigate to: http://localhost:8080")
    print("Press Ctrl+C to stop the server")
    
    try:
        app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        camera.release()
        ser.close()
        print("Dashboard stopped.")
[file content end]
from flask import Flask, render_template, Response, jsonify
import serial
import threading
import cv2
import time
import io
import math
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ===== Configuration =====
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
CAMERA_INDEX = 0

# ===== Distance Estimation Parameters =====
DISTANCE_SMOOTHING_WINDOW = 5  # Number of samples for moving average
MIN_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to show distance

# ===== Flask App =====
app = Flask(__name__)

# ===== Enhanced Data Storage with Distance Confidence =====
class SensorData:
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.co2_values = deque(maxlen=max_history)
        self.presence_values = deque(maxlen=max_history)
        self.distance_values = deque(maxlen=50)  # Valid distances only
        self.raw_distances = deque(maxlen=20)    # Raw distances for filtering
        self.human_present = False
        self.current_distance = 0
        self.distance_confidence = 0
        self.last_human_time = 0
        self.human_timeout = 3  # seconds
        self.last_update = time.time()
        
    def update_from_serial(self, co2_ppm, presence, distance_str):
        """Update from ESP32 serial data"""
        self.co2_values.append(co2_ppm)
        self.presence_values.append(presence)
        
        try:
            distance = float(distance_str)
            
            # Store raw distance
            self.raw_distances.append(distance)
            
            # Only process if presence detected
            if presence == 1 and distance > 0.1:
                self.human_present = True
                self.last_human_time = time.time()
                
                # Apply moving average filter
                filtered_distance = self.apply_distance_filter()
                
                if filtered_distance > 0:
                    self.current_distance = filtered_distance
                    self.distance_values.append(filtered_distance)
                    self.calculate_confidence()
            else:
                # Check timeout
                if time.time() - self.last_human_time > self.human_timeout:
                    self.human_present = False
                    self.current_distance = 0
                    self.distance_confidence = 0
                    
        except ValueError:
            pass
            
        self.last_update = time.time()
        
    def apply_distance_filter(self):
        """Apply moving average and outlier rejection"""
        if len(self.raw_distances) < 3:
            return 0
            
        # Convert to list for processing
        distances = list(self.raw_distances)
        
        # Remove outliers using IQR method
        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        filtered = [d for d in distances if lower_bound <= d <= upper_bound]
        
        if not filtered:
            return 0
            
        # Apply weighted moving average (recent samples have more weight)
        weights = list(range(1, len(filtered) + 1))
        weighted_avg = sum(d * w for d, w in zip(filtered, weights)) / sum(weights)
        
        return round(weighted_avg, 2)  # Round to 2 decimal places
        
    def calculate_confidence(self):
        """Calculate confidence level for distance measurement"""
        if len(self.distance_values) < 3:
            self.distance_confidence = 0.3
            return
            
        # Confidence based on:
        # 1. Consistency of readings
        # 2. Number of consecutive detections
        # 3. Signal-to-noise ratio
        
        distances = list(self.distance_values)
        
        # Calculate standard deviation (lower = more confident)
        if len(distances) > 1:
            std_dev = np.std(distances)
            consistency = max(0, 1 - (std_dev / 2))  # Normalize
        else:
            consistency = 0.5
            
        # Recent activity bonus
        recency_bonus = min(1.0, len(self.distance_values) / 10)
        
        # Combined confidence
        self.distance_confidence = min(0.95, 0.7 * consistency + 0.3 * recency_bonus)
        
    def get_distance_category(self):
        """Categorize distance for UI display"""
        if not self.human_present or self.current_distance == 0:
            return "N/A"
        
        dist = self.current_distance
        
        if dist < 1.0:
            return f"VERY CLOSE ({dist}m)"
        elif dist < 2.0:
            return f"CLOSE ({dist}m)"
        elif dist < 4.0:
            return f"MEDIUM ({dist}m)"
        else:
            return f"FAR ({dist}m)"
            
    def get_data(self):
        has_valid_distance = (self.human_present and 
                             self.current_distance > 0 and 
                             self.distance_confidence > MIN_CONFIDENCE_THRESHOLD)
        
        return {
            "co2": list(self.co2_values),
            "presence": list(self.presence_values),
            "distance_values": list(self.distance_values) if has_valid_distance else [],
            "human": 1 if self.human_present else 0,
            "distance": self.current_distance,
            "distance_meters": self.current_distance,
            "distance_cm": self.current_distance * 100,
            "distance_category": self.get_distance_category(),
            "confidence": round(self.distance_confidence, 2),
            "co2_current": self.co2_values[-1] if self.co2_values else 0,
            "status": "HUMAN PRESENT" if self.human_present else "NO HUMAN",
            "has_distance_data": has_valid_distance,
            "timestamp": self.last_update,
            "detection_count": len(self.distance_values)
        }

sensor_data = SensorData()

# ===== Serial Communication =====
def serial_reader():
    """Read data from ESP32 with distance"""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for ESP32
        print(f"Connected to ESP32 on {SERIAL_PORT}")
        print("Waiting for sensor data (CO2,Presence,Distance,Status)...")
        
        while True:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line and ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 4:
                        co2_ppm = int(parts[0])
                        presence = 1 if parts[1] == '1' else 0
                        distance_str = parts[2]
                        status = parts[3]
                        
                        sensor_data.update_from_serial(co2_ppm, presence, distance_str)
                        
                        # Debug output
                        if presence == 1 and distance_str != '0.0':
                            print(f"[RADAR] Distance: {distance_str}m | Confidence: {sensor_data.distance_confidence:.2f}")
                            
            except (ValueError, IndexError) as e:
                continue
            except Exception as e:
                print(f"Serial error: {e}")
                
    except serial.SerialException as e:
        print(f"Serial connection error: {e}")

# ===== Enhanced Camera Stream with Distance Overlay =====
class CameraStream:
    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.daemon = True
        self.thread.start()
        
    def update_frame(self):
        while self.running:
            success, frame = self.camera.read()
            if success:
                # Get current sensor data
                data = sensor_data.get_data()
                
                # Add overlay with distance information
                self.add_sensor_overlay(frame, data)
                self.frame = frame
            time.sleep(0.033)  # ~30 FPS
    
    def add_sensor_overlay(self, frame, data):
        """Add sensor information overlay to frame"""
        height, width = frame.shape[:2]
        
        # Background for text (semi-transparent)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Status with color coding
        status_color = (0, 255, 0) if data['human'] else (0, 0, 255)
        cv2.putText(frame, f"Status: {data['status']}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Distance information (if available)
        if data['has_distance_data']:
            # Distance in meters and cm
            dist_text = f"Distance: {data['distance_meters']:.2f}m ({data['distance_cm']:.0f}cm)"
            cv2.putText(frame, dist_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Confidence indicator
            conf_text = f"Confidence: {data['confidence']*100:.0f}%"
            conf_color = (0, 255, 0) if data['confidence'] > 0.7 else (0, 165, 255) if data['confidence'] > 0.4 else (0, 0, 255)
            cv2.putText(frame, conf_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
            
            # Visual distance indicator (bar)
            bar_width = int(data['confidence'] * 100)
            cv2.rectangle(frame, (width - 120, 30), (width - 120 + bar_width, 50), conf_color, -1)
            cv2.rectangle(frame, (width - 120, 30), (width - 20, 50), (255, 255, 255), 1)
            cv2.putText(frame, "Confidence", (width - 120, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
            # Distance category
            cv2.putText(frame, f"Category: {data['distance_category']}", (width - 200, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
        
        return frame
            
    def get_frame(self):
        if self.frame is None:
            return None
        _, buffer = cv2.imencode('.jpg', self.frame)
        return buffer.tobytes()
        
    def stop(self):
        self.running = False
        self.camera.release()

camera = CameraStream(CAMERA_INDEX)

# ===== Enhanced Graph Generation =====
def generate_distance_graph():
    """Generate enhanced distance graph with confidence bands"""
    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    data = sensor_data.get_data()
    distances = data['distance_values']
    
    if distances and data['has_distance_data']:
        # Main distance plot
        x_vals = list(range(len(distances)))
        ax1.plot(x_vals, distances, color='purple', linewidth=3, marker='o', markersize=6)
        
        # Add confidence band
        if len(distances) > 1:
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            ax1.fill_between(x_vals, 
                           [d - std_dist*data['confidence'] for d in distances],
                           [d + std_dist*data['confidence'] for d in distances],
                           alpha=0.2, color='purple', label='Confidence Band')
        
        # Current distance line
        current_dist = data['distance']
        ax1.axhline(y=current_dist, color='red', linestyle='--', alpha=0.7, 
                   label=f'Current: {current_dist:.2f}m')
        
        # Zone indicators
        ax1.axhspan(0, 1, alpha=0.1, color='red', label='Very Close (0-1m)')
        ax1.axhspan(1, 2, alpha=0.1, color='orange', label='Close (1-2m)')
        ax1.axhspan(2, 4, alpha=0.1, color='yellow', label='Medium (2-4m)')
        ax1.axhspan(4, 6, alpha=0.1, color='green', label='Far (4-6m)')
        
        ax1.set_ylim(0, 6)
        ax1.set_ylabel('Distance (m)', fontsize=12)
        ax1.set_title(f'Human Distance: {current_dist:.2f}m | Confidence: {data["confidence"]*100:.0f}%', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Confidence plot
        confidence_vals = [data['confidence']] * len(distances)
        ax2.plot(x_vals, confidence_vals, color='blue', linewidth=2)
        ax2.fill_between(x_vals, confidence_vals, alpha=0.3, color='blue')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Confidence', fontsize=12)
        ax2.set_xlabel('Time (samples)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add confidence thresholds
        ax2.axhline(y=0.7, color='green', linestyle=':', alpha=0.5, label='High Conf')
        ax2.axhline(y=0.4, color='orange', linestyle=':', alpha=0.5, label='Medium Conf')
        
    else:
        # No human detected
        for ax in [ax1, ax2]:
            ax.text(0.5, 0.5, 'Waiting for human detection...\nRadar will show distance when human is detected', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax1.set_title('Distance Tracking - Standby Mode', fontsize=14)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def generate_co2_graph():
    """Generate CO2 graph"""
    plt.ioff()
    fig, ax = plt.subplots(figsize=(8, 4))
    
    co2_values = sensor_data.co2_values
    if co2_values:
        ax.plot(list(co2_values), color='green', linewidth=2)
        ax.fill_between(range(len(co2_values)), list(co2_values), alpha=0.3, color='green')
        
        # Add current value
        current = co2_values[-1] if co2_values else 0
        ax.annotate(f'Current: {current} ppm', 
                   xy=(len(co2_values)-1, current),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8))
        
        # Safe level indicator (1000 ppm)
        ax.axhline(y=1000, color='red', linestyle='--', alpha=0.5, label='Safe Limit')
        
        ax.set_ylim(400, max(co2_values + [2000]))
        ax.set_title('CO2 Concentration', fontsize=14, fontweight='bold')
        ax.set_ylabel('CO2 (ppm)', fontsize=12)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No CO2 data', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, color='gray')
        ax.set_ylim(0, 2000)
    
    ax.set_xlabel('Time (samples)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ===== Flask Routes =====
@app.route('/')
def index():
    """Serve main dashboard"""
    return render_template('index.html')

@app.route('/data')
def get_sensor_data():
    """JSON endpoint for sensor data"""
    return jsonify(sensor_data.get_data())

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            frame_bytes = camera.get_frame()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/distance_graph')
def distance_graph():
    """Distance graph image"""
    graph_bytes = generate_distance_graph()
    return Response(graph_bytes, mimetype='image/png')

@app.route('/co2_graph')
def co2_graph():
    """CO2 graph image"""
    graph_bytes = generate_co2_graph()
    return Response(graph_bytes, mimetype='image/png')

@app.route('/api/status')
def api_status():
    """API endpoint for system status"""
    data = sensor_data.get_data()
    return jsonify({
        "system": "online",
        "human_detected": data['human'],
        "distance_meters": data['distance_meters'],
        "distance_category": data['distance_category'],
        "confidence": data['confidence'],
        "co2_level": data['co2_current'],
        "detection_count": data['detection_count'],
        "timestamp": time.time()
    })

# ===== HTML Template =====
@app.route('/dashboard')
def dashboard():
    """Complete dashboard page"""
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Radar Distance Monitor</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            body {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                min-height: 100vh;
                padding: 20px;
                color: white;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding: 25px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .header h1 {
                font-size: 2.8em;
                margin-bottom: 10px;
                background: linear-gradient(45deg, #00dbde, #fc00ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .status-container {
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-top: 20px;
            }
            
            .status-box {
                padding: 15px 30px;
                border-radius: 15px;
                min-width: 200px;
                text-align: center;
                transition: all 0.3s ease;
            }
            
            .status-box.human {
                background: linear-gradient(45deg, #ff416c, #ff4b2b);
                animation: pulse 1.5s infinite;
                box-shadow: 0 0 20px rgba(255, 65, 108, 0.5);
            }
            
            .status-box.clear {
                background: linear-gradient(45deg, #11998e, #38ef7d);
                box-shadow: 0 0 20px rgba(56, 239, 125, 0.3);
            }
            
            .status-box.distance {
                background: linear-gradient(45deg, #8a2387, #f27121);
            }
            
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
            
            .dashboard-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                gap: 25px;
                margin-bottom: 30px;
            }
            
            .card {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 25px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: transform 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-10px);
                border-color: rgba(255, 255, 255, 0.3);
            }
            
            .card h2 {
                color: #fff;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid rgba(255, 255, 255, 0.2);
                font-size: 1.5em;
            }
            
            .card h2 i {
                margin-right: 10px;
            }
            
            .distance-display {
                text-align: center;
                padding: 30px;
            }
            
            .distance-value {
                font-size: 4em;
                font-weight: bold;
                background: linear-gradient(45deg, #00dbde, #fc00ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 20px 0;
            }
            
            .distance-category {
                font-size: 1.8em;
                color: #ffd700;
                margin-bottom: 15px;
            }
            
            .confidence-meter {
                margin-top: 20px;
            }
            
            .meter-bar {
                width: 100%;
                height: 25px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                overflow: hidden;
                margin-bottom: 10px;
            }
            
            .meter-fill {
                height: 100%;
                background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00);
                border-radius: 12px;
                transition: width 0.5s ease;
            }
            
            .camera-container img {
                width: 100%;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            }
            
            .graph-container {
                text-align: center;
            }
            
            .graph-container img {
                width: 100%;
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .sensor-readings {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-top: 20px;
            }
            
            .sensor-card {
                background: rgba(0, 0, 0, 0.3);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
            }
            
            .sensor-value {
                font-size: 2.5em;
                font-weight: bold;
                margin: 10px 0;
            }
            
            .co2-value {
                color: #00ff88;
            }
            
            .presence-value {
                color: #ff6b6b;
            }
            
            .footer {
                text-align: center;
                margin-top: 30px;
                color: rgba(255, 255, 255, 0.6);
                font-size: 0.9em;
            }
            
            .update-time {
                color: #00dbde;
                font-weight: bold;
            }
            
            .confidence-text {
                display: flex;
                justify-content: space-between;
                margin-top: 5px;
            }
            
            .low-conf { color: #ff6b6b; }
            .medium-conf { color: #ffd93d; }
            .high-conf { color: #6bcf7f; }
            
            @media (max-width: 768px) {
                .dashboard-grid {
                    grid-template-columns: 1fr;
                }
                
                .status-container {
                    flex-direction: column;
                    align-items: center;
                }
                
                .sensor-readings {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📡 Radar Distance Monitoring System</h1>
                <p>Real-time human detection with approximate distance estimation</p>
                
                <div class="status-container">
                    <div id="statusBox" class="status-box clear">
                        <h3>Status</h3>
                        <div id="statusText" class="status-text">AREA CLEAR</div>
                    </div>
                    
                    <div id="distanceBox" class="status-box distance">
                        <h3>Approximate Distance</h3>
                        <div id="distanceText" class="distance-category">N/A</div>
                        <div id="exactDistance" class="distance-value">0.00 m</div>
                    </div>
                    
                    <div class="status-box">
                        <h3>Confidence</h3>
                        <div id="confidenceValue" class="distance-value">0%</div>
                        <div class="confidence-meter">
                            <div class="meter-bar">
                                <div id="confidenceBar" class="meter-fill" style="width: 0%"></div>
                            </div>
                            <div class="confidence-text">
                                <span class="low-conf">Low</span>
                                <span class="medium-conf">Medium</span>
                                <span class="high-conf">High</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="dashboard-grid">
                <!-- Camera Feed -->
                <div class="card">
                    <h2>📹 Live Camera Feed</h2>
                    <div class="camera-container">
                        <img id="cameraFeed" src="/video_feed" alt="Live Camera">
                    </div>
                </div>
                
                <!-- Distance Graph -->
                <div class="card">
                    <h2>📈 Distance Tracking Graph</h2>
                    <div class="graph-container">
                        <img id="distanceGraph" src="/distance_graph" alt="Distance Graph">
                    </div>
                </div>
                
                <!-- CO2 Graph -->
                <div class="card">
                    <h2>🌫️ CO2 Levels</h2>
                    <div class="graph-container">
                        <img id="co2Graph" src="/co2_graph" alt="CO2 Graph">
                    </div>
                </div>
                
                <!-- Sensor Readings -->
                <div class="card">
                    <h2>📊 Sensor Readings</h2>
                    <div class="sensor-readings">
                        <div class="sensor-card">
                            <h4>CO2 Level</h4>
                            <div id="co2Value" class="sensor-value co2-value">0 ppm</div>
                            <div class="sensor-label">Current Reading</div>
                        </div>
                        <div class="sensor-card">
                            <h4>Human Presence</h4>
                            <div id="presenceValue" class="sensor-value presence-value">NO</div>
                            <div class="sensor-label">Detection Status</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 25px; padding: 15px; background: rgba(0,0,0,0.3); border-radius: 10px;">
                        <h4>📋 Detection Log</h4>
                        <div id="detectionLog" style="text-align: left; margin-top: 10px; font-family: monospace; font-size: 0.9em;">
                            Waiting for detections...
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <div>Last updated: <span id="updateTime" class="update-time">--:--:--</span></div>
                <div>Detection count: <span id="detectionCount" class="update-time">0</span> | System time: <span id="systemTime" class="update-time">0s</span></div>
            </div>
        </div>
        
        <script>
            let detectionHistory = [];
            let startTime = Date.now();
            
            function updateDashboard() {
                // Auto-refresh images with timestamp to prevent caching
                const timestamp = new Date().getTime();
                document.getElementById('distanceGraph').src = '/distance_graph?' + timestamp;
                document.getElementById('co2Graph').src = '/co2_graph?' + timestamp;
                document.getElementById('cameraFeed').src = '/video_feed?' + timestamp;
                
                // Fetch sensor data
                fetch('/data')
                    .then(response => response.json())
                    .then(data => {
                        // Update status
                        const statusBox = document.getElementById('statusBox');
                        const statusText = document.getElementById('statusText');
                        
                        if (data.human) {
                            statusBox.className = 'status-box human';
                            statusText.textContent = 'HUMAN DETECTED';
                            statusText.style.color = '#ffd700';
                        } else {
                            statusBox.className = 'status-box clear';
                            statusText.textContent = 'AREA CLEAR';
                            statusText.style.color = '#ffffff';
                        }
                        
                        // Update distance information
                        const distanceBox = document.getElementById('distanceBox');
                        const distanceText = document.getElementById('distanceText');
                        const exactDistance = document.getElementById('exactDistance');
                        
                        if (data.has_distance_data && data.distance > 0) {
                            distanceText.textContent = data.distance_category;
                            exactDistance.textContent = `${data.distance_meters.toFixed(2)} m`;
                            
                            // Color code based on distance
                            if (data.distance_meters < 1) {
                                exactDistance.style.color = '#ff416c';
                            } else if (data.distance_meters < 2) {
                                exactDistance.style.color = '#ffa726';
                            } else if (data.distance_meters < 4) {
                                exactDistance.style.color = '#ffd700';
                            } else {
                                exactDistance.style.color = '#00ff88';
                            }
                            
                            // Add to detection history
                            const detectionTime = new Date().toLocaleTimeString();
                            detectionHistory.unshift({
                                time: detectionTime,
                                distance: data.distance_meters.toFixed(2),
                                confidence: (data.confidence * 100).toFixed(0)
                            });
                            
                            if (detectionHistory.length > 5) {
                                detectionHistory.pop();
                            }
                            
                        } else {
                            distanceText.textContent = 'NO DISTANCE DATA';
                            exactDistance.textContent = '0.00 m';
                            exactDistance.style.color = '#888';
                        }
                        
                        // Update confidence
                        const confidenceValue = document.getElementById('confidenceValue');
                        const confidenceBar = document.getElementById('confidenceBar');
                        const confidencePercent = Math.round(data.confidence * 100);
                        
                        confidenceValue.textContent = `${confidencePercent}%`;
                        confidenceBar.style.width = `${confidencePercent}%`;
                        
                        // Color code confidence
                        if (confidencePercent > 70) {
                            confidenceValue.style.color = '#00ff88';
                        } else if (confidencePercent > 40) {
                            confidenceValue.style.color = '#ffd700';
                        } else {
                            confidenceValue.style.color = '#ff6b6b';
                        }
                        
                        // Update CO2
                        document.getElementById('co2Value').textContent = `${data.co2_current} ppm`;
                        
                        // Update presence
                        const presenceEl = document.getElementById('presenceValue');
                        presenceEl.textContent = data.human ? 'YES' : 'NO';
                        presenceEl.style.color = data.human ? '#ff6b6b' : '#00ff88';
                        
                        // Update detection log
                        const logEl = document.getElementById('detectionLog');
                        if (detectionHistory.length > 0) {
                            logEl.innerHTML = detectionHistory.map(d => 
                                `[${d.time}] Distance: ${d.distance}m | Confidence: ${d.confidence}%`
                            ).join('<br>');
                        }
                        
                        // Update detection count
                        document.getElementById('detectionCount').textContent = data.detection_count;
                        
                        // Update timestamps
                        const currentTime = new Date().toLocaleTimeString();
                        document.getElementById('updateTime').textContent = currentTime;
                        
                        const systemUptime = Math.floor((Date.now() - startTime) / 1000);
                        document.getElementById('systemTime').textContent = `${systemUptime}s`;
                    })
                    .catch(error => {
                        console.error('Error fetching data:', error);
                        document.getElementById('statusText').textContent = 'SYSTEM ERROR';
                        document.getElementById('statusBox').className = 'status-box human';
                    });
            }
            
            // Update every 1.5 seconds (faster updates for distance)
            setInterval(updateDashboard, 1500);
            
            // Initial update
            updateDashboard();
            
            // Update system time every second
            setInterval(() => {
                const systemUptime = Math.floor((Date.now() - startTime) / 1000);
                document.getElementById('systemTime').textContent = `${systemUptime}s`;
            }, 1000);
        </script>
    </body>
    </html>
    '''
    return html_template

# ===== Main Execution =====
def start_background_tasks():
    """Start all background threads"""
    # Start serial reader thread
    serial_thread = threading.Thread(target=serial_reader)
    serial_thread.daemon = True
    serial_thread.start()
    print("Serial reader thread started...")
    
    # Optional: If using HB-100 directly on Pi GPIO
    # radar_thread = threading.Thread(target=radar_processing)
    # radar_thread.daemon = True
    # radar_thread.start()

if __name__ == '__main__':
    print("=" * 60)
    print("Radar Distance Monitoring System")
    print("=" * 60)
    print("Features:")
    print("  • Human presence detection")
    print("  • Approximate distance estimation")
    print("  • Confidence-based filtering")
    print("  • Live camera feed with overlay")
    print("  • CO2 monitoring")
    print("=" * 60)
    
    start_background_tasks()
    time.sleep(1)
    
    print("\nStarting Flask server...")
    print("Dashboard available at:")
    print("  Local: http://localhost:5000/dashboard")
    print("  Network: http://<your-ip>:5000/dashboard")
    print("\nAPI endpoints:")
    print("  /data - JSON sensor data")
    print("  /api/status - System status")
    print("  /distance_graph - Distance graph")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
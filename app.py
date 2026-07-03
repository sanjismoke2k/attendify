import os
import cv2
import time
import json
import numpy as np
import pandas as pd
import base64
import hashlib
from datetime import datetime
from threading import Thread, Lock
from flask import Flask, jsonify, send_from_directory, request, Response, session, redirect, url_for

# --- Configuration ---
DATASET_DIR = "dataset"
ATTEND_CSV = "attendance.csv"
CONFIG_FILE = "config.json"
USERS_FILE = "users.json"
OPENCV_MODEL_PATH = "recognizer.yml"
OPENCV_LABELS_PATH = "labels.csv"

# ==========================================
# ADMIN CONFIGURATION
# Uses environment variable on Render, falls back to default for local dev
# ==========================================
SUPER_ADMIN_PASSWORD = os.environ.get("SUPER_ADMIN_PASSWORD", "admin123")
# ==========================================

# --- Default Settings ---
config = {
    "CAMERA_SOURCE": "remote_stream",  # 'local_server' or 'remote_stream'
    "MIN_PRESENCE_MINUTES": 0.1,
    "OPENCV_CONF_THRESHOLD": 70
}

# --- Data Management Helpers ---
def save_config():
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"[WARN] Could not save config: {e}")

def load_config():
    global config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config.update(json.load(f))
        except Exception as e:
            print(f"[WARN] Could not load config, using defaults: {e}")
    else:
        save_config()

def load_users():
    if not os.path.exists(USERS_FILE):
        default_db = {"teachers": [], "students": []}
        try:
            with open(USERS_FILE, 'w') as f:
                json.dump(default_db, f)
        except Exception as e:
            print(f"[WARN] Could not create users file: {e}")
        return default_db
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {"teachers": [], "students": []}

def save_users(data):
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[WARN] Could not save users: {e}")

def is_remote_source(source):
    """Check if camera source is a remote stream (handles both 'remote_stream' and 'remote_phone')."""
    return source in ('remote_stream', 'remote_phone')

os.makedirs(DATASET_DIR, exist_ok=True)
load_config()

# --- Seed Default Test Teacher ---
def seed_default_teacher():
    """Add a default test teacher (ID: 000, Password: 000) if not already present."""
    db = load_users()
    teachers = db.get("teachers", [])
    if not any(t['id'] == '000' for t in teachers):
        default_teacher = {
            "id": "000",
            "name": "Test Teacher",
            "password": hashlib.sha256("000".encode()).hexdigest(),
            "dept": "Testing",
            "status": "approved",
            "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        teachers.append(default_teacher)
        db["teachers"] = teachers
        save_users(db)
        print("[INFO] Default test teacher seeded (ID: 000, Password: 000)")

seed_default_teacher()

camera_running = False
camera_lock = Lock()
output_frame = None
input_frame = None
frame_lock = Lock()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def make_recognizer():
    try:
        return cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        raise RuntimeError("Install opencv-contrib-python-headless for LBPHFaceRecognizer.")

# --- Flask App Setup ---
app = Flask(__name__, static_folder='.', static_url_path='')
# Use a stable secret key from environment variable so sessions survive restarts on Render
app.secret_key = os.environ.get('SECRET_KEY', 'attendify-dev-secret-key-change-in-production')

# --- Health Check (for Render monitoring) ---
@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "camera_running": camera_running,
        "model_trained": os.path.exists(OPENCV_MODEL_PATH)
    })

# --- Server Info (for remote camera auto-configuration) ---
@app.route("/api/server_info")
def server_info():
    """Returns server URL info so remote camera can build correct upload URL."""
    return jsonify({
        "success": True,
        "camera_running": camera_running,
        "camera_source": config.get("CAMERA_SOURCE", "remote_stream")
    })

# --- Auth API Routes ---

@app.route("/api/auth/register", methods=["POST"])
def register_user():
    data = request.json
    role = data.get('role')
    user_id = data.get('id')
    password = data.get('password')
    name = data.get('name')

    if not all([role, user_id, password, name]):
        return jsonify({"success": False, "message": "Missing fields"}), 400

    db = load_users()
    collection = db.get(role + "s")

    if collection is None:
        return jsonify({"success": False, "message": "Invalid role"}), 400

    if any(u['id'] == user_id for u in collection):
        return jsonify({"success": False, "message": f"{role.capitalize()} ID already exists"}), 400

    pass_hash = hashlib.sha256(password.encode()).hexdigest()

    new_user = {
        "id": user_id,
        "name": name,
        "password": pass_hash,
        "dept": data.get('dept', ''),
        "status": "pending",
        "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    collection.append(new_user)
    db[role + "s"] = collection
    save_users(db)

    return jsonify({"success": True, "message": "Application submitted for approval."})

@app.route("/api/auth/login", methods=["POST"])
def login_user():
    data = request.json
    role = data.get('role')
    user_id = data.get('id')
    password = data.get('password')

    db = load_users()
    collection = db.get(role + "s", [])

    pass_hash = hashlib.sha256(password.encode()).hexdigest()

    user = next((u for u in collection if u['id'] == user_id and u['password'] == pass_hash), None)

    if user:
        status = user.get('status', 'approved')
        if status == 'pending':
            return jsonify({"success": False, "message": "Account Pending Admin Approval"}), 403
        if status == 'rejected':
            return jsonify({"success": False, "message": "Account Request Rejected"}), 403

        session['user_id'] = user['id']
        session['user_name'] = user['name']
        session['role'] = role
        session['is_admin'] = (role == 'teacher')

        return jsonify({"success": True, "user": user})
    else:
        return jsonify({"success": False, "message": "Invalid Credentials"}), 401

@app.route("/api/auth/logout", methods=["POST"])
def logout_user_api():
    session.clear()
    return jsonify({"success": True})

@app.route("/logout")
def logout_page_redirect():
    session.clear()
    return redirect("/login")

# --- Super Admin API Routes ---

@app.route("/api/super_admin/login", methods=["POST"])
def super_admin_login():
    data = request.json
    password = data.get('password')
    if password == SUPER_ADMIN_PASSWORD:
        session['is_super_admin'] = True
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Invalid Access Password"}), 401

@app.route("/api/super_admin/requests", methods=["GET"])
def get_pending_requests():
    if not session.get('is_super_admin'):
        return jsonify({"error": "Unauthorized"}), 401
    db = load_users()
    pending = []
    for s in db['students']:
        if s.get('status', 'approved') == 'pending':
            s['role'] = 'student'
            pending.append(s)
    for t in db['teachers']:
        if t.get('status', 'approved') == 'pending':
            t['role'] = 'teacher'
            pending.append(t)
    return jsonify(pending)

@app.route("/api/super_admin/action", methods=["POST"])
def admin_action():
    if not session.get('is_super_admin'):
        return jsonify({"error": "Unauthorized"}), 401
    data = request.json
    target_id, role, action = data.get('id'), data.get('role'), data.get('action')
    db = load_users()
    collection = db.get(role + "s", [])
    for user in collection:
        if user['id'] == target_id:
            user['status'] = 'approved' if action == 'approve' else 'rejected'
            save_users(db)
            return jsonify({"success": True})
    return jsonify({"success": False, "message": "User not found"}), 404

# --- Page Routes ---

@app.route("/")
def home():
    if session.get('user_id'):
        return send_from_directory(".", "index.html")
    return send_from_directory(".", "info.html")

@app.route("/login")
def login_page():
    return send_from_directory(".", "login.html")

@app.route("/info")
def info_page():
    return send_from_directory(".", "info.html")

@app.route("/access_manager")
def access_manager_page():
    return send_from_directory(".", "access_manager.html")

@app.route("/enroll")
def enroll_page():
    if session.get('role') != 'teacher':
        return redirect("/login")
    return send_from_directory(".", "enroll.html")

@app.route("/dashboard")
def dashboard_page():
    if not session.get('user_id'):
        return redirect("/login")
    return send_from_directory(".", "dashboard.html")

@app.route("/start_attendance")
def start_attendance_page():
    if session.get('role') != 'teacher':
        return redirect("/login")
    return send_from_directory(".", "attendance_control.html")

@app.route("/remote_camera")
def remote_camera_page():
    return send_from_directory(".", "remote_camera.html")

@app.route("/admin")
def admin_page():
    if not session.get('is_admin'):
        return redirect("/login")
    return send_from_directory(".", "admin.html")

# --- Camera & System API Routes ---

@app.route('/trigger_start_camera')
def trigger_start_camera():
    if not session.get('is_admin'):
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    global camera_running, input_frame
    with camera_lock:
        if not camera_running:
            camera_running = True
            load_config()
            input_frame = None
            Thread(target=recognize_and_attend, daemon=True).start()
            return jsonify({"success": True, "message": "Camera process starting."})
    return jsonify({"success": False, "message": "Camera is already running."})

@app.route('/trigger_stop_camera')
def trigger_stop_camera():
    if not session.get('is_admin'):
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    global camera_running
    with camera_lock:
        camera_running = False
    return jsonify({"success": True, "message": "Stop signal sent."})

@app.route('/upload_frame', methods=['POST', 'OPTIONS'])
def upload_frame():
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    global input_frame
    if not camera_running:
        resp = jsonify({"status": "ignored", "message": "Camera not running"})
        resp.headers.add('Access-Control-Allow-Origin', '*')
        return resp
    try:
        data = request.json['image']
        header, encoded = data.split(",", 1)
        image_data = base64.b64decode(encoded)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            resp = jsonify({"status": "error", "message": "Failed to decode image"})
            resp.headers.add('Access-Control-Allow-Origin', '*')
            return resp
        with frame_lock:
            input_frame = img.copy()
        resp = jsonify({"status": "received"})
        resp.headers.add('Access-Control-Allow-Origin', '*')
        return resp
    except Exception as e:
        resp = jsonify({"status": "error", "message": str(e)})
        resp.headers.add('Access-Control-Allow-Origin', '*')
        return resp

@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    if not session.get('is_admin'):
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    try:
        roll, name, photo = request.form['roll'], request.form['name'], request.files['photo']
        person_dir = os.path.join(DATASET_DIR, f"{roll}_{name}")
        os.makedirs(person_dir, exist_ok=True)
        photo.save(os.path.join(person_dir, f"{int(time.time()*1000)}.jpg"))
        # Clear existing models to force retrain
        for file in [OPENCV_MODEL_PATH, OPENCV_LABELS_PATH]:
            if os.path.exists(file):
                os.remove(file)
        Thread(target=train_opencv_model, daemon=True).start()
        return jsonify({"success": True, "message": "Photo saved. Models will update."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    global config
    if request.method == "POST":
        if not session.get('is_admin'):
            return jsonify({"success": False, "error": "Unauthorized"}), 401
        try:
            config.update(request.json)
            save_config()
            return jsonify({"success": True, "message": "Settings updated."})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    return jsonify({"success": True, "settings": config})

@app.route("/api/train_opencv", methods=["POST"])
def api_train_opencv():
    if not session.get('is_admin'):
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    Thread(target=train_opencv_model, daemon=True).start()
    return jsonify({"success": True, "message": "OpenCV model training initiated."})

@app.route("/api/dashboard_data")
def get_dashboard_data():
    if not session.get('user_id'):
        return jsonify({"error": "Unauthorized"}), 401
    try:
        all_students_df = pd.DataFrame()
        if os.path.exists(OPENCV_LABELS_PATH):
            try:
                all_students_df = pd.read_csv(OPENCV_LABELS_PATH).rename(columns={"roll": "Roll Number", "name": "Name"})
            except pd.errors.EmptyDataError:
                pass

        attendance_df = pd.DataFrame()
        if os.path.exists(ATTEND_CSV):
            try:
                attendance_df = pd.read_csv(ATTEND_CSV)
            except pd.errors.EmptyDataError:
                pass

        if not all_students_df.empty:
            all_students_df['Roll Number'] = all_students_df['Roll Number'].astype(str)
            if not attendance_df.empty:
                attendance_df['Roll Number'] = attendance_df['Roll Number'].astype(str)
                merged_df = pd.merge(all_students_df, attendance_df, on=["Roll Number", "Name"], how="left")
            else:
                merged_df = all_students_df.copy()
        elif not attendance_df.empty:
            merged_df = attendance_df.copy()
        else:
            # No data at all - return empty
            return jsonify({
                "attendance": [],
                "stats": {"total_students": 0, "present": 0, "absent": 0, "attendance_rate": 0}
            })

        # Safely fill missing columns and values
        if 'Status' not in merged_df.columns:
            merged_df['Status'] = 'absent'
        else:
            merged_df['Status'] = merged_df['Status'].fillna('absent')

        merged_df = merged_df.fillna('N/A')

        if session.get('role') == 'student':
            student_id = session.get('user_id')
            merged_df = merged_df[merged_df['Roll Number'] == student_id]

        total_students = len(all_students_df)
        present_students = len(merged_df[merged_df['Status'] == 'present'])
        stats = {
            "total_students": total_students,
            "present": present_students,
            "absent": total_students - present_students,
            "attendance_rate": round((present_students / total_students) * 100) if total_students > 0 else 0
        }
        return jsonify({"attendance": merged_df.to_dict(orient="records"), "stats": stats})
    except Exception as e:
        print(f"[ERROR] Dashboard: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/student_list")
def get_student_list():
    if not os.path.exists(OPENCV_LABELS_PATH):
        return jsonify([])
    try:
        return jsonify(pd.read_csv(OPENCV_LABELS_PATH).to_dict(orient='records'))
    except pd.errors.EmptyDataError:
        return jsonify([])

def stream_frames():
    global output_frame, camera_running
    while True:
        with camera_lock:
            if not camera_running:
                break
        with frame_lock:
            frame_to_encode = output_frame.copy() if output_frame is not None else None
        if frame_to_encode is None:
            frame_to_encode = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame_to_encode, "Waiting for camera...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode)
        if not flag:
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + encodedImage.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route("/video_feed")
def video_feed():
    return Response(stream_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def train_opencv_model():
    print("[INFO] Starting OpenCV model training...")
    faces, ids, label_map, current_id = [], [], {}, 0

    if not os.path.exists(DATASET_DIR):
        print("[ERROR] Dataset directory not found.")
        return

    for folder in sorted(os.listdir(DATASET_DIR)):
        person_path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(person_path):
            continue
        try:
            roll, name = folder.split("_", 1)
        except ValueError:
            continue

        label_map[current_id] = {"roll": roll, "name": name}

        for imgname in os.listdir(person_path):
            img_path = os.path.join(person_path, imgname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Detect face in the training image
            detected_faces = face_cascade.detectMultiScale(img, 1.2, 5)
            for (x, y, w, h) in detected_faces:
                faces.append(img[y:y+h, x:x+w])
                ids.append(current_id)

        current_id += 1

    if not faces:
        print("[ERROR] No faces found to train OpenCV.")
        return

    recognizer = make_recognizer()
    recognizer.train(faces, np.array(ids))
    recognizer.save(OPENCV_MODEL_PATH)

    # Save labels mapping
    label_list = [{"id": i, "roll": v["roll"], "name": v["name"]} for i, v in label_map.items()]
    pd.DataFrame(label_list).to_csv(OPENCV_LABELS_PATH, index=False)
    print("[SUCCESS] OpenCV model trained.")

def recognize_and_attend():
    global camera_running, output_frame, config, input_frame
    camera_source = config.get("CAMERA_SOURCE", "remote_stream")
    cap = None

    # --- Camera Initialization ---
    if camera_source == 'local_server':
        # NOTE: On Render, this won't work as there is no physical camera.
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[WARN] Cannot access local server camera. Falling back to remote mode.")
            cap = None  # Fall through to remote mode

    min_presence_seconds = config.get("MIN_PRESENCE_MINUTES", 0.1) * 60

    print("\n" + "="*50)
    print(f"SESSION STARTING\n  - Framework: OpenCV (Lite)\n  - Camera: {camera_source}")
    if cap is None and camera_source == 'local_server':
        print("  - NOTE: Local camera unavailable, waiting for remote frames")
    print("="*50)

    # --- Load OpenCV Models ---
    if not all([os.path.exists(OPENCV_MODEL_PATH), os.path.exists(OPENCV_LABELS_PATH)]):
        print("[ERROR] OpenCV model/labels not found. Please enroll a user first.")
        # Create dummy frame to show error
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy, "No Training Data", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(dummy, "Enroll students first", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        with frame_lock:
            output_frame = dummy
        time.sleep(5)
        with camera_lock:
            camera_running = False
        return

    recognizer = make_recognizer()
    recognizer.read(OPENCV_MODEL_PATH)

    df = pd.read_csv(OPENCV_LABELS_PATH)
    label_map = {int(r['id']): {"roll": str(r['roll']), "name": r['name']} for _, r in df.iterrows()}

    threshold = config.get("OPENCV_CONF_THRESHOLD", 70)
    attendance = {}
    last_frame_time = time.time()
    no_frame_count = 0

    while True:
        with camera_lock:
            if not camera_running:
                break

        frame = None

        # --- Frame Acquisition ---
        if camera_source == 'local_server' and cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
        else:
            # Remote Camera Mode (Phones sending images via /upload_frame)
            with frame_lock:
                if input_frame is not None:
                    frame = input_frame.copy()
            if frame is None:
                no_frame_count += 1
                # Show waiting message in the video feed
                if no_frame_count % 50 == 1:  # Update waiting frame periodically
                    waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(waiting_frame, "Waiting for remote camera...", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(waiting_frame, "Open /remote_camera on your phone", (70, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 1)
                    cv2.putText(waiting_frame, "and press 'Start Stream'", (130, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 1)
                    with frame_lock:
                        output_frame = waiting_frame
                time.sleep(0.1)
                continue

        no_frame_count = 0  # Reset counter when we get a frame

        # --- Timing ---
        now = datetime.now()
        time_delta = time.time() - last_frame_time
        last_frame_time = time.time()

        # --- Detection & Recognition ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces_detected:
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])

            name = "Unknown"
            roll = "Unknown"

            # LBPH Confidence: Lower is better. 0 = perfect match.
            if conf < threshold and id_ in label_map:
                roll = label_map[id_]["roll"]
                name = label_map[id_]["name"]

                # Attendance Logic
                if roll not in attendance:
                    attendance[roll] = {"name": name, "first_seen": now, "last_seen": now, "seconds": 0, "status": "absent"}

                attendance[roll]["last_seen"] = now
                attendance[roll]["seconds"] += time_delta

                if attendance[roll]["seconds"] >= min_presence_seconds:
                    attendance[roll]["status"] = "present"

            # Drawing
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            # Calculate time spent
            minutes = 0
            secs = 0
            if roll in attendance:
                minutes, secs = divmod(int(attendance[roll]['seconds']), 60)

            text = f"{roll} - {name} ({minutes}m {secs}s)"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Update output frame for the web feed
        with frame_lock:
            output_frame = frame.copy()

    if cap:
        cap.release()

    # Save final attendance
    if attendance:
        rows = []
        for r, v in attendance.items():
            row = {
                "Roll Number": r,
                "Name": v["name"],
                "First Seen": v["first_seen"].strftime("%Y-%m-%d %H:%M:%S"),
                "Last Seen": v["last_seen"].strftime("%Y-%m-%d %H:%M:%S"),
                "Status": v["status"]
            }
            seconds = v.get("seconds", 0)
            minutes, secs = divmod(int(seconds), 60)
            row["Time on Camera"] = f"{minutes}m {secs}s"
            rows.append(row)

        try:
            pd.DataFrame(rows).to_csv(ATTEND_CSV, index=False)
            print(f"[INFO] Attendance saved to {ATTEND_CSV}")
        except Exception as e:
            print(f"[ERROR] Could not save attendance: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n * Attendify Server (Lite) is running.\n * Access at: http://127.0.0.1:{port}\n")
    app.run(host='0.0.0.0', port=port, use_reloader=False)
import os, sys, json, time, uuid, datetime, threading, base64
from io import BytesIO

from flask import (
    Flask, render_template, Response, request,
    jsonify, send_file, redirect, url_for, session
)
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from model.predict   import EmotionDetector
from database.db     import init_db, new_session, save_record, close_session, \
                            get_records, get_session, list_sessions
from analytics.analytics import aggregate_session
from analytics.report_generator import generate_pdf

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.urandom(32)

REPORTS_DIR = os.path.join(BASE, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

init_db()

detector = EmotionDetector()
_camera: cv2.VideoCapture | None = None
_camera_lock = threading.Lock()
_latest_result: dict = {}
_recording = False
_active_session_key: str | None = None
_last_record_time: float = 0.0


def get_camera():
    global _camera
    with _camera_lock:
        if _camera is None or not _camera.isOpened():
            for idx in range(3):
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    _camera = cap
                    break
        return _camera


def release_camera():
    global _camera
    with _camera_lock:
        if _camera and _camera.isOpened():
            _camera.release()
        _camera = None

def generate_frames():
    global _latest_result, _recording, _active_session_key, _last_record_time

    while True:
        cam = get_camera()
        if cam is None:
            time.sleep(0.05)
            continue

        ret, frame = cam.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        annotated, results = detector.process_frame(frame)

        if results:
            r = results[0]
            _latest_result = {
                "face_found":      r.face_found,
                "dominant":        r.dominant,
                "scores":          r.scores,
                "confidence_score":r.confidence_score,
                "stress_level":    r.stress_level,
                "stress_score":    r.stress_score,
                "attention_state": r.attention_state,
                "attention_score": r.attention_score,
                "eye_contact_score":r.eye_contact_score,
                "head_stability":  r.head_stability,
                "smile_score":     r.smile_score,
                "fps":             r.fps,
                "face_count":      len([x for x in results if x.face_found]),
            }
            # Record to DB every ~1 second
            if _recording and _active_session_key and r.face_found:
                now = time.time()
                if now - _last_record_time >= 1.0:
                    save_record(_active_session_key, _latest_result)
                    _last_record_time = now

        ret2, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret2:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    sessions = list_sessions()
    return render_template("dashboard.html", sessions=sessions)


@app.route("/report/<session_key>")
def report_view(session_key):
    sess    = get_session(session_key)
    records = get_records(session_key)
    agg     = aggregate_session(records)

    timeline = []
    for i, r in enumerate(records):
        if i % 5 == 0:
            timeline.append({
                "t":   r.get("ts", "")[-8:],
                "emo": r.get("dominant", "Neutral"),
                "conf":r.get("confidence", 0),
                "str": r.get("stress_score", 0),
                "att": r.get("attention", 0),
            })

    return render_template("report.html",
                           sess=sess, agg=agg, timeline=timeline,
                           session_key=session_key)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    return jsonify(_latest_result)


@app.route("/api/session/start", methods=["POST"])
def session_start():
    global _recording, _active_session_key, _last_record_time
    data      = request.get_json(silent=True) or {}
    candidate = data.get("candidate", "Candidate")
    key       = uuid.uuid4().hex[:12]
    new_session(key, candidate)
    _active_session_key = key
    _recording          = True
    _last_record_time   = 0.0
    return jsonify({"session_key": key, "status": "started"})


@app.route("/api/session/stop", methods=["POST"])
def session_stop():
    global _recording, _active_session_key
    key = _active_session_key
    if key:
        close_session(key)
    _recording          = False
    _active_session_key = None
    return jsonify({"session_key": key, "status": "stopped"})


@app.route("/api/session/status")
def session_status():
    return jsonify({
        "recording": _recording,
        "session_key": _active_session_key,
    })


@app.route("/api/report/data/<session_key>")
def report_data(session_key):
    records = get_records(session_key)
    agg     = aggregate_session(records)
    sess    = get_session(session_key)
    return jsonify({"session": sess, "aggregated": agg, "records_count": len(records)})


@app.route("/api/report/pdf/<session_key>")
def download_pdf(session_key):
    records   = get_records(session_key)
    agg       = aggregate_session(records)
    sess      = get_session(session_key)
    fname     = f"report_{session_key}.pdf"
    out_path  = os.path.join(REPORTS_DIR, fname)
    generate_pdf({
        "candidate_name":  sess.get("candidate", "Candidate"),
        "date":            sess.get("started_at", ""),
        "duration_seconds":sess.get("duration_s", 0),
        "session_id":      session_key,
        "aggregated":      agg,
    }, out_path)
    return send_file(out_path, as_attachment=True,
                     download_name=f"EmotionSenseAI_Report_{session_key}.pdf")


@app.route("/api/camera/stop", methods=["POST"])
def camera_stop():
    release_camera()
    return jsonify({"status": "camera released"})


if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  EmotionSense AI  —  starting on http://127.0.0.1:5000")
    print("═" * 55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)

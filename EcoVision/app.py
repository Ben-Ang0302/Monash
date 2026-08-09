from flask import Flask, request, jsonify, send_from_directory, render_template_string, session
from ultralytics import YOLO
import joblib
import numpy as np
import pandas as pd
import cv2
import os
import time
import csv
import torch
import threading
import subprocess
import requests
import resend
from dotenv import load_dotenv
from datetime import datetime

# ======================================================
# CONFIG
# ======================================================

YOLO_MODEL_PATH = "best.pt"
SPECTRAL_MODEL_PATH = "random_forest_recycle.pkl"
SPECTRAL_FEATURE_COLUMNS_PATH = "random_forest_feature_columns.pkl"

HOST = "0.0.0.0"
PORT = 5000

CAPTURE_DIR = "static/captures"
LATEST_IMAGE_NAME = "latest.jpg"
DATASET_DIR = "static/dataset"
OPERATIONAL_LOG_DIR = "operational_logs"

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OPERATIONAL_LOG_DIR, exist_ok=True)
for folder in ["plastic", "paper", "glass", "metal", "null", "rejected"]:
    os.makedirs(os.path.join(DATASET_DIR, folder), exist_ok=True)

CLASS_NAMES = ["plastic", "paper", "glass", "metal", "null"]
SORTABLE_CLASSES = ["plastic", "paper", "glass", "metal"]

YOLO_TO_FINAL = {
    "plastic": "plastic",
    "paper": "paper",
    "cardboard": "paper",
    "glass": "glass",
    "metal": "metal",
    "can": "metal",
    "empty": "null",
    "null": "null",
    "trash": "null",
}

# Paper/report fusion setting
YOLO_WEIGHT = 1
SPECTRAL_WEIGHT = 0

# Decision safety rules
YOLO_NULL_OVERRIDE_THRESHOLD = 0.80      # if YOLO null >= 80%, ignore spectroscopy
REJECT_CONFIDENCE_THRESHOLD = 0.60      # reject/no-sort when final confidence < 60%
LOW_CONFIDENCE_THRESHOLD = 0.50         # Telegram alert when confidence < 50%

# Fullness and alert settings
FULLNESS_WARNING_LEVEL = 80
FULLNESS_FULL_LEVEL = 100
NULL_STREAK_LIMIT = 5
FULLNESS_STALE_SECONDS = 300            # 5 minutes after a completed cycle
PI_HEARTBEAT_TIMEOUT_SECONDS = 60        # Pi considered disconnected after no heartbeat for 60s
PI_STARTUP_GRACE_SECONDS = 60            # wait this long after server start before warning if no Pi heartbeat
ALERT_COOLDOWN_SECONDS = 600             # 10 minutes for repeat alert types
MODEL_DISAGREEMENT_ALERT = True

FUN_FACTS = [
    "Glass can be recycled repeatedly without losing quality.",
    "Clean recyclables are more likely to be accepted by recycling facilities.",
    "Paper recycling helps reduce the demand for new wood pulp.",
    "Plastic sorting is challenging because many plastics look visually similar.",
    "Smart recycling systems can reduce contamination in recycling streams.",
    "Recycling helps reduce landfill waste and saves natural resources.",
    "Every correctly sorted item helps improve the recycling stream."
]

# ======================================================
# NOTIFICATION CONFIG
# ======================================================

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
REPORT_EMAIL_TO = os.getenv("REPORT_EMAIL_TO", "")
REPORT_EMAIL_FROM = os.getenv("REPORT_EMAIL_FROM", "EcoVision <onboarding@resend.dev>")
EMAIL_REPORT_INTERVAL_SECONDS = int(os.getenv("EMAIL_REPORT_INTERVAL_SECONDS", "3600"))
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "http://100.108.237.7:5000")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "ben")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "1234")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "ecovision-local-admin-session-key")

resend.api_key = RESEND_API_KEY

telegram_alert_state = {
    "plastic": {"warning": False, "full": False},
    "paper": {"warning": False, "full": False},
    "glass": {"warning": False, "full": False},
    "metal": {"warning": False, "full": False},
}

notification_state = {
    "null_streak": 0,
    "fullness_stale_alert_sent": False,
    "all_full_alert_sent": False,
    "pi_connection_alert_sent": False,
    "alert_last_sent": {},
    "sensor_fault_alert_sent": {
        "plastic": False,
        "paper": False,
        "glass": False,
        "metal": False,
    },
}

# ======================================================
# DASHBOARD STATE
# ======================================================

def empty_confidence():
    return {cls: 0.0 for cls in CLASS_NAMES}


dashboard_state = {
    "system_state": "idle",
    "message": "Approach the bin to begin.",
    "item_number_today": 0,
    "rejected_today": 0,
    "latest_image": "/captures/latest.jpg",
    "image_ready": False,
    "image_version": 0,

    "yolo_prediction": "-",
    "spectral_prediction": "-",
    "detected_material": "-",
    "final_prediction": "-",
    "final_confidence": 0.0,
    "route_class": "-",
    "reject_reason": "",
    "decision_message": "",
    "image_saved_path": "",
    "server_time": 0.0,

    "yolo_confidence": empty_confidence(),
    "spectral_confidence": empty_confidence(),
    "fused_confidence": empty_confidence(),

    "bins": {
        "plastic": 0,
        "paper": 0,
        "glass": 0,
        "metal": 0,
    },
    "last_fullness_update": "-",
    "last_fullness_update_epoch": 0.0,
    "last_cycle_epoch": 0.0,
    "last_cycle_done_epoch": 0.0,
    "last_cycle_done": "-",
    "cycles_since_fullness_update": 0,
    "out_of_service": False,
    "out_of_service_reason": "",
    "operator_disabled": False,
    "service_mode": "normal",
    "pending_pi_commands": [],
    "last_pi_command": "-",
    "last_pi_command_result": "-",
    "last_pi_command_time": "-",

    # Raspberry Pi heartbeat/connection monitoring.
    # The Pi posts to /api/pi-heartbeat every few seconds.
    "server_start_epoch": time.time(),
    "last_pi_heartbeat": "-",
    "last_pi_heartbeat_epoch": 0.0,
    "pi_connected": False,
    "pi_status": "waiting_for_heartbeat",

    "sensor_faults": {
        "plastic": False,
        "paper": False,
        "glass": False,
        "metal": False,
    },

    "alerts": {
        "low_confidence_items": 0,
        "model_disagreements": 0,
        "null_streak": 0,
        "fullness_stale": False,
        "pi_connection_lost": False,
        "full_bin_rejections": 0,
        "low_confidence_rejections": 0,
        "yolo_null_overrides": 0,
    },

    "counts": {
        "plastic": 0,
        "paper": 0,
        "glass": 0,
        "metal": 0,
        "null": 0,
    },

    "logs": [
        "System online",
        "EcoVision interface initialized",
    ],

    "fun_fact": FUN_FACTS[0],
}

# ======================================================
# LOAD MODELS
# ======================================================

print("Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)

if torch.cuda.is_available():
    yolo_model.to("cuda")
    print("YOLO using GPU:", torch.cuda.get_device_name(0))
else:
    print("YOLO using CPU")

print("YOLO loaded")

print("Loading spectroscopy model...")
spectral_model = joblib.load(SPECTRAL_MODEL_PATH)

spectral_feature_columns = None
if os.path.exists(SPECTRAL_FEATURE_COLUMNS_PATH):
    try:
        spectral_feature_columns = joblib.load(SPECTRAL_FEATURE_COLUMNS_PATH)
        print(f"Spectroscopy feature columns loaded: {len(spectral_feature_columns)}")
    except Exception as e:
        print("Could not load spectroscopy feature columns:", e)

print("Spectroscopy model loaded")

# ======================================================
# FLASK
# ======================================================

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

from datetime import timedelta

app.permanent_session_lifetime = timedelta(minutes=10)

# ======================================================
# BASIC HELPERS
# ======================================================

def add_log(text):
    timestamp = datetime.now().strftime("%H:%M:%S")
    dashboard_state["logs"].insert(0, f"[{timestamp}] {text}")
    dashboard_state["logs"] = dashboard_state["logs"][:20]


def all_bins_full():
    return all(int(dashboard_state["bins"].get(cls, 0)) >= FULLNESS_FULL_LEVEL for cls in SORTABLE_CLASSES)


def class_bin_is_full(class_name):
    if class_name not in dashboard_state["bins"]:
        return False
    return int(dashboard_state["bins"].get(class_name, 0)) >= FULLNESS_FULL_LEVEL


def update_out_of_service_state():
    if dashboard_state.get("operator_disabled"):
        dashboard_state["out_of_service"] = True
        dashboard_state["out_of_service_reason"] = "operator_disabled"
        dashboard_state["service_mode"] = "operator_disabled"
        dashboard_state["system_state"] = "out_of_service"
        dashboard_state["message"] = "EcoVision is temporarily disabled by the operator."
        return True

    if all_bins_full():
        dashboard_state["out_of_service"] = True
        dashboard_state["out_of_service_reason"] = "all_bins_full"
        dashboard_state["service_mode"] = "all_bins_full"
        dashboard_state["system_state"] = "out_of_service"
        dashboard_state["message"] = "All recycling bins are full. Please call the operator."
        return True

    was_out = bool(dashboard_state.get("out_of_service"))
    dashboard_state["out_of_service"] = False
    dashboard_state["out_of_service_reason"] = ""
    dashboard_state["service_mode"] = "normal"
    if was_out and dashboard_state.get("system_state") == "out_of_service":
        dashboard_state["system_state"] = "idle"
        dashboard_state["message"] = "System ready for the next recycler."
    return False


def enqueue_pi_command(command, source="server"):
    """Queue a safe one-shot command for the Pi to execute on its next heartbeat."""
    command = str(command).strip().lower()
    allowed = {
        "disable_bin", "enable_bin", "reset_outputs", "restart_camera",
        "capture_test_image", "request_fullness", "status"
    }
    if command not in allowed:
        return False
    item = {
        "id": f"{int(time.time() * 1000)}_{len(dashboard_state.get('pending_pi_commands', []))}",
        "command": command,
        "source": source,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    dashboard_state.setdefault("pending_pi_commands", []).append(item)
    dashboard_state["last_pi_command"] = command
    dashboard_state["last_pi_command_time"] = item["created_at"]
    add_log(f"Queued Pi command: {command}")
    return True


def bin_status_label(fullness):
    fullness = int(fullness)
    if fullness >= FULLNESS_FULL_LEVEL:
        return "FULL"
    if fullness >= FULLNESS_WARNING_LEVEL:
        return "WARNING"
    return "NORMAL"


def reset_cycle_display():
    dashboard_state["yolo_prediction"] = "-"
    dashboard_state["spectral_prediction"] = "-"
    dashboard_state["detected_material"] = "-"
    dashboard_state["final_prediction"] = "-"
    dashboard_state["final_confidence"] = 0.0
    dashboard_state["route_class"] = "-"
    dashboard_state["reject_reason"] = ""
    dashboard_state["decision_message"] = ""
    dashboard_state["image_saved_path"] = ""
    dashboard_state["server_time"] = 0.0
    dashboard_state["yolo_confidence"] = empty_confidence()
    dashboard_state["spectral_confidence"] = empty_confidence()
    dashboard_state["fused_confidence"] = empty_confidence()
    dashboard_state["image_ready"] = False



# ======================================================
# ADMIN AUTH / ACCESS CONTROL
# ======================================================

def admin_logged_in():
    return bool(session.get("ecovision_admin"))


def require_admin_json(fn):
    def wrapper(*args, **kwargs):
        if not admin_logged_in():
            return jsonify({"status": "error", "message": "admin login required"}), 401
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


def log_system_event(event_type, details=""):
    os.makedirs(OPERATIONAL_LOG_DIR, exist_ok=True)
    filename = datetime.now().strftime("ecovision_events_%Y-%m-%d.csv")
    path = os.path.join(OPERATIONAL_LOG_DIR, filename)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "event_type", "details"])
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event_type,
            "details": details,
        })
# ======================================================
# TELEGRAM / EMAIL HELPERS
# ======================================================

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        add_log("Telegram not configured")
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        response = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        response.raise_for_status()
        add_log("Telegram notification sent")
        return True
    except Exception as e:
        add_log(f"Telegram notification error: {str(e)}")
        return False


def send_telegram_alert(alert_key, message, cooldown_seconds=ALERT_COOLDOWN_SECONDS):
    now = time.time()
    last_sent = notification_state["alert_last_sent"].get(alert_key, 0.0)
    if cooldown_seconds > 0 and now - last_sent < cooldown_seconds:
        add_log(f"Telegram alert skipped by cooldown: {alert_key}")
        return False

    ok = send_telegram_message(message)
    if ok:
        notification_state["alert_last_sent"][alert_key] = now
    return ok


def check_all_full_alert():
    if all_bins_full():
        if not notification_state["all_full_alert_sent"]:
            msg = f"""🚫 <b>EcoVision Out of Service</b>

All material bins are full.
Door opening and sorting are locked out until servicing is completed.

Dashboard: {DASHBOARD_URL}
Action required: Empty all bins and reset the system."""
            if send_telegram_alert("all_bins_full", msg, cooldown_seconds=0):
                notification_state["all_full_alert_sent"] = True
        update_out_of_service_state()
    else:
        if notification_state["all_full_alert_sent"]:
            msg = f"""✅ <b>EcoVision Service Restored</b>

At least one bin is no longer full. The kiosk can resume operation.

Dashboard: {DASHBOARD_URL}"""
            send_telegram_alert("all_bins_full_resolved", msg, cooldown_seconds=0)
        notification_state["all_full_alert_sent"] = False
        update_out_of_service_state()


def check_fullness_alerts(updated_bins=None):
    bins_to_check = updated_bins.keys() if updated_bins else dashboard_state["bins"].keys()

    for bin_name in bins_to_check:
        if bin_name not in dashboard_state["bins"]:
            continue

        fullness = int(dashboard_state["bins"][bin_name])
        state = telegram_alert_state[bin_name]

        if fullness >= FULLNESS_FULL_LEVEL and not state["full"]:
            msg = f"""🚨 <b>EcoVision Critical Alert</b>

<b>{bin_name.upper()}</b> bin is FULL.
Fullness: <b>{fullness}%</b>

System action: Items detected as {bin_name} will be rejected/no-sort until serviced.
Dashboard: {DASHBOARD_URL}"""
            if send_telegram_alert(f"bin_full_{bin_name}", msg, cooldown_seconds=0):
                state["full"] = True
                state["warning"] = True

        elif fullness >= FULLNESS_WARNING_LEVEL and not state["warning"]:
            msg = f"""⚠️ <b>EcoVision Warning</b>

<b>{bin_name.upper()}</b> bin is nearing full.
Fullness: <b>{fullness}%</b>

Action required: Empty during the next service round.
Dashboard: {DASHBOARD_URL}"""
            if send_telegram_alert(f"bin_warning_{bin_name}", msg, cooldown_seconds=0):
                state["warning"] = True

        elif fullness < FULLNESS_WARNING_LEVEL:
            if state["warning"] or state["full"]:
                msg = f"""✅ <b>EcoVision Bin Cleared</b>

<b>{bin_name.upper()}</b> bin is back to normal.
Current fullness: <b>{fullness}%</b>."""
                send_telegram_alert(f"bin_resolved_{bin_name}_{int(time.time())}", msg, cooldown_seconds=0)
            state["warning"] = False
            state["full"] = False

    check_all_full_alert()


def send_sensor_fault_alert(bin_name):
    if bin_name not in notification_state["sensor_fault_alert_sent"]:
        return
    if notification_state["sensor_fault_alert_sent"][bin_name]:
        return

    msg = f"""⚠️ <b>EcoVision Sensor Fault</b>

<b>{bin_name.upper()}</b> bin fullness sensor returned invalid reading.

Possible issue:
- Ultrasonic sensor wiring
- Trig/Echo pin problem
- Sensor alignment issue
- No echo received

Dashboard: {DASHBOARD_URL}
Action required: Check the {bin_name} bin fullness sensor."""
    if send_telegram_alert(f"sensor_fault_{bin_name}", msg, cooldown_seconds=0):
        notification_state["sensor_fault_alert_sent"][bin_name] = True


def check_classification_alerts(final_prediction, final_confidence, yolo_prediction,
                                spectral_prediction, route_class="-", reject_reason=""):
    if final_confidence < LOW_CONFIDENCE_THRESHOLD:
        dashboard_state["alerts"]["low_confidence_items"] += 1
        msg = f"""⚠️ <b>EcoVision Low Confidence Alert</b>

Decision: <b>{str(final_prediction).upper()}</b>
Route sent: <b>{str(route_class).upper()}</b>
Confidence: <b>{final_confidence * 100:.1f}%</b>
Reason: <b>{reject_reason or 'low model confidence'}</b>

Dashboard: {DASHBOARD_URL}
Action required: Inspect the item, lighting, camera view, or latest chamber image."""
        send_telegram_alert("low_confidence", msg)

    if route_class == "null" or final_prediction == "null":
        notification_state["null_streak"] += 1
    else:
        notification_state["null_streak"] = 0

    dashboard_state["alerts"]["null_streak"] = notification_state["null_streak"]

    if notification_state["null_streak"] == NULL_STREAK_LIMIT:
        msg = f"""⚠️ <b>EcoVision Sorting Quality Warning</b>

The system produced <b>{NULL_STREAK_LIMIT}</b> no-sort / NULL decisions in a row.

Possible causes:
- Item not placed correctly
- Poor chamber image
- Dirty or mixed material
- Spectroscopy reading unstable
- Target stream full causing repeated rejection

Dashboard: {DASHBOARD_URL}
Action required: Check chamber, lighting, bin status, camera view, and sensor readings."""
        send_telegram_alert("null_streak", msg)

    if MODEL_DISAGREEMENT_ALERT and yolo_prediction not in ["-", None] and spectral_prediction not in ["-", None] and yolo_prediction != spectral_prediction:
        dashboard_state["alerts"]["model_disagreements"] += 1
        msg = f"""⚠️ <b>EcoVision Model Disagreement</b>

Vision model: <b>{str(yolo_prediction).upper()}</b>
Spectroscopy model: <b>{str(spectral_prediction).upper()}</b>
Final decision: <b>{str(final_prediction).upper()}</b>
Route sent: <b>{str(route_class).upper()}</b>

Dashboard: {DASHBOARD_URL}
Action required: Review latest image and item placement if this repeats."""
        send_telegram_alert("model_disagreement", msg)


def check_fullness_update_watchdog():
    if dashboard_state["cycles_since_fullness_update"] <= 0:
        return

    last_cycle = float(dashboard_state.get("last_cycle_epoch", 0.0))
    last_fullness = float(dashboard_state.get("last_fullness_update_epoch", 0.0))
    if last_cycle <= 0 or last_fullness >= last_cycle:
        return

    if time.time() - last_cycle >= FULLNESS_STALE_SECONDS and not notification_state["fullness_stale_alert_sent"]:
        msg = f"""⚠️ <b>EcoVision Sensor Update Warning</b>

A classification cycle has completed, but no bin fullness update was received for 5 minutes.

Possible issue:
- Arduino USB serial problem
- Pi serial reading issue
- Ultrasonic fullness sensors not responding
- Laptop /api/fullness route not receiving data

Dashboard: {DASHBOARD_URL}
Action required: Check Arduino serial monitor, Pi USB connection, and sensor wiring."""
        if send_telegram_alert("fullness_stale", msg, cooldown_seconds=0):
            notification_state["fullness_stale_alert_sent"] = True
            dashboard_state["alerts"]["fullness_stale"] = True



def check_pi_connection_watchdog():
    """Warn operator if the laptop server has not received Pi heartbeat."""
    now = time.time()
    last_seen = float(dashboard_state.get("last_pi_heartbeat_epoch", 0.0))
    server_start = float(dashboard_state.get("server_start_epoch", now))

    # No heartbeat has ever been received since the server started.
    if last_seen <= 0:
        if now - server_start < PI_STARTUP_GRACE_SECONDS:
            return

        dashboard_state["pi_connected"] = False
        dashboard_state["pi_status"] = "not_established"
        dashboard_state["alerts"]["pi_connection_lost"] = True

        if not notification_state["pi_connection_alert_sent"]:
            msg = f"""⚠️ <b>EcoVision Pi Connection Warning</b>

No Raspberry Pi heartbeat received since laptop server startup.

Possible issue:
- Raspberry Pi is off
- Pi code is not running
- SERVER_BASE_URL is wrong in the Pi code
- WiFi/Tailscale connection issue

Dashboard: {DASHBOARD_URL}
Action required: Start the Pi code and confirm it can reach the laptop server."""
            if send_telegram_alert("pi_connection_not_established", msg, cooldown_seconds=0):
                notification_state["pi_connection_alert_sent"] = True
                add_log("Pi connection warning sent: heartbeat not established")
        return

    # Heartbeat was received before, but has gone stale.
    age = now - last_seen
    if age > PI_HEARTBEAT_TIMEOUT_SECONDS:
        dashboard_state["pi_connected"] = False
        dashboard_state["pi_status"] = "lost"
        dashboard_state["alerts"]["pi_connection_lost"] = True

        if not notification_state["pi_connection_alert_sent"]:
            msg = f"""⚠️ <b>EcoVision Pi Connection Lost</b>

No Raspberry Pi heartbeat received for more than {PI_HEARTBEAT_TIMEOUT_SECONDS} seconds.

Possible issue:
- Pi program stopped
- Pi lost WiFi/Tailscale
- Laptop server URL changed
- Network was interrupted

Last Pi heartbeat: {dashboard_state.get('last_pi_heartbeat', '-')}
Dashboard: {DASHBOARD_URL}"""
            if send_telegram_alert("pi_connection_lost", msg, cooldown_seconds=0):
                notification_state["pi_connection_alert_sent"] = True
                add_log("Pi connection lost warning sent")
    else:
        dashboard_state["pi_connected"] = True
        dashboard_state["pi_status"] = "online"
        dashboard_state["alerts"]["pi_connection_lost"] = False


def build_hourly_report_html():
    bins = dashboard_state["bins"]
    counts = dashboard_state["counts"]
    alerts = dashboard_state["alerts"]
    sensor_faults = dashboard_state["sensor_faults"]

    total_items = sum(counts.values())
    rejected = dashboard_state.get("rejected_today", 0)

    active_faults = [name.title() for name, fault in sensor_faults.items() if fault]
    active_fault_text = ", ".join(active_faults) if active_faults else "None"

    out_of_service = dashboard_state.get("out_of_service", False)
    pi_connected = dashboard_state.get("pi_connected", False)

    system_badge = "Out of Service" if out_of_service else "Operational"
    system_color = "#ef4444" if out_of_service else "#10b981"

    pi_badge = "Online" if pi_connected else dashboard_state.get("pi_status", "Waiting")
    pi_color = "#10b981" if pi_connected else "#f59e0b"

    def status_color(status):
        if status == "FULL":
            return "#ef4444"
        if status == "WARNING":
            return "#f59e0b"
        return "#10b981"

    def pill(text, color):
        return f"""
        <span style="
            display:inline-block;
            padding:5px 10px;
            border-radius:999px;
            background:{color}18;
            color:{color};
            font-size:12px;
            font-weight:700;
            letter-spacing:.3px;">
            {text}
        </span>
        """

    bin_rows = ""
    for name, value in bins.items():
        status = bin_status_label(value)
        color = status_color(status)
        fault = sensor_faults.get(name, False)

        bin_rows += f"""
        <tr>
            <td style="padding:14px 16px;border-bottom:1px solid #e5e7eb;font-weight:700;color:#111827;">
                {name.title()}
            </td>
            <td style="padding:14px 16px;border-bottom:1px solid #e5e7eb;color:#374151;">
                <div style="background:#e5e7eb;border-radius:999px;height:10px;width:130px;overflow:hidden;display:inline-block;vertical-align:middle;margin-right:10px;">
                    <div style="width:{int(value)}%;height:10px;background:{color};border-radius:999px;"></div>
                </div>
                <b>{value}%</b>
            </td>
            <td style="padding:14px 16px;border-bottom:1px solid #e5e7eb;">
                {pill(status, color)}
            </td>
            <td style="padding:14px 16px;border-bottom:1px solid #e5e7eb;">
                {pill("FAULT", "#ef4444") if fault else pill("OK", "#10b981")}
            </td>
        </tr>
        """

    count_rows = ""
    for name, value in counts.items():
        count_rows += f"""
        <tr>
            <td style="padding:12px 14px;border-bottom:1px solid #e5e7eb;color:#111827;font-weight:700;">
                {name.title()}
            </td>
            <td style="padding:12px 14px;border-bottom:1px solid #e5e7eb;color:#374151;text-align:right;">
                <b>{value}</b>
            </td>
        </tr>
        """

    recent_logs = "".join(
        f"""
        <tr>
            <td style="padding:10px 14px;border-bottom:1px solid #eef2f7;color:#4b5563;font-size:13px;">
                {log}
            </td>
        </tr>
        """
        for log in dashboard_state["logs"][:6]
    )

    return f"""
    <div style="margin:0;padding:0;background:#f3f6fb;font-family:Arial,Helvetica,sans-serif;color:#111827;">
        <div style="max-width:760px;margin:0 auto;padding:28px 14px;">

            <div style="
                background:linear-gradient(135deg,#0f172a,#164e63);
                border-radius:22px;
                padding:28px;
                color:white;
                box-shadow:0 12px 28px rgba(15,23,42,.18);">
                <div style="font-size:13px;letter-spacing:1.5px;text-transform:uppercase;color:#a5f3fc;font-weight:700;">
                    EcoVision Operator Report
                </div>
                <h1 style="margin:10px 0 8px;font-size:28px;line-height:1.2;">
                    Hourly Recycling System Summary
                </h1>
                <p style="margin:0;color:#d1e8ef;font-size:14px;">
                    Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}
                </p>
            </div>

            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0;">
                <div style="background:white;border-radius:18px;padding:18px;box-shadow:0 6px 18px rgba(15,23,42,.08);">
                    <div style="font-size:12px;color:#6b7280;font-weight:700;text-transform:uppercase;">Items Today</div>
                    <div style="font-size:30px;font-weight:800;margin-top:6px;color:#111827;">{total_items}</div>
                </div>
                <div style="background:white;border-radius:18px;padding:18px;box-shadow:0 6px 18px rgba(15,23,42,.08);">
                    <div style="font-size:12px;color:#6b7280;font-weight:700;text-transform:uppercase;">Rejected</div>
                    <div style="font-size:30px;font-weight:800;margin-top:6px;color:#111827;">{rejected}</div>
                </div>
                <div style="background:white;border-radius:18px;padding:18px;box-shadow:0 6px 18px rgba(15,23,42,.08);">
                    <div style="font-size:12px;color:#6b7280;font-weight:700;text-transform:uppercase;">System</div>
                    <div style="margin-top:12px;">{pill(system_badge, system_color)}</div>
                </div>
                <div style="background:white;border-radius:18px;padding:18px;box-shadow:0 6px 18px rgba(15,23,42,.08);">
                    <div style="font-size:12px;color:#6b7280;font-weight:700;text-transform:uppercase;">Pi Link</div>
                    <div style="margin-top:12px;">{pill(str(pi_badge).title(), pi_color)}</div>
                </div>
            </div>

            <div style="background:white;border-radius:22px;padding:22px;margin-bottom:18px;box-shadow:0 6px 18px rgba(15,23,42,.08);">
                <h2 style="margin:0 0 14px;font-size:18px;color:#111827;">Latest Decision</h2>
                <table style="width:100%;border-collapse:collapse;">
                    <tr>
                        <td style="padding:9px 0;color:#6b7280;">Final decision</td>
                        <td style="padding:9px 0;text-align:right;font-weight:800;color:#111827;">
                            {str(dashboard_state["final_prediction"]).upper()}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding:9px 0;color:#6b7280;">Route sent</td>
                        <td style="padding:9px 0;text-align:right;font-weight:800;color:#111827;">
                            {str(dashboard_state["route_class"]).upper()}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding:9px 0;color:#6b7280;">Confidence</td>
                        <td style="padding:9px 0;text-align:right;font-weight:800;color:#111827;">
                            {dashboard_state["final_confidence"] * 100:.1f}%
                        </td>
                    </tr>
                    <tr>
                        <td style="padding:9px 0;color:#6b7280;">Last fullness update</td>
                        <td style="padding:9px 0;text-align:right;font-weight:800;color:#111827;">
                            {dashboard_state["last_fullness_update"]}
                        </td>
                    </tr>
                </table>
            </div>

            <div style="background:white;border-radius:22px;padding:22px;margin-bottom:18px;box-shadow:0 6px 18px rgba(15,23,42,.08);">
                <h2 style="margin:0 0 14px;font-size:18px;color:#111827;">Bin Fullness</h2>
                <table style="width:100%;border-collapse:collapse;">
                    <tr style="background:#f8fafc;">
                        <th style="padding:12px 16px;text-align:left;color:#64748b;font-size:12px;text-transform:uppercase;">Bin</th>
                        <th style="padding:12px 16px;text-align:left;color:#64748b;font-size:12px;text-transform:uppercase;">Level</th>
                        <th style="padding:12px 16px;text-align:left;color:#64748b;font-size:12px;text-transform:uppercase;">Status</th>
                        <th style="padding:12px 16px;text-align:left;color:#64748b;font-size:12px;text-transform:uppercase;">Sensor</th>
                    </tr>
                    {bin_rows}
                </table>
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px;">
                <div style="background:white;border-radius:22px;padding:22px;box-shadow:0 6px 18px rgba(15,23,42,.08);">
                    <h2 style="margin:0 0 14px;font-size:18px;color:#111827;">Classification Counts</h2>
                    <table style="width:100%;border-collapse:collapse;">
                        {count_rows}
                    </table>
                </div>

                <div style="background:white;border-radius:22px;padding:22px;box-shadow:0 6px 18px rgba(15,23,42,.08);">
                    <h2 style="margin:0 0 14px;font-size:18px;color:#111827;">Alert Summary</h2>
                    <table style="width:100%;border-collapse:collapse;">
                        <tr><td style="padding:8px 0;color:#6b7280;">Sensor faults</td><td style="text-align:right;font-weight:800;">{active_fault_text}</td></tr>
                        <tr><td style="padding:8px 0;color:#6b7280;">Low-confidence items</td><td style="text-align:right;font-weight:800;">{alerts.get('low_confidence_items', 0)}</td></tr>
                        <tr><td style="padding:8px 0;color:#6b7280;">Model disagreements</td><td style="text-align:right;font-weight:800;">{alerts.get('model_disagreements', 0)}</td></tr>
                        <tr><td style="padding:8px 0;color:#6b7280;">Full-bin rejections</td><td style="text-align:right;font-weight:800;">{alerts.get('full_bin_rejections', 0)}</td></tr>
                        <tr><td style="padding:8px 0;color:#6b7280;">NULL/no-sort streak</td><td style="text-align:right;font-weight:800;">{alerts.get('null_streak', 0)}</td></tr>
                    </table>
                </div>
            </div>

            <div style="background:white;border-radius:22px;padding:22px;margin-bottom:18px;box-shadow:0 6px 18px rgba(15,23,42,.08);">
                <h2 style="margin:0 0 14px;font-size:18px;color:#111827;">Recent Activity</h2>
                <table style="width:100%;border-collapse:collapse;">
                    {recent_logs}
                </table>
            </div>

            <div style="text-align:center;color:#94a3b8;font-size:12px;padding:8px 0 20px;">
                EcoVision Smart Recycling Bin · Automated monitoring report
            </div>

        </div>
    </div>
    """


def send_hourly_email_report():
    if not RESEND_API_KEY or not REPORT_EMAIL_TO:
        add_log("Resend email not configured")
        return False

    try:
        resend.Emails.send({
            "from": REPORT_EMAIL_FROM,
            "to": [REPORT_EMAIL_TO],
            "subject": "EcoVision Hourly Operator Report",
            "html": build_hourly_report_html(),
        })
        add_log("Hourly email report sent")
        return True
    except Exception as e:
        add_log(f"Hourly email report error: {str(e)}")
        return False


def hourly_email_loop():
    time.sleep(5)
    while True:
        time.sleep(EMAIL_REPORT_INTERVAL_SECONDS)
        send_hourly_email_report()


def monitoring_loop():
    time.sleep(10)
    while True:
        time.sleep(15)
        check_fullness_update_watchdog()
        check_pi_connection_watchdog()

# ======================================================
# IMAGE / MODEL HELPERS
# ======================================================

def save_latest_image_from_upload(image_file):
    image_bytes = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image")

    latest_path = os.path.join(CAPTURE_DIR, LATEST_IMAGE_NAME)
    cv2.imwrite(latest_path, image)
    dashboard_state["latest_image"] = "/captures/latest.jpg"
    dashboard_state["image_ready"] = True
    dashboard_state["image_version"] += 1
    return image


def run_yolo(image):
    confidence = empty_confidence()
    result = yolo_model.predict(
        image,
        conf=0.30,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        half=torch.cuda.is_available(),
        verbose=False,
    )[0]

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            raw_class = yolo_model.names[cls_id]
            final_class = YOLO_TO_FINAL.get(raw_class)
            if final_class in confidence:
                confidence[final_class] = max(confidence[final_class], conf)

    if sum(confidence.values()) == 0:
        confidence["null"] = 1.0

    prediction = max(confidence, key=confidence.get)
    return prediction, confidence


WAVELENGTHS = [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940]


def build_spectral_feature_frame(raw_spectrum):
    raw = np.array(raw_spectrum, dtype=np.float32).flatten()
    if raw.size != 54:
        raise ValueError(f"Expected 54 raw spectral values, received {raw.size}")

    feature_names = []
    values = []

    for prefix, start in [("uv", 0), ("white", 18), ("ir", 36)]:
        for i, wl in enumerate(WAVELENGTHS):
            feature_names.append(f"{prefix}_{wl}")
            values.append(float(raw[start + i]))

    eps = 1e-8
    uv = raw[0:18]
    white = raw[18:36]
    ir = raw[36:54]

    for i, wl in enumerate(WAVELENGTHS):
        feature_names.append(f"ir_over_white_{wl}")
        values.append(float(ir[i] / (white[i] + eps)))
        feature_names.append(f"uv_over_white_{wl}")
        values.append(float(uv[i] / (white[i] + eps)))
        feature_names.append(f"ir_over_uv_{wl}")
        values.append(float(ir[i] / (uv[i] + eps)))

    df = pd.DataFrame([values], columns=feature_names)

    if spectral_feature_columns is not None:
        if len(spectral_feature_columns) == 54:
            return df.iloc[:, :54].copy()
        return df.reindex(columns=spectral_feature_columns, fill_value=0.0)

    expected = getattr(spectral_model, "n_features_in_", 108)
    if expected == 54:
        return df.iloc[:, :54].copy()
    return df


def run_spectroscopy(spectrum):
    X_spec = build_spectral_feature_frame(spectrum)
    probs = spectral_model.predict_proba(X_spec)[0]
    confidence = empty_confidence()

    for cls, prob in zip(spectral_model.classes_, probs):
        cls = str(cls)
        if cls in confidence:
            confidence[cls] = float(prob)

    prediction = max(confidence, key=confidence.get)
    return prediction, confidence


def fuse_results(yolo_confidence, spectral_confidence):
    fused = {}
    for cls in CLASS_NAMES:
        fused[cls] = YOLO_WEIGHT * yolo_confidence.get(cls, 0.0) + SPECTRAL_WEIGHT * spectral_confidence.get(cls, 0.0)

    total = sum(fused.values())
    if total > 0:
        for cls in fused:
            fused[cls] /= total

    prediction = max(fused, key=fused.get)
    confidence = fused[prediction]
    return prediction, confidence, fused


def build_decision(yolo_prediction, yolo_confidence, spectral_prediction, spectral_confidence):
    fused_prediction, fused_confidence, fused_map = fuse_results(yolo_confidence, spectral_confidence)
    yolo_null_conf = float(yolo_confidence.get("null", 0.0))

    decision = {
        "detected_material": fused_prediction,
        "final_prediction": fused_prediction,
        "final_confidence": float(fused_confidence),
        "fused_confidence": fused_map,
        "route_class": fused_prediction,
        "reject_reason": "",
        "decision_message": f"{fused_prediction.upper()} identified",
        "save_folder": fused_prediction if fused_prediction in CLASS_NAMES else "rejected",
    }

    if yolo_null_conf >= YOLO_NULL_OVERRIDE_THRESHOLD:
        dashboard_state["alerts"]["yolo_null_overrides"] += 1
        decision.update({
            "detected_material": "null",
            "final_prediction": "null",
            "final_confidence": yolo_null_conf,
            "route_class": "null",
            "reject_reason": "yolo_null_override",
            "decision_message": "Unable to sort: item appears invalid or non-recyclable.",
            "save_folder": "null",
        })
        return decision

    if fused_confidence < REJECT_CONFIDENCE_THRESHOLD:
        dashboard_state["alerts"]["low_confidence_rejections"] += 1
        decision.update({
            "detected_material": fused_prediction,
            "final_prediction": "null",
            "final_confidence": float(fused_confidence),
            "route_class": "null",
            "reject_reason": "low_confidence",
            "decision_message": f"Unable to sort confidently. Detected {fused_prediction}, but confidence is below 60%.",
            "save_folder": "rejected",
        })
        return decision

    if fused_prediction in SORTABLE_CLASSES and class_bin_is_full(fused_prediction):
        dashboard_state["alerts"]["full_bin_rejections"] += 1
        decision.update({
            "detected_material": fused_prediction,
            "final_prediction": fused_prediction,
            "final_confidence": float(fused_confidence),
            "route_class": "null",
            "reject_reason": "target_bin_full",
            "decision_message": f"{fused_prediction.upper()} detected, but its bin is full. Item rejected/no-sort.",
            "save_folder": fused_prediction,
        })
        return decision

    return decision


def safe_filename_part(value):
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_classified_image(image, save_folder, decision):
    if save_folder not in ["plastic", "paper", "glass", "metal", "null", "rejected"]:
        save_folder = "rejected"

    folder_path = os.path.join(DATASET_DIR, save_folder)
    os.makedirs(folder_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    detected = safe_filename_part(decision.get("detected_material", "unknown"))
    route = safe_filename_part(decision.get("route_class", "unknown"))
    reason = safe_filename_part(decision.get("reject_reason", "sorted") or "sorted")
    conf = float(decision.get("final_confidence", 0.0))
    filename = f"{timestamp}_{detected}_route-{route}_{reason}_conf{conf:.2f}.jpg"
    save_path = os.path.join(folder_path, filename)
    cv2.imwrite(save_path, image)
    return save_path

# ======================================================
# CSV LOGGING
# ======================================================

def current_log_path():
    date_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(OPERATIONAL_LOG_DIR, f"ecovision_cycles_{date_str}.csv")


def write_cycle_log(row):
    log_path = current_log_path()
    os.makedirs(OPERATIONAL_LOG_DIR, exist_ok=True)
    file_exists = os.path.exists(log_path)

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_cycle_log_row(timestamp, image_saved_path, raw_spectrum, feature_df,
                        yolo_prediction, yolo_confidence, spectral_prediction,
                        spectral_confidence, decision, server_time):
    row = {
        "timestamp": timestamp,
        "image_saved_path": image_saved_path,
        "latest_image_path": os.path.join(CAPTURE_DIR, LATEST_IMAGE_NAME),
        "yolo_prediction": yolo_prediction,
        "spectral_prediction": spectral_prediction,
        "detected_material": decision.get("detected_material", ""),
        "final_prediction": decision.get("final_prediction", ""),
        "final_confidence": decision.get("final_confidence", 0.0),
        "route_class": decision.get("route_class", ""),
        "reject_reason": decision.get("reject_reason", ""),
        "decision_message": decision.get("decision_message", ""),
        "server_time_s": server_time,
        "bin_plastic": dashboard_state["bins"].get("plastic"),
        "bin_paper": dashboard_state["bins"].get("paper"),
        "bin_glass": dashboard_state["bins"].get("glass"),
        "bin_metal": dashboard_state["bins"].get("metal"),
        "sensor_fault_plastic": dashboard_state["sensor_faults"].get("plastic"),
        "sensor_fault_paper": dashboard_state["sensor_faults"].get("paper"),
        "sensor_fault_glass": dashboard_state["sensor_faults"].get("glass"),
        "sensor_fault_metal": dashboard_state["sensor_faults"].get("metal"),
    }

    for cls in CLASS_NAMES:
        row[f"yolo_conf_{cls}"] = yolo_confidence.get(cls, 0.0)
        row[f"spectral_conf_{cls}"] = spectral_confidence.get(cls, 0.0)
        row[f"fused_conf_{cls}"] = decision.get("fused_confidence", {}).get(cls, 0.0)

    for idx, value in enumerate(raw_spectrum):
        row[f"raw_spectrum_{idx:02d}"] = value

    if feature_df is not None and len(feature_df) > 0:
        for col in feature_df.columns:
            row[f"feature_{col}"] = float(feature_df.iloc[0][col])

    return row

# ======================================================
# DASHBOARD UPDATE
# ======================================================

def update_dashboard_after_result(yolo_prediction, yolo_confidence, spectral_prediction,
                                  spectral_confidence, decision, server_time,
                                  image_saved_path=""):
    final_prediction = decision.get("final_prediction", "null")
    final_confidence = float(decision.get("final_confidence", 0.0))
    route_class = decision.get("route_class", "null")
    reject_reason = decision.get("reject_reason", "")

    dashboard_state["system_state"] = "complete"
    dashboard_state["message"] = decision.get("decision_message", f"{final_prediction.upper()} identified")
    dashboard_state["item_number_today"] += 1
    dashboard_state["yolo_prediction"] = yolo_prediction
    dashboard_state["spectral_prediction"] = spectral_prediction
    dashboard_state["detected_material"] = decision.get("detected_material", final_prediction)
    dashboard_state["final_prediction"] = final_prediction
    dashboard_state["final_confidence"] = final_confidence
    dashboard_state["route_class"] = route_class
    dashboard_state["reject_reason"] = reject_reason
    dashboard_state["decision_message"] = decision.get("decision_message", "")
    dashboard_state["image_saved_path"] = image_saved_path
    dashboard_state["server_time"] = server_time
    dashboard_state["yolo_confidence"] = yolo_confidence
    dashboard_state["spectral_confidence"] = spectral_confidence
    dashboard_state["fused_confidence"] = decision.get("fused_confidence", empty_confidence())

    if final_prediction in dashboard_state["counts"]:
        dashboard_state["counts"][final_prediction] += 1

    if route_class == "null" or reject_reason:
        dashboard_state["rejected_today"] += 1

    dashboard_state["fun_fact"] = FUN_FACTS[dashboard_state["item_number_today"] % len(FUN_FACTS)]
    dashboard_state["last_cycle_epoch"] = time.time()
    dashboard_state["cycles_since_fullness_update"] += 1

    if reject_reason:
        add_log(f"Decision: {final_prediction} ({final_confidence:.2f}) -> NO SORT ({reject_reason})")
    else:
        add_log(f"Final decision: {final_prediction} ({final_confidence:.2f})")

    check_classification_alerts(final_prediction, final_confidence, yolo_prediction,
                                spectral_prediction, route_class=route_class,
                                reject_reason=reject_reason)

# ======================================================
# API ROUTES
# ======================================================


@app.route("/api/pi-heartbeat", methods=["POST"])
def api_pi_heartbeat():
    """Raspberry Pi heartbeat endpoint with safe command return."""
    try:
        data = request.get_json(silent=True) or {}
        now = time.time()
        previous_connected = bool(dashboard_state.get("pi_connected"))
        previous_status = dashboard_state.get("pi_status", "waiting_for_heartbeat")
        alert_was_sent = bool(notification_state.get("pi_connection_alert_sent"))

        dashboard_state["last_pi_heartbeat_epoch"] = now
        dashboard_state["last_pi_heartbeat"] = datetime.now().strftime("%H:%M:%S")
        dashboard_state["pi_connected"] = True
        dashboard_state["pi_status"] = data.get("status", "online")
        dashboard_state["alerts"]["pi_connection_lost"] = False

        if data.get("last_command"):
            dashboard_state["last_pi_command_result"] = data.get("last_command_result", "reported")
            dashboard_state["last_pi_command_time"] = datetime.now().strftime("%H:%M:%S")

        if alert_was_sent or ((not previous_connected) and previous_status in ["not_established", "lost"]):
            send_telegram_alert(
                "pi_connection_restored",
                f"✅ <b>EcoVision Pi Connection Restored</b>\n\nRaspberry Pi heartbeat received again at {dashboard_state['last_pi_heartbeat']}.\nDashboard: {DASHBOARD_URL}",
                cooldown_seconds=0,
            )

        notification_state["pi_connection_alert_sent"] = False
        update_out_of_service_state()

        pending = list(dashboard_state.get("pending_pi_commands", []))
        dashboard_state["pending_pi_commands"] = []
        service_lockout = bool(dashboard_state.get("operator_disabled") or all_bins_full())

        add_log("Pi heartbeat received")
        return jsonify({
            "status": "ok",
            "pi_connected": True,
            "last_pi_heartbeat": dashboard_state["last_pi_heartbeat"],
            "out_of_service": dashboard_state["out_of_service"],
            "out_of_service_reason": dashboard_state.get("out_of_service_reason", ""),
            "operator_disabled": bool(dashboard_state.get("operator_disabled")),
            "all_bins_full": all_bins_full(),
            "service_lockout": service_lockout,
            "arduino_command": "DISABLE" if service_lockout else "ENABLE",
            "commands": pending,
        })
    except Exception as e:
        add_log(f"Pi heartbeat error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/event", methods=["POST"])
def api_event():
    try:
        data = request.get_json(force=True)
        state = data.get("state", "idle")
        message = data.get("message", "")

        if state == "cycle_done":
            dashboard_state["last_cycle_done_epoch"] = time.time()
            dashboard_state["last_cycle_done"] = datetime.now().strftime("%H:%M:%S")
            if message:
                dashboard_state["message"] = message
            # Keep the public UI in the routing/transport phase; the frontend will
            # complete the cycle as soon as it sees last_cycle_done_epoch.
            if not dashboard_state.get("out_of_service"):
                dashboard_state["system_state"] = "sorting"
            add_log(f"Event: cycle_done - {message}")
            return jsonify({"status": "ok", "last_cycle_done": dashboard_state["last_cycle_done"]})

        if all_bins_full():
            dashboard_state["system_state"] = "out_of_service"
            dashboard_state["message"] = "All recycling bins are full. Please call the operator."
            dashboard_state["out_of_service"] = True
            add_log("Event ignored because all bins are full")
            return jsonify({"status": "ok", "system_state": "out_of_service"})

        dashboard_state["system_state"] = state
        dashboard_state["out_of_service"] = False
        if message:
            dashboard_state["message"] = message
        if state in ["idle", "door_open", "insert", "processing"]:
            reset_cycle_display()
        add_log(f"Event: {state} - {message}")
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/fullness", methods=["POST"])
def api_fullness():
    try:
        data = request.get_json(force=True)
        updated = {}
        faults = data.get("faults", [])

        for bin_name in faults:
            bin_name = str(bin_name).strip().lower()
            if bin_name in dashboard_state["sensor_faults"]:
                dashboard_state["sensor_faults"][bin_name] = True
                add_log(f"Sensor fault reported: {bin_name}")
                send_sensor_fault_alert(bin_name)

        for bin_name in SORTABLE_CLASSES:
            if bin_name in data:
                value = int(float(data[bin_name]))
                if value < 0:
                    dashboard_state["sensor_faults"][bin_name] = True
                    add_log(f"Sensor fault reported: {bin_name}")
                    send_sensor_fault_alert(bin_name)
                    continue

                value = max(0, min(100, value))
                dashboard_state["bins"][bin_name] = value
                dashboard_state["sensor_faults"][bin_name] = False
                notification_state["sensor_fault_alert_sent"][bin_name] = False
                updated[bin_name] = value

        if updated or faults:
            dashboard_state["last_fullness_update"] = datetime.now().strftime("%H:%M:%S")
            dashboard_state["last_fullness_update_epoch"] = time.time()
            dashboard_state["cycles_since_fullness_update"] = 0
            notification_state["fullness_stale_alert_sent"] = False
            dashboard_state["alerts"]["fullness_stale"] = False

        if updated:
            add_log("Bin fullness updated: " + ", ".join(f"{k}={v}%" for k, v in updated.items()))
            check_fullness_alerts(updated)
        else:
            check_all_full_alert()

        return jsonify({
            "status": "ok",
            "bins": dashboard_state["bins"],
            "sensor_faults": dashboard_state["sensor_faults"],
            "last_fullness_update": dashboard_state["last_fullness_update"],
            "out_of_service": dashboard_state["out_of_service"],
        })
    except Exception as e:
        add_log(f"Fullness update error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/image", methods=["POST"])
def api_image():
    try:
        image_file = request.files["image"]
        save_latest_image_from_upload(image_file)
        dashboard_state["system_state"] = "classifying"
        dashboard_state["message"] = "Understanding material structure."
        add_log("New chamber image received")
        return jsonify({
            "status": "ok",
            "latest_image": dashboard_state["latest_image"],
            "image_version": dashboard_state["image_version"],
        })
    except Exception as e:
        dashboard_state["system_state"] = "error"
        dashboard_state["message"] = "Image upload error"
        add_log(f"ERROR: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/infer", methods=["POST"])
def infer():
    try:
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        add_log("Received image and spectroscopy data from Pi")

        if "image" in request.files:
            image = save_latest_image_from_upload(request.files["image"])
        else:
            latest_path = os.path.join(CAPTURE_DIR, LATEST_IMAGE_NAME)
            image = cv2.imread(latest_path)
            if image is None:
                raise ValueError("No image uploaded and latest.jpg does not exist")

        dashboard_state["system_state"] = "classifying"
        dashboard_state["message"] = "Understanding material structure."

        spectrum_text = request.form["spectrum"]
        spectrum = [float(x) for x in spectrum_text.split(",") if x.strip() != ""]
        feature_df = build_spectral_feature_frame(spectrum)

        yolo_prediction, yolo_confidence = run_yolo(image)
        add_log(f"Vision result: {yolo_prediction}")

        spectral_prediction, spectral_confidence = run_spectroscopy(spectrum)
        add_log(f"Material result: {spectral_prediction}")

        decision = build_decision(yolo_prediction, yolo_confidence, spectral_prediction, spectral_confidence)
        elapsed = time.time() - start_time

        image_saved_path = save_classified_image(image, decision.get("save_folder", "rejected"), decision)
        update_dashboard_after_result(yolo_prediction, yolo_confidence, spectral_prediction,
                                      spectral_confidence, decision, elapsed,
                                      image_saved_path=image_saved_path)

        log_row = build_cycle_log_row(timestamp, image_saved_path, spectrum, feature_df,
                                      yolo_prediction, yolo_confidence, spectral_prediction,
                                      spectral_confidence, decision, elapsed)
        write_cycle_log(log_row)
        add_log("Cycle logged to CSV")

        print("\n==============================")
        print("INFERENCE RESULT")
        print("==============================")
        print("VISION   :", yolo_prediction, yolo_confidence)
        print("MATERIAL :", spectral_prediction, spectral_confidence)
        print("DETECTED :", decision.get("detected_material"))
        print("FINAL    :", decision.get("final_prediction"), decision.get("final_confidence"))
        print("ROUTE    :", decision.get("route_class"), "REASON:", decision.get("reject_reason"))
        print("IMAGE    :", image_saved_path)
        print(f"Time     : {elapsed:.3f}s")
        print("==============================\n")

        return jsonify({
            "status": "ok",
            "yolo_prediction": yolo_prediction,
            "yolo_confidence": yolo_confidence,
            "spectral_prediction": spectral_prediction,
            "spectral_confidence": spectral_confidence,
            "detected_material": decision.get("detected_material"),
            "final_prediction": decision.get("final_prediction"),
            "final_confidence": decision.get("final_confidence"),
            "fused_confidence": decision.get("fused_confidence"),
            "route_class": decision.get("route_class"),
            "reject_reason": decision.get("reject_reason"),
            "decision_message": decision.get("decision_message"),
            "image_saved_path": image_saved_path,
            "server_time": elapsed,
        })
    except Exception as e:
        dashboard_state["system_state"] = "error"
        dashboard_state["message"] = "Analysis error"
        add_log(f"ERROR: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route("/api/test/pi-heartbeat", methods=["POST", "GET"])
def api_test_pi_heartbeat():
    with app.test_request_context('/api/pi-heartbeat', method='POST', json={"device": "manual_test", "status": "online"}):
        return api_pi_heartbeat()


@app.route("/api/test/telegram", methods=["GET", "POST"])
def api_test_telegram():
    ok = send_telegram_message("✅ <b>EcoVision Test Alert</b>\n\nTelegram notification service is working.")
    return jsonify({"status": "ok" if ok else "error"})


@app.route("/api/test/email", methods=["GET", "POST"])
def api_test_email():
    ok = send_hourly_email_report()
    return jsonify({"status": "ok" if ok else "error"})


@app.route("/api/test/fullness-warning", methods=["GET", "POST"])
def api_test_fullness_warning():
    dashboard_state["bins"]["plastic"] = 82
    check_fullness_alerts({"plastic": 82})
    return jsonify({"status": "ok", "plastic": 82})


@app.route("/api/test/fullness-full", methods=["GET", "POST"])
def api_test_fullness_full():
    dashboard_state["bins"]["plastic"] = 100
    check_fullness_alerts({"plastic": 100})
    return jsonify({"status": "ok", "plastic": 100})


@app.route("/api/test/sensor-fault", methods=["GET", "POST"])
def api_test_sensor_fault():
    dashboard_state["sensor_faults"]["glass"] = True
    send_sensor_fault_alert("glass")
    return jsonify({"status": "ok", "fault": "glass"})


@app.route("/api/test/all-full", methods=["GET", "POST"])
def api_test_all_full():
    for name in SORTABLE_CLASSES:
        dashboard_state["bins"][name] = 100
    check_all_full_alert()
    return jsonify({"status": "ok", "bins": dashboard_state["bins"], "out_of_service": dashboard_state["out_of_service"]})


@app.route("/api/admin/reset", methods=["POST", "GET"])
@require_admin_json
def api_admin_reset():
    dashboard_state["item_number_today"] = 0
    dashboard_state["rejected_today"] = 0
    dashboard_state["counts"] = {"plastic": 0, "paper": 0, "glass": 0, "metal": 0, "null": 0}
    dashboard_state["alerts"] = {
        "low_confidence_items": 0,
        "model_disagreements": 0,
        "null_streak": 0,
        "fullness_stale": False,
        "pi_connection_lost": False,
        "full_bin_rejections": 0,
        "low_confidence_rejections": 0,
        "yolo_null_overrides": 0,
    }
    notification_state["null_streak"] = 0
    notification_state["fullness_stale_alert_sent"] = False
    notification_state["pi_connection_alert_sent"] = False
    notification_state["alert_last_sent"] = {}
    reset_cycle_display()
    update_out_of_service_state()
    add_log("Daily counters and alert cooldowns reset")
    return jsonify({"status": "ok", "message": "EcoVision daily counters reset"})




@app.route("/api/admin/login", methods=["POST"])
def api_admin_login():
    data = request.get_json(force=True, silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session.permanent = True
        session["ecovision_admin"] = True       
        add_log("Admin login successful")
        return jsonify({"status": "ok", "message": "admin unlocked"})
    add_log("Admin login failed")
    return jsonify({"status": "error", "message": "invalid username or password"}), 401


@app.route("/api/admin/logout", methods=["POST", "GET"])
def api_admin_logout():
    session.pop("ecovision_admin", None)
    return jsonify({"status": "ok", "message": "admin logged out"})


@app.route("/api/admin/auth-status")
def api_admin_auth_status():
    return jsonify({"status": "ok", "logged_in": admin_logged_in()})


@app.route("/api/admin/disable", methods=["POST", "GET"])
@require_admin_json
def api_admin_disable():
    dashboard_state["operator_disabled"] = True
    update_out_of_service_state()
    enqueue_pi_command("disable_bin", source="admin")
    add_log("Operator disabled bin from admin UI")
    log_system_event("operator_disable", "Bin disabled from admin UI")
    send_telegram_alert(
        "operator_disabled",
        f"🛑 <b>EcoVision Operator Disable</b>\n\nThe recycling bin has been disabled from the admin interface.\nDashboard: {DASHBOARD_URL}",
        cooldown_seconds=0,
    )
    return jsonify({"status": "ok", "operator_disabled": True, "out_of_service": True})


@app.route("/api/admin/enable", methods=["POST", "GET"])
@require_admin_json
def api_admin_enable():
    dashboard_state["operator_disabled"] = False
    enqueue_pi_command("enable_bin", source="admin")
    update_out_of_service_state()
    add_log("Operator enabled bin from admin UI")
    log_system_event("operator_enable", "Bin enabled from admin UI")
    send_telegram_alert(
        "operator_enabled",
        f"✅ <b>EcoVision Service Enabled</b>\n\nThe recycling bin has been enabled from the admin interface.\nDashboard: {DASHBOARD_URL}",
        cooldown_seconds=0,
    )
    return jsonify({"status": "ok", "operator_disabled": False, "out_of_service": dashboard_state["out_of_service"]})


@app.route("/api/admin/clear-alerts", methods=["POST", "GET"])
@require_admin_json
def api_admin_clear_alerts():
    notification_state["fullness_stale_alert_sent"] = False
    notification_state["pi_connection_alert_sent"] = False
    notification_state["all_full_alert_sent"] = False
    notification_state["alert_last_sent"] = {}
    for b in SORTABLE_CLASSES:
        notification_state["sensor_fault_alert_sent"][b] = False
        telegram_alert_state[b]["warning"] = False
        telegram_alert_state[b]["full"] = False
    dashboard_state["alerts"]["fullness_stale"] = False
    dashboard_state["alerts"]["pi_connection_lost"] = False
    add_log("Admin cleared alert states and cooldowns")
    log_system_event("clear_alerts", "Alert states and cooldowns cleared")
    return jsonify({"status": "ok", "message": "alerts cleared"})


@app.route("/api/admin/pi-command", methods=["POST"])
@require_admin_json
def api_admin_pi_command():
    data = request.get_json(force=True, silent=True) or {}
    command = str(data.get("command", "")).strip().lower()
    if not enqueue_pi_command(command, source="admin"):
        return jsonify({"status": "error", "message": "invalid command"}), 400
    log_system_event("pi_command", command)
    return jsonify({"status": "ok", "queued": command})


@app.route("/api/admin/today-csv")
@require_admin_json
def api_admin_today_csv():
    filename = datetime.now().strftime("ecovision_cycles_%Y-%m-%d.csv")
    path = os.path.join(OPERATIONAL_LOG_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"status": "error", "message": "today cycle CSV not found yet"}), 404
    return send_from_directory(OPERATIONAL_LOG_DIR, filename, as_attachment=True)


@app.route("/api/pi-command-result", methods=["POST"])
def api_pi_command_result():
    data = request.get_json(force=True, silent=True) or {}
    command = str(data.get("command", "-"))
    result = str(data.get("result", "reported"))
    message = str(data.get("message", ""))
    dashboard_state["last_pi_command"] = command
    dashboard_state["last_pi_command_result"] = result if not message else f"{result}: {message}"
    dashboard_state["last_pi_command_time"] = datetime.now().strftime("%H:%M:%S")
    add_log(f"Pi command result: {command} -> {dashboard_state['last_pi_command_result']}")
    return jsonify({"status": "ok"})


@app.route("/api/status")
def api_status():
    update_out_of_service_state()
    return jsonify(dashboard_state)


@app.route("/captures/<path:filename>")
def captures(filename):
    return send_from_directory(CAPTURE_DIR, filename)


@app.route("/dataset/<path:filename>")
def dataset_files(filename):
    return send_from_directory(DATASET_DIR, filename)

# ======================================================
# FRONTEND
# ======================================================

HTML_PAGE = r"""

<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>EcoVision Kiosk</title><meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=Rajdhani:wght@500;600;700&display=swap');:root{--bg:#050814;--panel:rgba(8,18,36,.84);--card:rgba(255,255,255,.055);--cyan:#00e5ff;--green:#00ff9c;--purple:#8b5cf6;--yellow:#ffd166;--red:#ff4d6d;--text:#e8faff;--muted:#8aa4b8}*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:var(--bg);font-family:'Rajdhani',sans-serif;color:var(--text)}body:before{content:"";position:fixed;inset:0;background-image:linear-gradient(rgba(0,229,255,.055) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,.055) 1px,transparent 1px);background-size:44px 44px;animation:grid 12s linear infinite;z-index:0}body:after{content:"";position:fixed;inset:-30%;background:radial-gradient(circle at 20% 20%,rgba(0,229,255,.18),transparent 30%),radial-gradient(circle at 80% 30%,rgba(139,92,246,.18),transparent 30%),radial-gradient(circle at 50% 88%,rgba(0,255,156,.12),transparent 35%);animation:ambient 8s ease-in-out infinite alternate;z-index:0}@keyframes grid{to{transform:translateY(44px)}}@keyframes ambient{from{transform:translate(-1%,1%) scale(1)}to{transform:translate(1%,-1%) scale(1.04)}}.settings{position:fixed;top:22px;right:28px;z-index:60;width:52px;height:52px;border-radius:50%;border:1px solid rgba(0,229,255,.45);background:rgba(5,10,24,.75);backdrop-filter:blur(10px);color:#dffaff;display:flex;align-items:center;justify-content:center;cursor:pointer;box-shadow:0 0 24px rgba(0,229,255,.22);font-size:23px}.settings:hover{transform:rotate(45deg) scale(1.08);box-shadow:0 0 36px rgba(0,229,255,.65)}.page{position:fixed;inset:0;display:none;align-items:center;justify-content:center;padding:46px;text-align:center;z-index:5}.page.active{display:flex;animation:pageIn .45s ease forwards}@keyframes pageIn{from{opacity:0;transform:scale(1.02);filter:blur(8px)}to{opacity:1;transform:scale(1);filter:blur(0)}}.panel{position:relative;width:min(1180px,92vw);height:min(730px,84vh);border-radius:36px;background:var(--panel);border:1px solid rgba(0,229,255,.32);box-shadow:0 0 54px rgba(0,229,255,.20),inset 0 0 35px rgba(255,255,255,.035);backdrop-filter:blur(18px);overflow:hidden;padding:42px}.brand{font-family:'Orbitron';font-size:48px;letter-spacing:4px;text-shadow:0 0 24px rgba(0,229,255,.95);margin-bottom:8px}.subtitle{font-size:26px;color:var(--muted);margin-bottom:24px}.main-title{font-family:'Orbitron';font-size:58px;margin:22px 0 12px;text-shadow:0 0 18px rgba(0,229,255,.28)}.main-text{font-size:30px;color:#d6fbff}.status-pill{display:inline-block;margin-top:30px;padding:14px 34px;border-radius:999px;border:1px solid rgba(0,255,156,.5);color:var(--green);font-family:'Orbitron';font-size:15px;letter-spacing:1px;background:rgba(0,255,156,.08);box-shadow:0 0 28px rgba(0,255,156,.25)}.recycle-orb{width:245px;height:245px;border-radius:50%;margin:36px auto 26px;border:2px solid rgba(0,229,255,.5);display:flex;align-items:center;justify-content:center;font-size:105px;box-shadow:0 0 50px rgba(0,229,255,.5),inset 0 0 70px rgba(0,255,156,.15)}.red{color:var(--red);text-shadow:0 0 28px rgba(255,77,109,.55)}.scan-box{width:680px;height:385px;margin:24px auto 20px;border-radius:30px;position:relative;overflow:hidden;border:1px solid rgba(0,229,255,.45);background:radial-gradient(circle,rgba(0,229,255,.1),rgba(0,0,0,.55));box-shadow:0 0 40px rgba(0,229,255,.25)}.scan-box img{width:100%;height:100%;object-fit:cover;opacity:.88;position:absolute;inset:0;z-index:1}.scan-line{position:absolute;left:0;right:0;height:5px;background:var(--cyan);box-shadow:0 0 30px var(--cyan);animation:scanLine 1.6s linear infinite;z-index:5}@keyframes scanLine{from{top:-5%;opacity:0}20%{opacity:1}to{top:105%;opacity:0}}.analysis-grid{position:absolute;inset:0;z-index:3;mix-blend-mode:screen;background-image:linear-gradient(rgba(0,229,255,.15) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,.15) 1px,transparent 1px);background-size:34px 34px;opacity:.42}.material-icon{width:190px;height:190px;margin:30px auto 18px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:92px;background:radial-gradient(circle,rgba(0,255,156,.26),rgba(0,0,0,.26));border:2px solid rgba(0,255,156,.48);box-shadow:0 0 55px rgba(0,255,156,.36),inset 0 0 38px rgba(255,255,255,.04)}.material{font-family:'Orbitron';font-size:72px;color:var(--green);text-shadow:0 0 32px rgba(0,255,156,.8);margin-top:14px}.confidence{font-size:29px;color:#d6fbff;margin-top:12px}.item-number{margin-top:20px;font-family:'Orbitron';color:var(--yellow);font-size:25px}.thank{font-size:116px;text-shadow:0 0 35px var(--cyan)}.modal{display:none;position:fixed;inset:0;z-index:100;background:rgba(0,0,0,.68);backdrop-filter:blur(10px);align-items:center;justify-content:center}.modal.active{display:flex}.login-card{width:min(440px,90vw);border-radius:28px;background:rgba(5,10,24,.96);border:1px solid rgba(0,229,255,.35);padding:30px;box-shadow:0 0 70px rgba(0,229,255,.30)}.login-card h2{font-family:'Orbitron';margin:0 0 6px;color:var(--cyan)}.login-card p{color:var(--muted);font-size:19px}.field{width:100%;padding:14px 16px;border-radius:14px;border:1px solid rgba(0,229,255,.30);background:rgba(255,255,255,.06);color:var(--text);font-size:20px;margin:8px 0 12px;font-family:'Rajdhani'}.admin-screen{display:none;position:fixed;inset:24px;z-index:90;background:rgba(5,10,24,.97);border:1px solid rgba(0,229,255,.35);border-radius:30px;padding:24px;overflow:auto;box-shadow:0 0 60px rgba(0,229,255,.28)}.admin-screen.active{display:block}.admin-top{display:flex;align-items:center;justify-content:space-between;gap:18px;margin-bottom:18px}.admin-title{font-family:'Orbitron';font-size:30px;color:var(--cyan);letter-spacing:2px}.badge{padding:7px 13px;border-radius:999px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);font-family:'Orbitron';font-size:12px}.badge.ok{color:var(--green);border-color:rgba(0,255,156,.45)}.badge.bad{color:var(--red);border-color:rgba(255,77,109,.45)}.admin-grid{display:grid;grid-template-columns:1.25fr .9fr;gap:18px}.card{background:var(--card);border:1px solid rgba(0,229,255,.18);border-radius:22px;padding:18px}.card h3{font-family:'Orbitron';font-size:17px;color:#bdf8ff;margin:0 0 14px}.metric-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}.metric{padding:16px;border-radius:18px;background:rgba(0,229,255,.055);border:1px solid rgba(0,229,255,.14)}.metric .label{color:var(--muted);font-size:15px}.metric .value{font-family:'Orbitron';font-size:28px;margin-top:5px}.split{display:grid;grid-template-columns:1fr 1fr;gap:18px}.bar{height:12px;background:rgba(255,255,255,.08);border-radius:999px;overflow:hidden;margin:7px 0 13px}.bar-fill{height:100%;background:linear-gradient(90deg,var(--purple),var(--cyan),var(--green));transition:width .6s}.action-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.btn{padding:12px 14px;border-radius:14px;border:1px solid rgba(0,229,255,.30);background:rgba(0,229,255,.10);color:#e8faff;font-family:'Rajdhani';font-size:17px;cursor:pointer;text-align:left}.btn:hover{box-shadow:0 0 18px rgba(0,229,255,.35)}.btn.primary{border-color:rgba(0,255,156,.45);background:rgba(0,255,156,.10)}.btn.danger{border-color:rgba(255,77,109,.50);background:rgba(255,77,109,.10)}.btn.small{text-align:center;padding:10px;font-size:15px}.log{font-family:monospace;color:#b8f7ff;line-height:1.55;max-height:290px;overflow:auto;font-size:12px}.mini{color:var(--muted);font-size:14px}.kv{display:grid;grid-template-columns:125px 1fr;gap:8px;font-size:18px;margin:5px 0}.kv span:first-child{color:var(--muted)}


/* === MECHANICAL ROUTING ANIMATION: PRO VERSION === */
.route-stage{--target-x:0px;--accent:var(--green);width:900px;height:390px;margin:12px auto 8px;position:relative;border-radius:24px;border:1px solid rgba(0,229,255,.28);background:linear-gradient(180deg,rgba(7,18,34,.92),rgba(3,8,18,.84));box-shadow:inset 0 0 70px rgba(0,229,255,.08),0 0 38px rgba(0,229,255,.14);overflow:hidden}
.route-stage:before{content:"";position:absolute;inset:0;background-image:linear-gradient(rgba(0,229,255,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,.045) 1px,transparent 1px);background-size:42px 42px;opacity:.42;animation:routeGrid 7s linear infinite}
.route-stage:after{content:"";position:absolute;left:78px;right:78px;top:168px;height:1px;background:linear-gradient(90deg,transparent,rgba(0,229,255,.72),rgba(0,255,156,.58),transparent);box-shadow:0 0 22px rgba(0,229,255,.42)}
.route-stage.target-metal{--target-x:-315px;--accent:#ffd166}.route-stage.target-glass{--target-x:-105px;--accent:#8b5cf6}.route-stage.target-paper{--target-x:105px;--accent:#00ff9c}.route-stage.target-plastic{--target-x:315px;--accent:#00e5ff}.route-stage.no-sort{--target-x:-315px;--accent:#ff4d6d}
@keyframes routeGrid{to{transform:translate(42px,42px)}}
.route-label{position:absolute;left:34px;top:24px;z-index:5;font-family:'Orbitron';font-size:13px;letter-spacing:1.8px;color:#dffaff;text-align:left;text-shadow:0 0 14px rgba(0,229,255,.45)}
.route-label:before{content:"ACTIVE ROUTE";display:block;margin-bottom:6px;font-size:10px;letter-spacing:1.4px;color:var(--muted);opacity:.85}
.route-status-strip{position:absolute;right:34px;top:28px;z-index:5;display:flex;gap:8px}.route-step{padding:7px 11px;border-radius:999px;border:1px solid rgba(0,229,255,.22);background:rgba(0,229,255,.055);font-family:'Orbitron';font-size:10px;color:#9fbacf;letter-spacing:1px}.route-step.active{color:var(--accent);border-color:color-mix(in srgb,var(--accent) 60%,transparent);box-shadow:0 0 16px color-mix(in srgb,var(--accent) 20%,transparent)}
.route-rail{position:absolute;left:82px;right:82px;top:120px;height:112px;border-radius:18px;border:1px solid rgba(0,229,255,.22);background:linear-gradient(180deg,rgba(255,255,255,.045),rgba(255,255,255,.012));box-shadow:inset 0 0 38px rgba(0,229,255,.045);overflow:hidden}.route-rail:before{content:"";position:absolute;left:26px;right:26px;top:51px;height:12px;border-radius:999px;background:repeating-linear-gradient(90deg,rgba(0,229,255,.18) 0 18px,rgba(0,255,156,.16) 18px 24px);box-shadow:0 0 24px rgba(0,229,255,.20);animation:railMove 1.1s linear infinite}.route-rail:after{content:"";position:absolute;left:28px;right:28px;bottom:22px;height:5px;border-radius:999px;background:linear-gradient(90deg,transparent,rgba(0,229,255,.62),rgba(0,255,156,.56),transparent);box-shadow:0 0 20px rgba(0,255,156,.25)}@keyframes railMove{to{transform:translateX(24px)}}
.route-target-beam{position:absolute;left:calc(50% - 78px);top:88px;width:156px;height:202px;transform:translateX(var(--target-x));border-left:1px solid color-mix(in srgb,var(--accent) 58%,transparent);border-right:1px solid color-mix(in srgb,var(--accent) 58%,transparent);background:linear-gradient(180deg,color-mix(in srgb,var(--accent) 10%,transparent),transparent 68%);box-shadow:0 0 28px color-mix(in srgb,var(--accent) 20%,transparent);opacity:.72;z-index:2}
.route-chamber{position:absolute;left:calc(50% - 78px);top:128px;width:156px;height:80px;z-index:6;animation:chamberRoute 8.2s cubic-bezier(.45,0,.18,1) infinite}.route-stage.no-sort .route-chamber{animation:chamberRoute 8.2s cubic-bezier(.45,0,.18,1) infinite}.route-chamber-shell{position:absolute;inset:0;border-radius:13px;border:2px solid color-mix(in srgb,var(--accent) 72%,rgba(0,229,255,.35));background:linear-gradient(145deg,rgba(0,229,255,.14),rgba(5,12,26,.88));box-shadow:0 0 28px color-mix(in srgb,var(--accent) 30%,transparent),inset 0 0 22px rgba(255,255,255,.075);overflow:hidden}.route-chamber-shell:before{content:"";position:absolute;left:-45%;top:10px;width:55%;height:18px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.22),transparent);transform:skewX(-20deg);animation:chamberShine 2.2s ease-in-out infinite}.route-hatch{position:absolute;left:38px;right:38px;bottom:-4px;height:9px;background:var(--accent);box-shadow:0 0 18px var(--accent);transform-origin:center top;animation:hatchOpen 8.2s ease-in-out infinite}.route-item{position:absolute;left:50%;top:44%;transform:translate(-50%,-50%);font-size:38px;line-height:1;filter:drop-shadow(0 0 10px color-mix(in srgb,var(--accent) 62%,transparent));animation:itemRelease 8.2s ease-in-out infinite}.route-drop{position:absolute;left:calc(50% - 2px);top:207px;width:4px;height:88px;background:linear-gradient(180deg,var(--accent),transparent);transform:translateX(var(--target-x));box-shadow:0 0 18px var(--accent);opacity:0;animation:dropBeam 8.2s ease-in-out infinite;z-index:4}
.route-bin-deck{position:absolute;left:68px;right:68px;bottom:24px;display:grid;grid-template-columns:repeat(4,1fr);gap:18px;z-index:5}.route-bin{height:62px;position:relative;border-radius:12px 12px 18px 18px;border:1px solid rgba(0,229,255,.22);background:linear-gradient(180deg,rgba(255,255,255,.045),rgba(0,0,0,.22));font-family:'Orbitron';font-size:11px;color:#c7f7ff;letter-spacing:1px;display:flex;align-items:flex-end;justify-content:center;text-align:center;padding:0 8px 12px;box-shadow:inset 0 -18px 32px rgba(0,229,255,.035)}.route-bin:before{content:"";position:absolute;left:14px;right:14px;top:8px;height:17px;border-radius:4px;background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.055);transform:skewX(-12deg)}.route-bin.active{color:var(--accent);border-color:color-mix(in srgb,var(--accent) 72%,transparent);background:linear-gradient(180deg,color-mix(in srgb,var(--accent) 10%,transparent),rgba(0,0,0,.18));box-shadow:0 0 22px color-mix(in srgb,var(--accent) 24%,transparent),inset 0 -18px 34px color-mix(in srgb,var(--accent) 7%,transparent)}.route-bin.active:after{content:"TARGET";position:absolute;top:-24px;left:0;right:0;text-align:center;font-size:9px;color:var(--accent);text-shadow:0 0 12px var(--accent);letter-spacing:1.2px}.route-reject-slot{display:none}.route-stage.no-sort .route-bin[data-bin="metal"]:after{content:"NO-SORT"}.route-stage.no-sort .route-bin[data-bin="metal"]{color:var(--red);border-color:rgba(255,77,109,.65);background:linear-gradient(180deg,rgba(255,77,109,.12),rgba(0,0,0,.18));box-shadow:0 0 22px rgba(255,77,109,.22),inset 0 -18px 34px rgba(255,77,109,.07)}
.route-timeline{position:absolute;left:86px;right:86px;bottom:103px;height:3px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden;z-index:7}.route-timeline span{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--green));box-shadow:0 0 14px rgba(0,255,156,.45);animation:routeProgress var(--ui-route-duration,15s) linear forwards}.route-stage.no-sort .route-timeline span{background:linear-gradient(90deg,var(--red),var(--yellow))}@keyframes routeProgress{to{width:100%}}@keyframes chamberRoute{0%,10%{transform:translateX(0)}34%,66%{transform:translateX(var(--target-x))}90%,100%{transform:translateX(0)}}@keyframes chamberShine{0%,100%{opacity:.18;left:-45%}55%{opacity:.9;left:85%}}@keyframes hatchOpen{0%,47%,72%,100%{transform:rotateX(0deg);opacity:.9}54%,64%{transform:rotateX(72deg);opacity:1}}@keyframes itemRelease{0%,45%{opacity:1;transform:translate(-50%,-50%) scale(1)}58%{opacity:1;transform:translate(-50%,104px) scale(.74)}66%,100%{opacity:0;transform:translate(-50%,124px) scale(.55)}}@keyframes dropBeam{0%,49%,70%,100%{opacity:0}55%,64%{opacity:.85}}
@media(max-width:900px){.admin-grid,.split,.metric-row{grid-template-columns:1fr}.action-grid{grid-template-columns:1fr}.admin-screen{inset:8px;padding:15px}.brand{font-size:34px}.main-title{font-size:38px}.main-text{font-size:22px}.panel{padding:28px}.scan-box{width:90vw;height:50vw}.metric .value{font-size:22px}}
/* === RESTORED ORIGINAL ECOVISION ANIMATIONS START === */
.panel:after{
  content:"";
  position:absolute;
  top:0;
  left:-70%;
  height:100%;
  width:45%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.08),transparent);
  transform:skewX(-18deg);
  animation:shine 5s ease-in-out infinite;
  pointer-events:none;
}

@keyframes shine{0%,68%{left:-70%}100%{left:130%}}

.brand{
  font-family:'Orbitron';
  font-size:48px;
  letter-spacing:4px;
  text-shadow:0 0 24px rgba(0,229,255,.95);
  margin-bottom:8px;
}

.subtitle{
  font-size:26px;
  color:var(--muted);
  margin-bottom:24px;
}

.main-title{
  font-family:'Orbitron';
  font-size:58px;
  margin:22px 0 12px;
  text-shadow:0 0 18px rgba(0,229,255,.28);
}

.main-text{
  font-size:30px;
  color:#d6fbff;
}

.status-pill{
  display:inline-block;
  margin-top:30px;
  padding:14px 34px;
  border-radius:999px;
  border:1px solid rgba(0,255,156,.5);
  color:var(--green);
  font-family:'Orbitron';
  font-size:15px;
  letter-spacing:1px;
  background:rgba(0,255,156,.08);
  box-shadow:0 0 28px rgba(0,255,156,.25);
  animation:pulse 1.8s ease-in-out infinite;
}

@keyframes pulse{50%{transform:scale(1.06);opacity:1}0%,100%{opacity:.72}}

.recycle-orb{
  width:245px;
  height:245px;
  border-radius:50%;
  margin:36px auto 26px;
  border:2px solid rgba(0,229,255,.5);
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:105px;
  box-shadow:
    0 0 50px rgba(0,229,255,.5),
    inset 0 0 70px rgba(0,255,156,.15);
  animation:orb 2.8s ease-in-out infinite;
}

@keyframes orb{50%{transform:scale(1.08)}}


/* ======================================================
   DOOR OPEN: EXACT SAME MOTION, CLEANER ARROW VISUAL
====================================================== */

.door-stage{
  width:780px;
  height:400px;
  margin:22px auto 8px;
  position:relative;
  perspective:900px;
}

.door-frame{
  position:absolute;
  inset:0;
  border-radius:34px;
  overflow:hidden;
  clip-path:inset(0 round 34px);
  contain:paint;
  isolation:isolate;
  border:2px solid rgba(0,229,255,.45);
  background:
    linear-gradient(145deg,rgba(0,229,255,.08),rgba(0,0,0,.55));
  box-shadow:
    inset 0 0 48px rgba(0,229,255,.16),
    0 0 44px rgba(0,229,255,.24);
}

.tunnel{
  position:absolute;
  inset:30px 70px 30px 70px;
  border-radius:30px;
  background:
    radial-gradient(ellipse at 50% 52%,rgba(0,255,156,.20),rgba(0,0,0,.70) 58%,rgba(0,0,0,.94));
  box-shadow:
    inset 0 0 80px rgba(0,255,156,.14),
    inset 0 0 30px rgba(0,229,255,.12);
  opacity:0;
  animation:tunnelReveal 4.2s ease-in-out infinite;
}

@keyframes tunnelReveal{
  0%,28%{opacity:.12;filter:blur(6px)}
  45%,100%{opacity:1;filter:blur(0)}
}

.door-panel{
  position:absolute;
  top:-2px;
  bottom:-2px;
  width:calc(50% + 3px);
  z-index:10;
  background:linear-gradient(135deg,rgba(0,229,255,.18),rgba(255,255,255,.04));
  border:1px solid rgba(0,229,255,.42);
  backdrop-filter:blur(10px);
  box-shadow:inset 0 0 24px rgba(0,229,255,.08);
}

.left-door{left:-2px;animation:doorOpenL 4.2s ease-in-out infinite}
.right-door{right:-2px;animation:doorOpenR 4.2s ease-in-out infinite}

@keyframes doorOpenL{
  0%,25%{transform:translateX(0)}
  55%,100%{transform:translateX(-103%)}
}

@keyframes doorOpenR{
  0%,25%{transform:translateX(0)}
  55%,100%{transform:translateX(103%)}
}

.floor-glow{
  position:absolute;
  bottom:0;
  left:10%;
  width:80%;
  height:14px;
  background:linear-gradient(90deg,transparent,rgba(0,255,156,.90),transparent);
  box-shadow:
    0 0 30px rgba(0,255,156,.85),
    0 0 70px rgba(0,255,156,.50);
  opacity:0;
  animation:floorGlow 4.2s ease-in-out infinite;
}

@keyframes floorGlow{
  0%,42%{opacity:0}
  55%,100%{opacity:1}
}

.depth-ring{
  position:absolute;
  left:50%;
  top:70%;
  border-radius:50%;
  border:2px solid rgba(0,255,156,.22);
  transform:translate(-50%,-50%) rotateX(72deg);
  opacity:0;
  box-shadow:0 0 18px rgba(0,255,156,.24);
  animation:ringIn 1.6s ease-in-out infinite;
}

.depth-ring.r1{width:470px;height:170px;animation-delay:2.15s}
.depth-ring.r2{width:320px;height:115px;animation-delay:2.45s}
.depth-ring.r3{width:180px;height:64px;animation-delay:2.75s}

@keyframes ringIn{
  0%,15%{opacity:0;transform:translate(-50%,-50%) rotateX(72deg) scale(1.08)}
  35%{opacity:1}
  100%{opacity:0;transform:translate(-50%,-50%) rotateX(72deg) scale(.46)}
}

.arrow-depth-stack{
  position:absolute;
  inset:0;
  z-index:7;
  opacity:0;
  animation:arrowStackDelay 4.2s ease-in-out infinite;
  pointer-events:none;
}

@keyframes arrowStackDelay{
  0%,50%{opacity:0}
  59%,100%{opacity:1}
}

/* Same movement as your uploaded version, but the solid green box arrow is replaced
   with a softer neon glass arrowhead so it matches the chamber lighting. */
.depth-arrow{
  position:absolute;
  left:50%;
  bottom:42px;
  width:126px;
  height:86px;
  opacity:0;
  transform:translateX(-50%) rotateX(68deg) scale(1.38);
  animation:depthArrowMove 1.75s ease-in-out infinite;
  filter:
    drop-shadow(0 0 12px rgba(0,255,156,.88))
    drop-shadow(0 0 32px rgba(0,255,156,.48));
}

.depth-arrow:before{
  content:"";
  position:absolute;
  inset:0;
  background:
    linear-gradient(180deg,rgba(0,255,156,.95),rgba(0,229,255,.40));
  clip-path:polygon(50% 0%, 100% 52%, 78% 52%, 78% 76%, 22% 76%, 22% 52%, 0% 52%);
  opacity:.72;
}

.depth-arrow:after{
  content:"";
  position:absolute;
  inset:7px 13px 13px 13px;
  background:
    linear-gradient(180deg,rgba(9,20,35,.88),rgba(9,20,35,.35));
  clip-path:polygon(50% 2%, 96% 50%, 72% 50%, 72% 67%, 28% 67%, 28% 50%, 4% 50%);
  opacity:.65;
  filter:blur(.2px);
}

.depth-arrow.a2{animation-delay:.32s}
.depth-arrow.a3{animation-delay:.64s}

@keyframes depthArrowMove{
  0%{
    opacity:0;
    bottom:35px;
    transform:translateX(-50%) rotateX(68deg) scale(1.45);
  }
  18%{opacity:1}
  60%{opacity:.95}
  100%{
    opacity:0;
    bottom:235px;
    transform:translateX(-50%) rotateX(68deg) scale(.30);
  }
}

.chamber-side-line{
  position:absolute;
  top:70px;
  bottom:70px;
  width:2px;
  background:linear-gradient(transparent,rgba(0,255,156,.45),transparent);
  opacity:0;
  animation:sideLine 4.2s ease-in-out infinite;
}

.chamber-side-line.left{left:125px}
.chamber-side-line.right{right:125px}

@keyframes sideLine{
  0%,45%{opacity:0}
  60%,100%{opacity:1}
}

/* ======================================================
   DOOR CLOSE
====================================================== */

.close-frame{
  width:720px;
  height:360px;
  margin:32px auto 16px;
  position:relative;
  overflow:hidden;
  clip-path:inset(0 round 34px);
  border-radius:34px;
  border:2px solid rgba(0,229,255,.45);
  background:rgba(0,0,0,.36);
  box-shadow:
    inset 0 0 45px rgba(0,229,255,.15),
    0 0 40px rgba(0,229,255,.25);
}

.close-door-l,.close-door-r{
  position:absolute;
  top:-2px;
  bottom:-2px;
  width:calc(50% + 3px);
  background:linear-gradient(135deg,rgba(0,229,255,.18),rgba(255,255,255,.04));
  border:1px solid rgba(0,229,255,.42);
}

.close-door-l{left:-2px;animation:closeL 2.5s ease-in-out infinite}
.close-door-r{right:-2px;animation:closeR 2.5s ease-in-out infinite}

@keyframes closeL{from{transform:translateX(-103%)}60%,to{transform:translateX(0)}}
@keyframes closeR{from{transform:translateX(103%)}60%,to{transform:translateX(0)}}

/* ======================================================
   PROCESSING: HELIX CANVAS
====================================================== */

.processing-panel .main-title{margin-top:16px}

.processing-visual{
  width:840px;
  height:360px;
  margin:14px auto 8px;
  position:relative;
  overflow:hidden;
  border-radius:30px;
  contain:paint;
  isolation:isolate;
  transform:translateZ(0);
  background-clip:padding-box;
  border:1px solid rgba(0,229,255,.25);
  background:
    radial-gradient(circle at 50% 45%,rgba(0,229,255,.10),transparent 48%),
    linear-gradient(180deg,rgba(0,229,255,.04),rgba(0,0,0,.28));
  box-shadow:
    inset 0 0 58px rgba(0,229,255,.06),
    0 0 32px rgba(0,229,255,.12);
}

#dnaCanvas{
  position:absolute;
  inset:0;
  width:100%;
  height:100%;
  z-index:2;
  display:block;
  background:transparent;
  backface-visibility:hidden;
  transform:translateZ(0);
}

.processing-visual:before{
  content:"";
  position:absolute;
  inset:0;
  background-image:
    linear-gradient(rgba(0,229,255,.055) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,229,255,.055) 1px,transparent 1px);
  background-size:34px 34px;
  animation:processingGrid 6s linear infinite;
  opacity:.5;
  z-index:1;
}

@keyframes processingGrid{to{transform:translate(34px,34px)}}

.processing-chip-row{
  position:absolute;
  left:50%;
  bottom:18px;
  transform:translateX(-50%);
  display:flex;
  gap:16px;
  z-index:5;
}

.processing-chip{
  padding:9px 17px;
  border-radius:999px;
  border:1px solid rgba(0,229,255,.38);
  background:rgba(0,229,255,.065);
  font-family:'Orbitron';
  font-size:12px;
  letter-spacing:1px;
  color:#c9f7ff;
  animation:chipPulse 2s ease-in-out infinite;
}
.processing-chip:nth-child(2){animation-delay:.35s}
.processing-chip:nth-child(3){animation-delay:.70s}
@keyframes chipPulse{0%,100%{opacity:.48;box-shadow:none}50%{opacity:1;color:var(--green);border-color:var(--green);box-shadow:0 0 18px rgba(0,255,156,.32)}}

/* ======================================================
   CLASSIFYING
====================================================== */

.scan-box{
  width:680px;
  height:385px;
  margin:24px auto 20px;
  border-radius:30px;
  position:relative;
  overflow:hidden;
  border:1px solid rgba(0,229,255,.45);
  background:radial-gradient(circle,rgba(0,229,255,.1),rgba(0,0,0,.55));
  box-shadow:0 0 40px rgba(0,229,255,.25);
}
.mock-object{display:none!important}
.scan-box img{width:100%;height:100%;object-fit:cover;opacity:.86;position:absolute;inset:0;z-index:1}
.scan-line{position:absolute;left:0;right:0;height:5px;background:var(--cyan);box-shadow:0 0 30px var(--cyan);animation:scanLine 1.6s linear infinite;z-index:5}
@keyframes scanLine{from{top:-5%;opacity:0}20%{opacity:1}to{top:105%;opacity:0}}
.analysis-grid{position:absolute;inset:0;z-index:3;mix-blend-mode:screen;background-image:linear-gradient(rgba(0,229,255,.15) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,.15) 1px,transparent 1px);background-size:34px 34px;opacity:.42;animation:imageGrid 3s linear infinite}
@keyframes imageGrid{to{transform:translate(34px,34px)}}
.pixel{position:absolute;width:14px;height:14px;z-index:4;border:1px solid rgba(0,229,255,.9);background:rgba(0,229,255,.18);box-shadow:0 0 12px rgba(0,229,255,.75);opacity:0;animation:pixelScan 3.1s linear infinite}
.p1{left:8%;top:18%}.p2{left:22%;top:68%;animation-delay:.2s}.p3{left:38%;top:30%;animation-delay:.4s}.p4{left:57%;top:76%;animation-delay:.6s}.p5{left:74%;top:24%;animation-delay:.8s}.p6{left:88%;top:58%;animation-delay:1s}.p7{left:14%;top:45%;animation-delay:1.2s}.p8{left:46%;top:58%;animation-delay:1.4s}.p9{left:66%;top:42%;animation-delay:1.6s}.p10{left:82%;top:12%;animation-delay:1.8s}
@keyframes pixelScan{0%{opacity:0;transform:translate(-18px,-18px) scale(.65)}15%{opacity:1}45%{opacity:.9;transform:translate(20px,16px) scale(1.1)}100%{opacity:0;transform:translate(28px,-14px) scale(.65)}}

.material-icon{width:190px;height:190px;margin:30px auto 18px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:92px;background:radial-gradient(circle,rgba(0,255,156,.26),rgba(0,0,0,.26));border:2px solid rgba(0,255,156,.48);box-shadow:0 0 55px rgba(0,255,156,.36),inset 0 0 38px rgba(255,255,255,.04)}
.material{font-family:'Orbitron';font-size:72px;color:var(--green);text-shadow:0 0 32px rgba(0,255,156,.8);margin-top:14px}
.confidence{font-size:29px;color:#d6fbff;margin-top:12px}
.item-number{margin-top:20px;font-family:'Orbitron';color:var(--yellow);font-size:25px}
.thank{font-size:116px;text-shadow:0 0 35px var(--cyan)}

/* === RESTORED ORIGINAL ECOVISION ANIMATIONS END === */

.bin-visual-grid{display:grid;grid-template-columns:repeat(4,minmax(72px,1fr));gap:12px;align-items:end;margin-top:8px}.bin3d{text-align:center;position:relative;min-width:0}.bin3d-stage{height:155px;display:flex;align-items:flex-end;justify-content:center;perspective:520px}.bin3d-body{position:relative;width:72px;height:122px;border-radius:14px 14px 18px 18px;border:1px solid rgba(0,229,255,.42);background:linear-gradient(160deg,rgba(0,229,255,.10),rgba(255,255,255,.025));box-shadow:inset 0 0 22px rgba(0,229,255,.16),0 0 22px rgba(0,229,255,.13);overflow:hidden;transform:rotateX(8deg) rotateY(-10deg);transform-style:preserve-3d}.bin3d-body:before{content:"";position:absolute;left:9px;right:9px;top:-8px;height:17px;border-radius:50%;border:1px solid rgba(0,229,255,.45);background:radial-gradient(ellipse at center,rgba(0,229,255,.22),rgba(0,0,0,.10));box-shadow:0 0 16px rgba(0,229,255,.24);z-index:5}.bin3d-fill{position:absolute;left:4px;right:4px;bottom:4px;height:0;border-radius:10px 10px 15px 15px;background:linear-gradient(180deg,rgba(0,229,255,.92),rgba(0,103,255,.52));box-shadow:0 0 24px rgba(0,229,255,.55),inset 0 0 20px rgba(255,255,255,.20);transition:height .9s cubic-bezier(.2,.8,.2,1);overflow:hidden}.bin3d-fill:before{content:"";position:absolute;left:-30%;top:-9px;width:160%;height:20px;border-radius:50%;background:rgba(168,245,255,.78);filter:blur(.2px);animation:binWave 2.2s ease-in-out infinite}.bin3d-fill:after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.16),transparent);animation:binShimmer 2.8s linear infinite}.bin3d-shine{position:absolute;top:13px;left:12px;width:12px;height:82%;border-radius:999px;background:linear-gradient(180deg,rgba(255,255,255,.45),rgba(255,255,255,0));opacity:.32;z-index:6}.bin3d-percent{font-family:'Orbitron';font-size:18px;color:#e8faff;margin-top:7px;text-shadow:0 0 14px rgba(0,229,255,.65)}.bin3d-name{font-family:'Orbitron';font-size:11px;letter-spacing:1px;color:#a5f3fc;margin-top:8px;white-space:nowrap}.bin3d-status{display:inline-block;margin-top:6px;padding:3px 8px;border-radius:999px;font-family:'Orbitron';font-size:10px;border:1px solid rgba(0,255,156,.38);color:var(--green);background:rgba(0,255,156,.08)}.bin3d.warning .bin3d-body{border-color:rgba(255,209,102,.7);box-shadow:inset 0 0 22px rgba(255,209,102,.12),0 0 24px rgba(255,209,102,.22)}.bin3d.warning .bin3d-status{border-color:rgba(255,209,102,.65);color:var(--yellow);background:rgba(255,209,102,.10)}.bin3d.full .bin3d-body{border-color:rgba(255,77,109,.75);box-shadow:inset 0 0 22px rgba(255,77,109,.14),0 0 28px rgba(255,77,109,.32);animation:fullPulse 1.25s ease-in-out infinite}.bin3d.full .bin3d-status{border-color:rgba(255,77,109,.65);color:var(--red);background:rgba(255,77,109,.10)}.bin3d.fault .bin3d-body{filter:grayscale(.75);opacity:.58}.bin3d.fault:after{content:"⚠";position:absolute;top:38px;left:50%;transform:translateX(-50%);font-size:26px;text-shadow:0 0 16px rgba(255,77,109,.85);z-index:8}.share-bars{margin-top:12px}.share-row{margin:0 0 13px}.share-top{display:flex;justify-content:space-between;align-items:center;font-size:15px}.share-top b{font-family:'Orbitron';font-size:13px;color:#e8faff}.share-meta{color:var(--muted);font-size:13px}.share-track{height:12px;border-radius:999px;background:rgba(255,255,255,.075);overflow:hidden;margin-top:6px;border:1px solid rgba(255,255,255,.06)}.share-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--purple),var(--cyan),var(--green));box-shadow:0 0 16px rgba(0,229,255,.38);transition:width .75s ease}.share-empty{padding:16px;border-radius:16px;border:1px dashed rgba(0,229,255,.24);color:var(--muted);text-align:center}.admin-viz-split{grid-template-columns:1.28fr .72fr}.bin-viz-card{min-height:262px}@keyframes binWave{0%,100%{transform:translateX(-4%) skewX(-10deg)}50%{transform:translateX(4%) skewX(10deg)}}@keyframes binShimmer{from{transform:translateX(-115%)}to{transform:translateX(115%)}}@keyframes fullPulse{50%{box-shadow:inset 0 0 28px rgba(255,77,109,.22),0 0 38px rgba(255,77,109,.48)}}

/* ======================================================
   ADMIN REDESIGN: DARK OPERATIONS CONSOLE
   Professional sidebar layout inspired by product-ops dashboards
====================================================== */
.admin-entry{
  width:auto !important;
  height:auto !important;
  min-width:84px;
  padding:13px 18px !important;
  border-radius:999px !important;
  font-family:'Orbitron',sans-serif !important;
  font-size:11px !important;
  letter-spacing:1.4px;
  border:1px solid rgba(118,242,255,.36) !important;
  background:linear-gradient(180deg,rgba(10,24,42,.82),rgba(6,13,26,.72)) !important;
  color:#dffaff !important;
  box-shadow:0 14px 34px rgba(0,0,0,.26),0 0 24px rgba(0,229,255,.14) !important;
}
.admin-entry:hover{
  transform:translateY(-2px) !important;
  box-shadow:0 18px 42px rgba(0,0,0,.34),0 0 32px rgba(0,229,255,.28) !important;
}
.admin-screen{
  display:none;
  position:fixed;
  inset:18px;
  z-index:90;
  padding:0;
  overflow:hidden;
  border-radius:32px;
  background:
    radial-gradient(circle at 18% 0%,rgba(0,229,255,.13),transparent 34%),
    radial-gradient(circle at 88% 12%,rgba(0,255,156,.08),transparent 30%),
    linear-gradient(135deg,rgba(5,12,24,.98),rgba(3,8,18,.985));
  border:1px solid rgba(120,226,255,.25);
  box-shadow:0 26px 90px rgba(0,0,0,.55),inset 0 0 80px rgba(0,229,255,.035);
}
.admin-screen.active{display:grid;grid-template-columns:268px minmax(0,1fr)}
.admin-sidebar{
  position:relative;
  padding:26px 18px;
  border-right:1px solid rgba(120,226,255,.14);
  background:linear-gradient(180deg,rgba(8,20,38,.72),rgba(4,10,22,.82));
  overflow:hidden;
}
.admin-sidebar:before{
  content:"";
  position:absolute;
  inset:0;
  background-image:linear-gradient(rgba(0,229,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,.035) 1px,transparent 1px);
  background-size:34px 34px;
  opacity:.55;
  pointer-events:none;
}
.admin-sidebar > *{position:relative;z-index:2}
.admin-brand-block{padding:4px 10px 20px}
.admin-brand-name{
  font-family:'Orbitron',sans-serif;
  font-size:25px;
  letter-spacing:1.6px;
  color:#f3fdff;
  text-shadow:0 0 20px rgba(0,229,255,.35);
}
.admin-brand-sub{
  margin-top:7px;
  color:#90a9bb;
  font-size:14px;
  letter-spacing:.3px;
}
.admin-side-status{display:grid;grid-template-columns:1fr;gap:8px;margin:4px 8px 20px}
.admin-nav{
  display:flex;
  flex-direction:column;
  gap:8px;
  margin-top:6px;
}
.admin-nav-btn{
  width:100%;
  display:flex;
  align-items:center;
  gap:12px;
  min-height:44px;
  padding:12px 13px;
  border-radius:14px;
  border:1px solid transparent;
  background:transparent;
  color:#9fb8c8;
  font-family:'Rajdhani',sans-serif;
  font-size:17px;
  font-weight:700;
  letter-spacing:.2px;
  cursor:pointer;
  text-align:left;
  transition:.18s ease;
}
.admin-nav-btn:hover{
  color:#ecfeff;
  background:rgba(255,255,255,.045);
  border-color:rgba(120,226,255,.12);
}
.admin-nav-btn.active{
  color:#ecfeff;
  background:linear-gradient(90deg,rgba(0,255,156,.13),rgba(0,229,255,.075));
  border-color:rgba(0,255,156,.34);
  box-shadow:0 0 20px rgba(0,255,156,.10),inset 0 0 18px rgba(0,229,255,.035);
}
.nav-glyph{
  width:25px;
  height:25px;
  flex:0 0 auto;
  border-radius:9px;
  border:1px solid rgba(120,226,255,.22);
  background:rgba(0,229,255,.065);
  position:relative;
}
.admin-nav-btn.active .nav-glyph{
  border-color:rgba(0,255,156,.52);
  background:rgba(0,255,156,.12);
  box-shadow:0 0 15px rgba(0,255,156,.20);
}
.nav-glyph:after{
  content:"";
  position:absolute;
  inset:7px;
  border-radius:50%;
  background:currentColor;
  opacity:.85;
}
.admin-sidebar-footer{
  position:absolute;
  left:18px;
  right:18px;
  bottom:18px;
  z-index:3;
}
.admin-main{
  min-width:0;
  overflow:auto;
  padding:26px;
}
.admin-main::-webkit-scrollbar,.admin-log-stream::-webkit-scrollbar{width:10px}
.admin-main::-webkit-scrollbar-thumb,.admin-log-stream::-webkit-scrollbar-thumb{background:rgba(120,226,255,.18);border-radius:999px}
.admin-header{
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  gap:18px;
  margin-bottom:18px;
}
.admin-title-kicker{
  font-family:'Orbitron',sans-serif;
  font-size:11px;
  color:#78e2ff;
  letter-spacing:1.7px;
  text-transform:uppercase;
}
.admin-heading{
  margin-top:8px;
  font-family:'Orbitron',sans-serif;
  font-size:30px;
  color:#f4fcff;
  letter-spacing:1.2px;
}
.admin-header-copy{
  margin-top:8px;
  color:#90a9bb;
  font-size:17px;
}
.admin-header-actions{display:flex;align-items:center;gap:10px;flex-wrap:wrap;justify-content:flex-end}
.admin-section{display:none;animation:adminFade .22s ease forwards}
.admin-section.active{display:block}
@keyframes adminFade{from{opacity:.35;transform:translateY(5px)}to{opacity:1;transform:none}}
.admin-stat-grid{
  display:grid;
  grid-template-columns:repeat(4,minmax(150px,1fr));
  gap:14px;
  margin-bottom:18px;
}
.admin-stat-card{
  min-height:112px;
  padding:18px;
  border-radius:20px;
  border:1px solid rgba(120,226,255,.16);
  background:linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.025));
  box-shadow:0 16px 44px rgba(0,0,0,.18),inset 0 0 28px rgba(0,229,255,.025);
}
.admin-stat-label{
  color:#8da6b8;
  font-size:14px;
  letter-spacing:.4px;
  font-weight:700;
}
.admin-stat-value{
  margin-top:10px;
  font-family:'Orbitron',sans-serif;
  font-size:31px;
  color:#f2fcff;
  letter-spacing:.6px;
}
.admin-stat-note{
  margin-top:8px;
  font-size:13px;
  color:#5eead4;
}
.admin-layout-2{
  display:grid;
  grid-template-columns:minmax(0,1.25fr) minmax(340px,.75fr);
  gap:18px;
}
.admin-layout-even{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
.admin-screen .card{
  background:linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.025));
  border:1px solid rgba(120,226,255,.16);
  border-radius:22px;
  padding:20px;
  box-shadow:0 16px 44px rgba(0,0,0,.18),inset 0 0 30px rgba(0,229,255,.022);
}
.admin-screen .card h3{
  font-family:'Orbitron',sans-serif;
  font-size:15px;
  color:#e6fbff;
  letter-spacing:1px;
  margin:0 0 8px;
}
.admin-screen .mini{color:#8fa8bb;font-size:14px;line-height:1.45}
.admin-screen .btn{
  min-height:44px;
  padding:12px 14px;
  border-radius:13px;
  border:1px solid rgba(120,226,255,.18);
  background:linear-gradient(180deg,rgba(120,226,255,.075),rgba(255,255,255,.025));
  color:#e6fbff;
  font-family:'Rajdhani',sans-serif;
  font-size:16px;
  font-weight:800;
  letter-spacing:.25px;
  text-align:left;
  cursor:pointer;
  box-shadow:none;
  transition:.18s ease;
}
.admin-screen .btn:hover{
  transform:translateY(-1px);
  border-color:rgba(120,226,255,.38);
  box-shadow:0 10px 28px rgba(0,229,255,.08);
}
.admin-screen .btn.primary{
  border-color:rgba(0,255,156,.32);
  background:linear-gradient(180deg,rgba(0,255,156,.12),rgba(0,255,156,.045));
}
.admin-screen .btn.danger{
  border-color:rgba(255,77,109,.38);
  background:linear-gradient(180deg,rgba(255,77,109,.12),rgba(255,77,109,.045));
}
.admin-screen .btn.small{
  min-height:36px;
  padding:9px 13px;
  text-align:center;
  font-size:14px;
}
.admin-action-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
.admin-action-grid.single{grid-template-columns:1fr}
.badge{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  gap:7px;
  padding:8px 12px;
  border-radius:999px;
  background:rgba(255,255,255,.055);
  border:1px solid rgba(255,255,255,.12);
  font-family:'Orbitron',sans-serif;
  font-size:11px;
  letter-spacing:.9px;
  color:#b8d0df;
}
.badge:before{
  content:"";
  width:7px;
  height:7px;
  border-radius:50%;
  background:#94a3b8;
  box-shadow:0 0 12px rgba(148,163,184,.4);
}
.badge.ok{color:#7fffd2;border-color:rgba(0,255,156,.38);background:rgba(0,255,156,.075)}
.badge.ok:before{background:#00ff9c;box-shadow:0 0 12px rgba(0,255,156,.55)}
.badge.bad{color:#ff8aa0;border-color:rgba(255,77,109,.40);background:rgba(255,77,109,.075)}
.badge.bad:before{background:#ff4d6d;box-shadow:0 0 12px rgba(255,77,109,.55)}
.admin-preview{
  position:relative;
  min-height:298px;
  overflow:hidden;
  border-radius:22px;
  border:1px solid rgba(120,226,255,.16);
  background:radial-gradient(circle at 50% 30%,rgba(0,229,255,.10),rgba(0,0,0,.28) 60%);
}
.admin-preview img{
  position:absolute;
  inset:0;
  width:100%;
  height:100%;
  object-fit:cover;
  opacity:.78;
}
.admin-preview-overlay{
  position:absolute;
  inset:0;
  background:linear-gradient(180deg,rgba(3,8,18,.08),rgba(3,8,18,.78));
}
.admin-preview-caption{
  position:absolute;
  left:18px;
  right:18px;
  bottom:16px;
}
.admin-preview-caption b{
  display:block;
  font-family:'Orbitron';
  font-size:14px;
  color:#e6fbff;
  letter-spacing:1px;
}
.admin-log-stream{
  height:520px;
  overflow:auto;
  padding:10px 2px 2px;
  font-family:ui-monospace,SFMono-Regular,Consolas,monospace;
  color:#c8f7ff;
  line-height:1.7;
  font-size:13px;
}
.admin-log-stream div{
  padding:10px 12px;
  margin-bottom:8px;
  border-radius:12px;
  border:1px solid rgba(120,226,255,.10);
  background:rgba(120,226,255,.035);
}
.admin-screen .kv{
  display:grid;
  grid-template-columns:132px minmax(0,1fr);
  gap:10px;
  align-items:start;
  font-size:18px;
  margin:8px 0;
}
.admin-screen .kv span:first-child{color:#8fa8bb}
.admin-screen .kv b{
  color:#eafcff;
  overflow-wrap:anywhere;
  font-weight:800;
}
.admin-panel-stack{display:grid;gap:18px}
.admin-screen .bin3d.fault:after{content:"!";font-family:'Orbitron';color:#ff4d6d}
@media(max-width:1050px){
  .admin-screen.active{grid-template-columns:1fr}
  .admin-sidebar{border-right:0;border-bottom:1px solid rgba(120,226,255,.14);padding:18px}
  .admin-nav{display:grid;grid-template-columns:repeat(3,1fr)}
  .admin-sidebar-footer{position:relative;left:auto;right:auto;bottom:auto;margin-top:14px}
  .admin-main{padding:18px}
  .admin-layout-2,.admin-layout-even,.admin-stat-grid{grid-template-columns:1fr}
}
@media(max-width:620px){
  .admin-screen{inset:8px;border-radius:22px}
  .admin-nav{grid-template-columns:1fr 1fr}
  .admin-action-grid{grid-template-columns:1fr}
  .admin-header{display:block}
  .admin-header-actions{justify-content:flex-start;margin-top:14px}
}


</style></head><body>
<button class="settings admin-entry" onclick="openAdminLogin()" type="button"><span>ADMIN</span></button>
<section id="idlePage" class="page active"><div class="panel"><div class="brand">EcoVision</div><div class="subtitle">Intelligent Recycling Experience</div><div class="recycle-orb">♻</div><div class="main-title">Recycle with Confidence</div><div class="main-text">A cleaner way to sort everyday waste.</div><div class="status-pill">Approach to begin</div></div></section>
<section id="door_openPage" class="page">
  <div class="panel">
    <div class="brand">EcoVision</div>
    <div class="subtitle">Access opening</div>
    <div class="door-stage">
      <div class="door-frame">
        <div class="tunnel"></div>
        <div class="depth-ring r1"></div>
        <div class="depth-ring r2"></div>
        <div class="depth-ring r3"></div>
        <div class="chamber-side-line left"></div>
        <div class="chamber-side-line right"></div>
        <div class="arrow-depth-stack">
          <div class="depth-arrow a1"></div>
          <div class="depth-arrow a2"></div>
          <div class="depth-arrow a3"></div>
        </div>
        <div class="floor-glow"></div>
        <div class="door-panel left-door"></div>
        <div class="door-panel right-door"></div>
      </div>
    </div>
    <div class="main-title">Insert Recyclable</div>
    <div class="main-text">Place one item into the illuminated chamber.</div>
  </div>
</section>
<section id="door_closePage" class="page">
  <div class="panel">
    <div class="brand">EcoVision</div>
    <div class="subtitle">Item received</div>
    <div class="close-frame"><div class="close-door-l"></div><div class="close-door-r"></div></div>
    <div class="main-title">Chamber Closing</div>
    <div class="main-text">Please stand clear from the door area.</div>
  </div>
</section>
<section id="processingPage" class="page">
  <div class="panel processing-panel">
    <div class="brand">EcoVision</div>
    <div class="subtitle">Processing item profile</div>
    <div class="processing-visual">
      <canvas id="dnaCanvas"></canvas>
      <div class="processing-chip-row">
        <div class="processing-chip">CAPTURE</div>
        <div class="processing-chip">MATERIAL PROFILE</div>
        <div class="processing-chip">COMPOSITION MAP</div>
      </div>
    </div>
    <div class="main-title">Processing</div>
    <div class="main-text">Mapping the item signature.</div>
  </div>
</section>
<section id="classifyingPage" class="page">
  <div class="panel">
    <div class="brand">EcoVision</div>
    <div class="subtitle">Understanding material</div>
    <div class="scan-box">
      
      <img id="scanImage" src="/captures/latest.jpg" onerror="this.style.display='none'">
      <div class="analysis-grid"></div>
      <div class="scan-line"></div>
      <div class="pixel p1"></div><div class="pixel p2"></div><div class="pixel p3"></div><div class="pixel p4"></div><div class="pixel p5"></div>
      <div class="pixel p6"></div><div class="pixel p7"></div><div class="pixel p8"></div><div class="pixel p9"></div><div class="pixel p10"></div>
    </div>
    <div class="main-title">Identifying Material</div>
    <div class="main-text" id="processingFact">Analysing surface and material signals...</div>
  </div>
</section>
<section id="resultPage" class="page"><div class="panel"><div class="brand">EcoVision</div><div class="subtitle">Classification complete</div><div class="material-icon" id="materialIcon">🥤</div><div class="material" id="finalPrediction">PLASTIC</div><div class="confidence" id="confidenceText">Directed to plastic recovery stream</div><div class="item-number" id="itemNumber">Item #0 today</div><div class="status-pill" id="confidencePill">98% confidence</div></div></section>
<section id="thank_youPage" class="page"><div class="panel"><div class="brand">EcoVision</div><div class="subtitle">Thank you for recycling</div><div class="recycle-orb thank">✓</div><div class="main-title">Sorted Successfully</div><div class="main-text" id="thankFact">Your action makes a big difference.</div></div></section>
<section id="sortingPage" class="page"><div class="panel"><div class="brand">EcoVision</div><div class="subtitle">Routing item</div><div class="route-stage target-paper" id="routeStage"><div class="route-label" id="routeTargetBin">MECHANICAL ROUTING</div><div class="route-status-strip"><span class="route-step active">MOVE</span><span class="route-step active">DROP</span><span class="route-step active">RETURN</span></div><div class="route-rail"></div><div class="route-target-beam"></div><div class="route-drop"></div><div class="route-chamber"><div class="route-chamber-shell"><div class="route-item" id="routeTrashIcon">♻</div><div class="route-hatch"></div></div></div><div class="route-timeline"><span id="routeProgressBar"></span></div><div class="route-bin-deck"><div class="route-bin" data-bin="metal">METAL / NO-SORT</div><div class="route-bin" data-bin="glass">GLASS</div><div class="route-bin" data-bin="paper">PAPER</div><div class="route-bin" data-bin="plastic">PLASTIC</div></div><div class="route-reject-slot">NO-SORT DROP ZONE</div></div><div class="main-title">Transporting Item</div><div class="main-text" id="routeMessage">Chamber is moving to the selected recovery stream.</div></div></section>
<section id="out_of_servicePage" class="page"><div class="panel"><div class="brand">EcoVision</div><div class="subtitle">Service required</div><div class="recycle-orb">🚫</div><div class="main-title red">Out of Service</div><div class="main-text" id="outServiceText">This bin is temporarily unavailable.</div><div class="status-pill">Maintenance required</div></div></section>
<div id="loginModal" class="modal"><div class="login-card"><h2>Admin Access</h2><p>Enter operator credentials to view diagnostics and controls.</p><input id="loginUser" class="field" placeholder="Username"><input id="loginPass" class="field" placeholder="Password" type="password"><button class="btn primary" style="width:100%;text-align:center" onclick="adminLogin()">Unlock Admin Panel</button><button class="btn small" style="width:100%;margin-top:10px" onclick="closeAdminLogin()">Cancel</button><p id="loginMsg" class="mini"></p></div></div>

<div id="adminScreen" class="admin-screen">
  <aside class="admin-sidebar">
    <div class="admin-brand-block">
      <div class="admin-brand-name">EcoVision</div>
      <div class="admin-brand-sub">Operations Console</div>
    </div>

    <div class="admin-side-status">
      <span id="serviceBadge" class="badge">SERVICE</span>
      <span id="piBadge" class="badge">PI LINK</span>
    </div>

    <nav class="admin-nav" aria-label="Admin sections">
      <button class="admin-nav-btn active" type="button" data-admin-tab="overview" onclick="setAdminTab('overview')"><span class="nav-glyph"></span>Overview</button>
      <button class="admin-nav-btn" type="button" data-admin-tab="devices" onclick="setAdminTab('devices')"><span class="nav-glyph"></span>Devices</button>
      <button class="admin-nav-btn" type="button" data-admin-tab="analytics" onclick="setAdminTab('analytics')"><span class="nav-glyph"></span>Analytics</button>
      <button class="admin-nav-btn" type="button" data-admin-tab="logs" onclick="setAdminTab('logs')"><span class="nav-glyph"></span>Activity Logs</button>
      <button class="admin-nav-btn" type="button" data-admin-tab="alerts" onclick="setAdminTab('alerts')"><span class="nav-glyph"></span>Alerts</button>
      <button class="admin-nav-btn" type="button" data-admin-tab="reports" onclick="setAdminTab('reports')"><span class="nav-glyph"></span>Reports</button>
      <button class="admin-nav-btn" type="button" data-admin-tab="controls" onclick="setAdminTab('controls')"><span class="nav-glyph"></span>Controls</button>
    </nav>

    <div class="admin-sidebar-footer">
      <button class="btn small" type="button" style="width:100%" onclick="adminLogout()">Lock Console</button>
    </div>
  </aside>

  <main class="admin-main">
    <div class="admin-header">
      <div>
        <div class="admin-title-kicker">Smart Recycling Bin</div>
        <div class="admin-heading">Operations Monitor</div>
        <div class="admin-header-copy">Clean system view for maintenance, recovery actions, diagnostics, and daily data export.</div>
      </div>
      <div class="admin-header-actions">
        <button class="btn small" type="button" onclick="pollStatus()">Refresh</button>
        <button class="btn small" type="button" onclick="adminLogout()">Lock</button>
      </div>
    </div>

    <section class="admin-section active" data-admin-section="overview">
      <div class="admin-stat-grid">
        <div class="admin-stat-card">
          <div class="admin-stat-label">Items Today</div>
          <div class="admin-stat-value" id="mItems">0</div>
          <div class="admin-stat-note">Processed cycles</div>
        </div>
        <div class="admin-stat-card">
          <div class="admin-stat-label">Rejected</div>
          <div class="admin-stat-value" id="mRejected">0</div>
          <div class="admin-stat-note">No-sort decisions</div>
        </div>
        <div class="admin-stat-card">
          <div class="admin-stat-label">Final Class</div>
          <div class="admin-stat-value" id="mFinal">-</div>
          <div class="admin-stat-note">Latest decision</div>
        </div>
        <div class="admin-stat-card">
          <div class="admin-stat-label">Confidence</div>
          <div class="admin-stat-value" id="mConf">0%</div>
          <div class="admin-stat-note">Fused model score</div>
        </div>
      </div>

      <div class="admin-layout-2">
        <div class="admin-panel-stack">
          <div class="card bin-viz-card">
            <h3>Bin Fullness</h3>
            <div class="mini">Live fill-level view for each recovery stream.</div>
            <div id="binBars" class="bin-visual-grid"></div>
          </div>

          <div class="card">
            <h3>Latest Decision</h3>
            <div class="admin-layout-even">
              <div>
                <div class="kv"><span>Vision</span><b id="adminYolo">-</b></div>
                <div class="kv"><span>Spectrum</span><b id="adminSpec">-</b></div>
                <div class="kv"><span>Detected</span><b id="adminDetected">-</b></div>
                <div class="kv"><span>Route</span><b id="adminRoute">-</b></div>
              </div>
              <div>
                <div class="kv"><span>Reject Reason</span><b id="adminReject">-</b></div>
                <div class="kv"><span>Server Time</span><b id="adminTime">0.0s</b></div>
                <div class="kv"><span>Saved Image</span><b id="adminImagePath">-</b></div>
              </div>
            </div>
          </div>
        </div>

        <div class="admin-panel-stack">
          <div class="admin-preview">
            <img id="adminLatestImage" src="/captures/latest.jpg" onerror="this.style.display='none'">
            <div class="admin-preview-overlay"></div>
            <div class="admin-preview-caption">
              <b>Latest Chamber Image</b>
              <span class="mini">Used for visual classification and operator inspection.</span>
            </div>
          </div>

          <div class="card">
            <h3>System Health</h3>
            <div class="kv"><span>Pi</span><b id="adminPiStatus">-</b></div>
            <div class="kv"><span>Heartbeat</span><b id="adminPiHeartbeat">-</b></div>
            <div class="kv"><span>Faults</span><b id="sensorFaultText">-</b></div>
            <div class="kv"><span>Service Mode</span><b id="serviceText">Service mode: normal</b></div>
          </div>
        </div>
      </div>
    </section>

    <section class="admin-section" data-admin-section="devices">
      <div class="admin-layout-2">
        <div class="card">
          <h3>Raspberry Pi Link</h3>
          <div class="mini">Connection and last command telemetry from the Pi heartbeat loop.</div>
          <div style="margin-top:14px">
            <div class="kv"><span>Pi Status</span><b id="adminPiStatusMirror">See overview</b></div>
            <div class="kv"><span>Last Heartbeat</span><b id="adminPiHeartbeatMirror">See overview</b></div>
            <div class="kv"><span>Last Command</span><b id="lastPiCmd">-</b></div>
            <div class="kv"><span>Result</span><b id="lastPiResult">-</b></div>
          </div>
        </div>

        <div class="card">
          <h3>Pi Recovery Actions</h3>
          <div class="mini">Queue safe one-shot commands. The Pi receives them on the next heartbeat.</div>
          <div class="admin-action-grid" style="margin-top:14px">
            <button class="btn" type="button" onclick="piCmd('capture_test_image')">Capture Test Image</button>
            <button class="btn" type="button" onclick="piCmd('restart_camera')">Restart Camera</button>
            <button class="btn" type="button" onclick="piCmd('reset_outputs')">Set GPIO Outputs Low</button>
            <button class="btn" type="button" onclick="piCmd('request_fullness')">Request Fullness Update</button>
          </div>
        </div>
      </div>
    </section>

    <section class="admin-section" data-admin-section="analytics">
      <div class="admin-layout-even">
        <div class="card">
          <h3>Material Share</h3>
          <div class="mini">Classification mix today, normalised to 100%.</div>
          <div id="countBars" class="share-bars"></div>
        </div>

        <div class="card">
          <h3>Decision Pipeline</h3>
          <div class="mini">Current fusion output from visual and spectral models.</div>
          <div style="margin-top:14px">
            <div class="kv"><span>Visual Model</span><b id="adminYoloMirror">-</b></div>
            <div class="kv"><span>Spectral Model</span><b id="adminSpecMirror">-</b></div>
            <div class="kv"><span>Final Route</span><b id="adminRouteMirror">-</b></div>
            <div class="kv"><span>Confidence</span><b id="mConfMirror">0%</b></div>
          </div>
        </div>
      </div>
    </section>

    <section class="admin-section" data-admin-section="logs">
      <div class="card">
        <h3>Activity Log</h3>
        <div class="mini">Recent system events, model decisions, heartbeat updates, and alerts.</div>
        <div class="admin-log-stream" id="logBox"></div>
      </div>
    </section>

    <section class="admin-section" data-admin-section="alerts">
      <div class="admin-layout-even">
        <div class="card">
          <h3>Alert State</h3>
          <div class="mini">Clear alert cooldowns after maintenance or use notification tests during commissioning.</div>
          <div style="margin-top:14px">
            <div class="kv"><span>Sensor Faults</span><b id="sensorFaultTextMirror">none</b></div>
            <div class="kv"><span>Service</span><b id="serviceTextMirror">normal</b></div>
          </div>
          <div class="admin-action-grid" style="margin-top:14px">
            <button class="btn" type="button" onclick="adminApi('/api/admin/clear-alerts')">Clear Alert States</button>
            <button class="btn" type="button" onclick="callApi('/api/test/sensor-fault')">Simulate Sensor Fault</button>
          </div>
        </div>

        <div class="card">
          <h3>Notification Tests</h3>
          <div class="mini">Verify the external notification paths without changing the main system logic.</div>
          <div class="admin-action-grid" style="margin-top:14px">
            <button class="btn" type="button" onclick="callApi('/api/test/telegram')">Send Telegram Test</button>
            <button class="btn" type="button" onclick="callApi('/api/test/email')">Send Email Test</button>
            <button class="btn" type="button" onclick="callApi('/api/test/pi-heartbeat')">Simulate Pi Heartbeat</button>
            <button class="btn" type="button" onclick="callApi('/api/test/fullness-warning')">Simulate Fullness Warning</button>
          </div>
        </div>
      </div>
    </section>

    <section class="admin-section" data-admin-section="reports">
      <div class="admin-layout-even">
        <div class="card">
          <h3>Cycle Data Export</h3>
          <div class="mini">Download today's CSV log for report evidence, debugging, and dataset review.</div>
          <div class="admin-action-grid single" style="margin-top:14px">
            <button class="btn primary" type="button" onclick="downloadCsv()">Download Today CSV</button>
          </div>
        </div>

        <div class="card">
          <h3>Saved Assets</h3>
          <div class="mini">Classified images are saved by class under <b>static/dataset/</b>.</div>
          <div style="margin-top:14px">
            <div class="kv"><span>Latest Save</span><b id="adminImagePathMirror">-</b></div>
            <div class="kv"><span>Latest Capture</span><b>static/captures/latest.jpg</b></div>
          </div>
        </div>
      </div>
    </section>

    <section class="admin-section" data-admin-section="controls">
      <div class="admin-layout-even">
        <div class="card">
          <h3>Service Control</h3>
          <div class="mini">Disable the station during maintenance or enable it after bins and sensors are checked.</div>
          <div class="admin-action-grid" style="margin-top:14px">
            <button class="btn danger" type="button" onclick="adminApi('/api/admin/disable')">Disable Station</button>
            <button class="btn primary" type="button" onclick="adminApi('/api/admin/enable')">Enable Station</button>
            <button class="btn" type="button" onclick="adminApi('/api/admin/reset')">Reset Daily Counters</button>
            <button class="btn" type="button" onclick="adminApi('/api/admin/clear-alerts')">Clear Alert States</button>
          </div>
        </div>

        <div class="card">
          <h3>Manual Checks</h3>
          <div class="mini">Quick checks for common demo-day recovery cases.</div>
          <div class="admin-action-grid" style="margin-top:14px">
            <button class="btn" type="button" onclick="callApi('/api/test/fullness-full')">Simulate Full Bin</button>
            <button class="btn" type="button" onclick="callApi('/api/test/all-full')">Simulate All Bins Full</button>
            <button class="btn" type="button" onclick="callApi('/api/test/pi-heartbeat')">Confirm Pi Link</button>
            <button class="btn" type="button" onclick="piCmd('status')">Request Pi Status</button>
          </div>
        </div>
      </div>
    </section>
  </main>
</div>
<script>
const pages = ["idle", "door_open", "door_close", "processing", "classifying", "result", "thank_you", "sorting", "out_of_service"];
let currentPage = "idle";
let lastItemNumber = 0;
let resultTimer = null;
let idleTimer = null;
let sortingTimer = null;
let waitingForCycleComplete = false;
let cycleCompleteShown = false;
let latestCycleData = null;
const icons = { plastic: "🥤", paper: "📦", glass: "🍾", metal: "🥫", null: "⚠", unknown: "?" };
let factIndex = 0;
const funFacts = [
  "Clean recyclables are more likely to be accepted by recycling facilities.",
  "Glass can be recycled repeatedly without losing quality.",
  "Paper recycling helps reduce the demand for new wood pulp.",
  "Correct sorting helps reduce recycling contamination.",
  "Every correctly sorted item improves the recycling stream."
];


function showPage(name) {
  if (!pages.includes(name)) return;
  const pageEl = document.getElementById(name + "Page");
  if (!pageEl) return;
  if (currentPage === name && pageEl.classList.contains("active")) return;
  pages.forEach(p => document.getElementById(p + "Page").classList.remove("active"));
  pageEl.classList.add("active");
  currentPage = name;
  if (name === "processing" && typeof resizeCanvas === "function") requestAnimationFrame(() => resizeCanvas());
}


function openAdminLogin() {
  fetch("/api/admin/auth-status")
    .then(r => r.json())
    .then(d => { if (d.logged_in) openAdmin(); else document.getElementById("loginModal").classList.add("active"); })
    .catch(() => document.getElementById("loginModal").classList.add("active"));
}
function closeAdminLogin() { document.getElementById("loginModal").classList.remove("active"); }
async function adminLogin() {
  const u = document.getElementById("loginUser").value;
  const p = document.getElementById("loginPass").value;
  const msg = document.getElementById("loginMsg");
  try {
    const r = await fetch("/api/admin/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username: u, password: p }) });
    const d = await r.json();
    if (r.ok) { closeAdminLogin(); openAdmin(); msg.textContent = ""; }
    else msg.textContent = d.message || "Login failed";
  } catch(e) { msg.textContent = "Login request failed"; }
}
function setAdminTab(tab) {
  document.querySelectorAll("[data-admin-tab]").forEach(btn => btn.classList.toggle("active", btn.dataset.adminTab === tab));
  document.querySelectorAll("[data-admin-section]").forEach(sec => sec.classList.toggle("active", sec.dataset.adminSection === tab));
}
function openAdmin() {
  document.getElementById("adminScreen").classList.add("active");
  setAdminTab("overview");
}
async function adminLogout() { await fetch("/api/admin/logout", { method: "POST" }); document.getElementById("adminScreen").classList.remove("active"); }

function renderBars(id, data) {
  const c = document.getElementById(id); if (!c) return; c.innerHTML = "";
  Object.entries(data || {}).forEach(([k, v]) => {
    let p = Number(v) || 0; if (p <= 1) p *= 100;
    c.innerHTML += `<div>${k.toUpperCase()} <span style="float:right">${Math.round(p)}%</span></div><div class="bar"><div class="bar-fill" style="width:${Math.min(100, Math.max(0, p))}%"></div></div>`;
  });
}
function binStatus(v) { v = Number(v) || 0; if (v >= 100) return "FULL"; if (v >= 80) return "WARNING"; return "NORMAL"; }
function renderBin3D(id, bins, faults) {
  const c = document.getElementById(id); if (!c) return;
  const order = ["metal", "glass", "paper", "plastic"];
  c.innerHTML = "";
  order.forEach(name => {
    const v = Math.max(0, Math.min(100, Number((bins || {})[name]) || 0));
    const status = binStatus(v);
    const fault = !!((faults || {})[name]);
    const cls = status === "FULL" ? "full" : status === "WARNING" ? "warning" : "normal";
    c.innerHTML += `<div class="bin3d ${cls} ${fault ? "fault" : ""}"><div class="bin3d-stage"><div class="bin3d-body"><div class="bin3d-fill" style="height:${v}%"></div><div class="bin3d-shine"></div></div></div><div class="bin3d-name">${name.toUpperCase()}</div><div class="bin3d-percent">${Math.round(v)}%</div><div class="bin3d-status">${fault ? "FAULT" : status}</div></div>`;
  });
}
function normalisedPercentages(data) {
  const order = ["metal", "glass", "paper", "plastic", "null"];
  const entries = order.map(k => [k, Number((data || {})[k]) || 0]);
  const total = entries.reduce((a, [, v]) => a + v, 0);
  if (total <= 0) return entries.map(([k, v]) => ({ name: k, count: v, pct: 0 }));
  const raw = entries.map(([k, v]) => {
    const exact = v * 100 / total;
    return { name: k, count: v, base: Math.floor(exact), rem: exact - Math.floor(exact) };
  });
  let used = raw.reduce((a, x) => a + x.base, 0);
  raw.sort((a, b) => b.rem - a.rem);
  for (let i = 0; i < raw.length && used < 100; i++, used++) raw[i].base += 1;
  raw.sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));
  return raw.map(x => ({ name: x.name, count: x.count, pct: x.base }));
}
function renderCountShare(id, data) {
  const c = document.getElementById(id); if (!c) return;
  const total = Object.values(data || {}).reduce((a, v) => a + (Number(v) || 0), 0);
  if (total <= 0) { c.innerHTML = `<div class="share-empty">No classified items yet</div>`; return; }
  const rows = normalisedPercentages(data);
  c.innerHTML = "";
  rows.forEach(r => {
    c.innerHTML += `<div class="share-row"><div class="share-top"><span>${r.name.toUpperCase()}</span><b>${r.pct}%</b></div><div class="share-track"><div class="share-fill" style="width:${r.pct}%"></div></div><div class="share-meta">${r.count} item${r.count === 1 ? "" : "s"}</div></div>`;
  });
}
function isNoSortCycle(data) { return !!(data && (data.route_class === "null" || data.final_prediction === "null" || data.reject_reason)); }
function getRouteClass(data) {
  if (!data) return "null";
  if (isNoSortCycle(data)) return "null";
  return String(data.detected_material || data.final_prediction || "null").toLowerCase();
}
function getTransportDurationMs(data) {
  const cls = getRouteClass(data);

  if (cls === "plastic") return 16000;
  if (cls === "metal" || cls === "null") return 15000;
  if (cls === "glass" || cls === "paper") return 11000;

  return 15000;
}
function scheduleResultFlow(data) {
  clearTimeout(resultTimer); clearTimeout(idleTimer); clearTimeout(sortingTimer);
  waitingForCycleComplete = true;
  cycleCompleteShown = false;
  latestCycleData = data || {};
  resultTimer = setTimeout(() => {
    updateRoutingVisual(latestCycleData);
    showPage("sorting");
    sortingTimer = setTimeout(() => finishPhysicalCycle(latestCycleData), getTransportDurationMs(latestCycleData));
  }, 2600);
}
function finishPhysicalCycle(data) {
  if (cycleCompleteShown) return;
  cycleCompleteShown = true;
  waitingForCycleComplete = false;
  latestCycleData = data || latestCycleData || {};
  clearTimeout(resultTimer); clearTimeout(sortingTimer);
  const title = document.querySelector("#thank_youPage .main-title");
  const fact = document.getElementById("thankFact");
  if (isNoSortCycle(latestCycleData)) {
    if (title) title.textContent = "Item Discarded";
    if (fact) fact.textContent = "This item was not sent to a recyclable stream.";
  } else {
    if (title) title.textContent = "Sorted Successfully";
    if (fact) fact.textContent = (latestCycleData && latestCycleData.fun_fact) || "Your action makes a big difference.";
  }
  showPage("thank_you");
  idleTimer = setTimeout(() => showPage("idle"), 3600);
}
function updateRoutingVisual(data) {
  latestCycleData = data || latestCycleData || {};
  const isNoSort = isNoSortCycle(latestCycleData);
  const detected = String(latestCycleData.detected_material || latestCycleData.final_prediction || "unknown").toLowerCase();
  const cls = isNoSort ? "null" : detected;
  const target = isNoSort ? "metal" : cls;
  const stage = document.getElementById("routeStage");
  if (stage) {
    stage.classList.remove("target-plastic", "target-paper", "target-glass", "target-metal", "no-sort");
    stage.classList.add(isNoSort ? "no-sort" : ("target-" + (target || "paper")));
    stage.style.setProperty("--ui-route-duration", (getTransportDurationMs(latestCycleData) / 1000) + "s");
  }
  const progress = document.getElementById("routeProgressBar");
  if (progress) {
    progress.style.animation = "none";
    void progress.offsetWidth;
    progress.style.animation = `routeProgress ${getTransportDurationMs(latestCycleData) / 1000}s linear forwards`;
  }
  const icon = document.getElementById("routeTrashIcon"); if (icon) icon.textContent = icons[cls] || "♻";
  const label = document.getElementById("routeTargetBin"); if (label) label.textContent = isNoSort ? "NO-SORT / LEFT DROP" : "ROUTING TO " + cls.toUpperCase() + " BIN";
  const msg = document.getElementById("routeMessage"); if (msg) msg.textContent = isNoSort ? "No-sort item is moved to the left drop path and discarded." : "Chamber moves to the " + cls + " stream, releases the item, then returns to centre.";
  document.querySelectorAll(".route-bin").forEach(b => b.classList.toggle("active", (isNoSort && b.dataset.bin === "metal") || (!isNoSort && b.dataset.bin === cls)));
}
setInterval(() => {
  if (document.getElementById("classifyingPage").classList.contains("active")) {
    const f = document.getElementById("processingFact");
    if (f) { f.textContent = funFacts[factIndex % funFacts.length]; factIndex++; }
  }
}, 3000);
async function callApi(url) { try { const r = await fetch(url, { method: "POST" }); const d = await r.json(); alert((d.status || "ok") + ": " + (d.message || JSON.stringify(d))); } catch(e) { alert("Request failed"); } }
async function adminApi(url) { try { const r = await fetch(url, { method: "POST" }); if (!r.ok && r.status === 401) { openAdminLogin(); return; } } catch(e) { alert("Admin request failed"); } }
async function piCmd(command) { try { const r = await fetch("/api/admin/pi-command", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ command }) }); const d = await r.json(); if (!r.ok) alert(d.message || "Command failed"); } catch(e) { alert("Pi command failed"); } }
function downloadCsv() { window.open("/api/admin/today-csv", "_blank"); }
async function pollStatus() { try { const res = await fetch("/api/status"); const data = await res.json(); updatePublic(data); updateAdmin(data); } catch(e) { console.log("poll failed", e); } }
function updatePublic(data) {
  if (data.out_of_service) {
    let reason = data.out_of_service_reason || "";
    document.getElementById("outServiceText").textContent = reason === "operator_disabled" ? "This station has been disabled by the operator for maintenance." : reason === "all_bins_full" ? "All recycling streams are full. Operator service required." : "This bin is temporarily unavailable.";
    showPage("out_of_service");
    return;
  }
  if (waitingForCycleComplete) {
    updateRoutingVisual(data);
    const active = document.querySelector(".page.active");
    if (active && active.id !== "resultPage" && active.id !== "sortingPage") showPage("sorting");
    return;
  }
  const s = data.system_state;
  if (s === "idle") showPage("idle");
  else if (s === "door_open" || s === "insert") showPage("door_open");
  else if (s === "door_close" || s === "door") showPage("door_close");
  else if (s === "processing") showPage("processing");
  else if (s === "classifying" || s === "scanning") {
    if (data.image_ready) {
      showPage("classifying");
      const img = document.getElementById("scanImage");
      img.style.display = "block";
      img.src = data.latest_image + "?v=" + data.image_version + "&t=" + Date.now();
    } else showPage("processing");
  } else if (s === "sorting") showPage("sorting");

  const itemNo = data.item_number_today || 0;
  if (itemNo > lastItemNumber && data.final_prediction && data.final_prediction !== "-") {
    lastItemNumber = itemNo;
    const cls = (data.route_class === "null" || data.final_prediction === "null") ? "null" : String(data.final_prediction || "unknown").toLowerCase();
    const pretty = cls === "null" ? "UNABLE TO SORT" : cls.toUpperCase();
    document.getElementById("materialIcon").textContent = icons[cls] || "♻";
    document.getElementById("finalPrediction").textContent = pretty;
    document.getElementById("confidenceText").textContent = cls === "null" ? (data.decision_message || "Please try another item") : "Directed to " + cls + " recovery stream";
    document.getElementById("itemNumber").textContent = "Item #" + itemNo + " today";
    document.getElementById("confidencePill").textContent = Math.round((data.final_confidence || 0) * 100) + "% confidence";
    document.getElementById("thankFact").textContent = data.fun_fact || "Your action makes a big difference.";
    updateRoutingVisual(data);
    showPage("result");
    scheduleResultFlow(data);
  }
}
function updateAdmin(data) {
  document.getElementById("mItems").textContent = data.item_number_today || 0;
  document.getElementById("mRejected").textContent = data.rejected_today || 0;
  document.getElementById("mFinal").textContent = (data.final_prediction || "-").toUpperCase();
  document.getElementById("mConf").textContent = Math.round((data.final_confidence || 0) * 100) + "%";
  renderBin3D("binBars", data.bins || {}, data.sensor_faults || {});
  renderCountShare("countBars", data.counts || {});
  document.getElementById("adminYolo").textContent = data.yolo_prediction || "-";
  document.getElementById("adminSpec").textContent = data.spectral_prediction || "-";
  document.getElementById("adminDetected").textContent = data.detected_material || "-";
  document.getElementById("adminRoute").textContent = data.route_class || "-";
  document.getElementById("adminReject").textContent = data.reject_reason || "-";
  document.getElementById("adminTime").textContent = (data.server_time || 0).toFixed(3) + "s";
  document.getElementById("adminImagePath").textContent = data.image_saved_path || "-";
  document.getElementById("adminPiStatus").textContent = data.pi_connected ? "online" : (data.pi_status || "offline");
  document.getElementById("adminPiHeartbeat").textContent = data.last_pi_heartbeat || "-";
  document.getElementById("lastPiCmd").textContent = data.last_pi_command || "-";
  document.getElementById("lastPiResult").textContent = data.last_pi_command_result || "-";
  document.getElementById("serviceText").textContent = "Service mode: " + (data.service_mode || "normal");
  let faults = Object.entries(data.sensor_faults || {}).filter(x => x[1]).map(x => x[0]);
  document.getElementById("sensorFaultText").textContent = faults.length ? faults.join(", ") : "none";
  document.getElementById("logBox").innerHTML = (data.logs || []).map(x => `<div>${x}</div>`).join("");
  const sb = document.getElementById("serviceBadge");
  sb.textContent = (data.service_mode || "normal").toUpperCase();
  sb.className = "badge " + (data.out_of_service ? "bad" : "ok");
  const pb = document.getElementById("piBadge");
  pb.textContent = data.pi_connected ? "PI ONLINE" : "PI OFFLINE";
  pb.className = "badge " + (data.pi_connected ? "ok" : "bad");

  const setText = (id, value) => { const el = document.getElementById(id); if (el) el.textContent = value; };
  setText("adminPiStatusMirror", data.pi_connected ? "online" : (data.pi_status || "offline"));
  setText("adminPiHeartbeatMirror", data.last_pi_heartbeat || "-");
  setText("adminYoloMirror", data.yolo_prediction || "-");
  setText("adminSpecMirror", data.spectral_prediction || "-");
  setText("adminRouteMirror", data.route_class || "-");
  setText("mConfMirror", Math.round((data.final_confidence || 0) * 100) + "%");
  setText("sensorFaultTextMirror", faults.length ? faults.join(", ") : "none");
  setText("serviceTextMirror", data.service_mode || "normal");
  setText("adminImagePathMirror", data.image_saved_path || "-");
  const preview = document.getElementById("adminLatestImage");
  if (preview && data.latest_image) {
    preview.style.display = "block";
    preview.src = data.latest_image + "?v=" + (data.image_version || 0) + "&t=" + Date.now();
  }
}
setInterval(pollStatus, 700);
pollStatus();

const canvas=document.getElementById("dnaCanvas");
const ctx=canvas.getContext("2d");
let dnaReady=false;
let phase=0;
let particles=[];
let profileSeed=Math.random()*1000;
let dnaCanvasW=0;
let dnaCanvasH=0;
let lastDnaFrame=0;

function resizeCanvas(){
  const rect=canvas.getBoundingClientRect();
  if(rect.width<=0 || rect.height<=0) return;
  const dpr=Math.min(window.devicePixelRatio || 1, 1.5);
  const nextW=Math.floor(rect.width*dpr);
  const nextH=Math.floor(rect.height*dpr);
  if(dnaReady && nextW===dnaCanvasW && nextH===dnaCanvasH) return;
  dnaCanvasW=nextW;
  dnaCanvasH=nextH;
  canvas.width=nextW;
  canvas.height=nextH;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  dnaReady=true;
  particles=Array.from({length:60},()=>({
    x:80+Math.random()*Math.max(1,rect.width-160),
    y:34+Math.random()*Math.max(1,rect.height-68),
    r:Math.random()*1.05+.25,
    vx:Math.random()*0.08+0.015,
    vy:(Math.random()-.5)*0.035,
    color:Math.random()<.50?"#00e5ff":(Math.random()<.72?"#8b5cf6":"#00ff9c"),
    alpha:0.025+Math.random()*0.08
  }));
}


function dot(context,x,y,r,color,alpha,blur=14){
  context.save();
  context.globalAlpha=alpha;
  context.fillStyle=color;
  context.shadowColor=color;
  context.shadowBlur=blur;
  context.beginPath();
  context.arc(x,y,r,0,Math.PI*2);
  context.fill();
  context.restore();
}

function profileAt(t){
  const TAU=Math.PI*2;
  const a=.70
    +.34*Math.sin(TAU*(1.12*t)+profileSeed+phase*.20)
    +.25*Math.sin(TAU*(2.70*t)-profileSeed*.63-phase*.145)
    +.15*Math.sin(TAU*(5.40*t)+profileSeed*1.41+phase*.25)
    +.10*Math.sin(TAU*(9.80*t)-profileSeed*2.10+phase*.18);
  return Math.max(.24,Math.min(1.45,a));
}

function drawDNA(now){
  requestAnimationFrame(drawDNA);

  const processingActive = document.getElementById("processingPage")?.classList.contains("active");
  if(!processingActive) return;

  if(now && now - lastDnaFrame < 33) return;
  lastDnaFrame = now || performance.now();

  const w=canvas.clientWidth;
  const h=canvas.clientHeight;
  if(w<=0 || h<=0) return;
  if(!dnaReady) resizeCanvas();
  if(!dnaReady) return;

  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  ctx.save();
  ctx.setTransform(1,0,0,1,0,0);
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.restore();
  ctx.setTransform(dpr,0,0,dpr,0,0);

  const bg=ctx.createRadialGradient(w*.50,h*.44,40,w*.50,h*.44,w*.68);
  bg.addColorStop(0,"rgba(0,229,255,.10)");
  bg.addColorStop(.45,"rgba(139,92,246,.055)");
  bg.addColorStop(1,"rgba(0,0,0,0)");
  ctx.fillStyle=bg;
  ctx.fillRect(0,0,w,h);

  const edgePad=86, topPad=34, bottomPad=38;
  for(const p of particles){
    p.x+=p.vx;
    p.y+=p.vy+Math.sin(phase*.35+p.x*.01)*.012;
    if(p.x>w-edgePad){p.x=edgePad;p.y=topPad+Math.random()*Math.max(1,h-topPad-bottomPad);}
    if(p.y<topPad)p.y=h-bottomPad;
    if(p.y>h-bottomPad)p.y=topPad;
    dot(ctx,p.x,p.y,p.r,p.color,p.alpha,5);
  }

  const startX=edgePad,endX=w-edgePad,len=endX-startX,centerY=h*.46,baseAmp=h*.19,turns=3.10,cols=112;
  const objects=[];

  function helixPoint(t,offset){
    const prof=profileAt(t);
    const theta=t*Math.PI*2*turns+phase+offset;
    const x=startX+t*len;
    const depth=(Math.cos(theta)+1)/2;
    const centerWarp=Math.sin(t*Math.PI*2.1+phase*.14)*h*.025+Math.sin(t*Math.PI*5.0+profileSeed)*h*.008;
    const amp=baseAmp*prof*(.92+.08*Math.sin(phase*.55+t*10.0+profileSeed));
    const y=centerY+centerWarp+Math.sin(theta)*amp;
    return {x,y,depth,prof};
  }

  for(let band=-2;band<=2;band++){
    for(let i=0;i<cols;i+=3){
      const t=i/(cols-1),prof=profileAt(t),x=startX+t*len;
      const y=centerY+band*18+Math.sin(t*Math.PI*2*turns+phase*.42+band*.55)*baseAmp*.22*prof;
      const c=band%2?"#8b5cf6":"#00e5ff";
      dot(ctx,x,y,.65,c,.045,4);
    }
  }

  for(let i=0;i<cols;i++){
    const t=i/(cols-1),pA=helixPoint(t,0),pB=helixPoint(t,Math.PI),prof=profileAt(t);

    if(i%3===0){
      const rungDots=4+Math.round(prof*4);
      for(let j=0;j<rungDots;j++){
        const k=j/(rungDots-1);
        const x=pA.x*(1-k)+pB.x*k;
        const y=pA.y*(1-k)+pB.y*k;
        const z=pA.depth*(1-k)+pB.depth*k;
        const color=k<.45?"#00e5ff":(k>.55?"#8b5cf6":"#66f2ff");
        objects.push({x,y,z:z-.18,r:.8+z*1.8,color,alpha:.07+z*.22,blur:5+z*7});
      }
    }

    const shimmer=.88+.12*Math.sin(phase*3.2+i*.31);
    objects.push({x:pA.x,y:pA.y,z:pA.depth,r:(1.5+pA.depth*3.6+prof*.7)*shimmer,color:pA.depth>.56?"#00ff9c":"#00e5ff",alpha:.18+pA.depth*.58,blur:8+pA.depth*16});
    objects.push({x:pB.x,y:pB.y,z:pB.depth,r:(1.5+pB.depth*3.6+prof*.7)*shimmer,color:pB.depth>.56?"#8b5cf6":"#66d9ff",alpha:.18+pB.depth*.56,blur:8+pB.depth*16});
  }

  objects.sort((a,b)=>a.z-b.z);
  for(const o of objects) dot(ctx,o.x,o.y,o.r,o.color,o.alpha,o.blur);
  phase+=.032;
}


window.addEventListener("resize",resizeCanvas);
requestAnimationFrame(()=>{resizeCanvas();requestAnimationFrame(drawDNA);});

</script></body></html>

"""
@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":

    email_thread = threading.Thread(
        target=hourly_email_loop,
        daemon=True
    )
    email_thread.start()

    monitor_thread = threading.Thread(
        target=monitoring_loop,
        daemon=True
    )
    monitor_thread.start()

    print(f"Hourly email interval: {EMAIL_REPORT_INTERVAL_SECONDS} seconds")
    print(f"Test Telegram: {DASHBOARD_URL}/api/test/telegram")
    print(f"Test Email   : {DASHBOARD_URL}/api/test/email")
    print(f"Reset Day    : {DASHBOARD_URL}/api/admin/reset")

    send_telegram_alert(
        "startup",
        f"✅ <b>EcoVision system online</b>\n\nLaptop server started.\nDashboard: {DASHBOARD_URL}",
        cooldown_seconds=0,
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )
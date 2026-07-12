"""
VisionGuard AI — Industrial Computer Vision Quality Control System
====================================================================
Stack: Streamlit + Ultralytics YOLOv11 + EfficientNet-B2 + Gemini Vision

Features
--------
1. Camera input        : webcam (quasi-live) or uploaded video file
2. Custom defect model  : YOLOv11 inference + in-app training pipeline
3. Fine-grained classifier: EfficientNet-B2 (scratch/dent/discoloration/contamination)
4. AI defect explainer  : Gemini Vision -> root cause + corrective action
5. Real-time dashboard  : hourly defect rate, type distribution, silhouette heatmap
6. Defect database      : SQLite + image archive on disk
7. SPC charts           : X-bar, R chart, P chart (Plotly)

No API key is bundled. Paste a Gemini API key in the sidebar to enable
the AI explainer — every other feature works without one.
"""

import io
import os
import sqlite3
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ------------------------------------------------------------------
# Optional heavy dependencies are imported defensively so the app can
# still launch (and explain what's missing) even if a package is absent.
# ------------------------------------------------------------------
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    GEMINI_SDK_AVAILABLE = False


# ====================================================================
# GLOBAL CONFIG
# ====================================================================
APP_TITLE = "VisionGuard AI"
APP_SUBTITLE = "Industrial Computer-Vision Quality Control System"
DB_PATH = "qc_system.db"
ARCHIVE_DIR = Path("defect_archive")
RUNS_DIR = Path("training_runs")
ARCHIVE_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)

DEFECT_CLASSES = ["scratch", "dent", "discoloration", "contamination"]
DEFECT_COLORS = {
    "scratch": "#FF7A1A",
    "dent": "#FFC93C",
    "discoloration": "#8E7CC3",
    "contamination": "#FF3B3B",
}

# Standard Shewhart control-chart constants (subgroup size n : A2, D3, D4)
SPC_CONSTANTS = {
    2: (1.880, 0.000, 3.267),
    3: (1.023, 0.000, 2.574),
    4: (0.729, 0.000, 2.282),
    5: (0.577, 0.000, 2.114),
    6: (0.483, 0.000, 2.004),
    7: (0.419, 0.076, 1.924),
    8: (0.373, 0.136, 1.864),
    9: (0.337, 0.184, 1.816),
    10: (0.308, 0.223, 1.777),
}

st.set_page_config(
    page_title=f"{APP_TITLE} | QC System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ====================================================================
# DARK INDUSTRIAL THEME
# ====================================================================
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@500;600;700&display=swap');

        html, body, [class*="css"]  { font-family: 'Rajdhani', sans-serif; }

        .stApp {
            background: radial-gradient(circle at top left, #14181c 0%, #0b0d0f 55%, #08090a 100%);
            color: #e7ecef;
        }

        section[data-testid="stSidebar"] {
            background: #101317;
            border-right: 2px solid #2a2f36;
        }

        h1, h2, h3 { font-family: 'Rajdhani', sans-serif; letter-spacing: 0.5px; color: #f5a623; }

        .vg-header {
            display:flex; align-items:center; gap:14px;
            border-bottom: 3px solid #f5a623;
            padding-bottom: 10px; margin-bottom: 18px;
        }
        .vg-badge {
            font-family:'Share Tech Mono', monospace;
            background:#f5a623; color:#101317; font-weight:700;
            padding:3px 10px; border-radius:3px; font-size:0.75rem;
            letter-spacing:1px;
        }
        .vg-pill {
            display:inline-block; padding:2px 10px; border-radius:12px;
            font-size:0.72rem; font-weight:600; margin-right:6px;
            border:1px solid #3a3f46; color:#c9cfd6; font-family:'Share Tech Mono',monospace;
        }
        .vg-ok   { border-color:#3ddc84; color:#3ddc84; }
        .vg-warn { border-color:#f5a623; color:#f5a623; }
        .vg-bad  { border-color:#ff4d4d; color:#ff4d4d; }

        div[data-testid="stMetric"] {
            background:#14181d; border:1px solid #2a2f36; border-radius:10px;
            padding:10px 14px; box-shadow: 0 0 0 1px rgba(245,166,35,0.05) inset;
        }
        div[data-testid="stMetricValue"] { color:#f5a623; }

        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            background:#14181d; border:1px solid #2a2f36; border-radius:6px 6px 0 0;
            padding:8px 16px; color:#9aa2ab;
        }
        .stTabs [aria-selected="true"] { color:#f5a623 !important; border-bottom:2px solid #f5a623 !important; }

        .stButton>button {
            background:#f5a623; color:#101317; border:none; font-weight:700;
            border-radius:6px; letter-spacing:0.5px;
        }
        .stButton>button:hover { background:#ffb84d; color:#101317; }

        code, .stCodeBlock { font-family:'Share Tech Mono', monospace; }

        hr { border-color:#2a2f36; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header():
    st.markdown(
        f"""
        <div class="vg-header">
            <h1 style="margin:0;">🏭 {APP_TITLE}</h1>
            <span class="vg-badge">QC-VISION v1.0</span>
        </div>
        <p style="margin-top:-10px;color:#9aa2ab;">{APP_SUBTITLE}</p>
        """,
        unsafe_allow_html=True,
    )


# ====================================================================
# DATABASE LAYER (SQLite + image archive)
# ====================================================================
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS defects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            lot_id TEXT NOT NULL,
            source TEXT,
            defect_type TEXT,
            det_confidence REAL,
            cls_confidence REAL,
            bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
            area_pct REAL,
            image_path TEXT,
            root_cause TEXT,
            corrective_action TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            lot_id TEXT NOT NULL,
            source TEXT,
            units_inspected INTEGER,
            units_defective INTEGER
        )
        """
    )
    conn.commit()
    conn.close()


def current_lot_id():
    """Bucket inspections/defects into hourly 'lots' for SPC/dashboard grouping."""
    return datetime.now().strftime("%Y-%m-%d %H:00")


def log_defect(source, defect_type, det_conf, cls_conf, bbox, area_pct,
               image_path, root_cause, corrective_action):
    conn = get_conn()
    conn.execute(
        """INSERT INTO defects
           (timestamp, lot_id, source, defect_type, det_confidence, cls_confidence,
            bbox_x, bbox_y, bbox_w, bbox_h, area_pct, image_path, root_cause, corrective_action)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now().isoformat(timespec="seconds"),
            current_lot_id(),
            source,
            defect_type,
            float(det_conf),
            float(cls_conf),
            *bbox,
            float(area_pct),
            image_path,
            root_cause,
            corrective_action,
        ),
    )
    conn.commit()
    conn.close()


def log_inspection(source, units_inspected, units_defective):
    conn = get_conn()
    conn.execute(
        """INSERT INTO inspections (timestamp, lot_id, source, units_inspected, units_defective)
           VALUES (?,?,?,?,?)""",
        (
            datetime.now().isoformat(timespec="seconds"),
            current_lot_id(),
            source,
            units_inspected,
            units_defective,
        ),
    )
    conn.commit()
    conn.close()


@st.cache_data(ttl=5)
def fetch_defects_df():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM defects ORDER BY id", conn)
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(ttl=5)
def fetch_inspections_df():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM inspections ORDER BY id", conn)
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def clear_database():
    conn = get_conn()
    conn.execute("DELETE FROM defects")
    conn.execute("DELETE FROM inspections")
    conn.commit()
    conn.close()
    fetch_defects_df.clear()
    fetch_inspections_df.clear()


# ====================================================================
# YOLOv11 DETECTION LAYER
# ====================================================================
@st.cache_resource(show_spinner="Loading YOLOv11 weights…")
def load_yolo_model(weights_path: str):
    if not YOLO_AVAILABLE:
        return None
    try:
        return YOLO(weights_path)
    except Exception as e:
        st.error(f"Could not load YOLO weights '{weights_path}': {e}")
        return None


def run_yolo_inference(model, frame_bgr, conf_thresh=0.35):
    """Returns list of dicts: {xyxy, conf, cls_name, cls_id}."""
    if model is None:
        return []
    results = model.predict(frame_bgr, conf=conf_thresh, verbose=False)
    dets = []
    if not results:
        return dets
    r = results[0]
    names = r.names
    if r.boxes is None:
        return dets
    for box in r.boxes:
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        dets.append({"xyxy": xyxy, "conf": conf, "cls_id": cls_id, "cls_name": cls_name})
    return dets


def draw_detections(frame_bgr, dets):
    if not CV2_AVAILABLE:
        return frame_bgr
    out = frame_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        label = f'{d.get("final_label", d["cls_name"])} {d["conf"]:.2f}'
        cv2.rectangle(out, (x1, y1), (x2, y2), (35, 166, 245), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), (35, 166, 245), -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2)
    return out


# ====================================================================
# EFFICIENTNET-B2 FINE-GRAINED CLASSIFIER
# ====================================================================
def build_efficientnet_b2(num_classes=len(DEFECT_CLASSES)):
    model = models.efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


@st.cache_resource(show_spinner="Loading EfficientNet-B2 classifier…")
def load_classifier(weights_path: str = None):
    if not TORCH_AVAILABLE:
        return None
    model = build_efficientnet_b2()
    if weights_path and os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            return {"model": model, "trained": True}
        except Exception as e:
            st.warning(f"Could not load classifier weights ({e}). Falling back to heuristic mode.")
    model.eval()
    return {"model": model, "trained": False}


CLASSIFIER_TRANSFORM = None
if TORCH_AVAILABLE:
    CLASSIFIER_TRANSFORM = transforms.Compose(
        [
            transforms.Resize((260, 260)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def heuristic_classify(crop_rgb: np.ndarray):
    """
    Lightweight, dependency-free fallback used whenever no fine-tuned
    EfficientNet-B2 checkpoint has been supplied. Uses simple image
    statistics (edge density, color variance, brightness) to produce a
    plausible defect-type label so the full pipeline remains demoable
    end-to-end without a trained model.
    """
    if crop_rgb.size == 0:
        return DEFECT_CLASSES[0], 0.25

    gray = crop_rgb.mean(axis=2)
    edge_density = float(np.mean(np.abs(np.gradient(gray)[0])) + np.mean(np.abs(np.gradient(gray)[1])))
    color_std = float(np.std(crop_rgb, axis=(0, 1)).mean())
    brightness = float(gray.mean())
    hue_spread = float(np.std(crop_rgb[..., 0].astype(float) - crop_rgb[..., 2].astype(float)))

    scores = {
        "scratch": edge_density * 1.4,
        "dent": edge_density * 0.6 + (255 - brightness) * 0.02,
        "discoloration": hue_spread * 1.2 + color_std * 0.3,
        "contamination": color_std * 1.1,
    }
    label = max(scores, key=scores.get)
    total = sum(scores.values()) + 1e-6
    confidence = min(0.95, max(0.35, scores[label] / total))
    return label, round(confidence, 3)


def classify_defect(crop_rgb: np.ndarray, clf_bundle):
    """Returns (label, confidence). Uses trained EfficientNet-B2 if available,
    otherwise falls back to a heuristic classifier."""
    if clf_bundle is None or not clf_bundle.get("trained") or not TORCH_AVAILABLE:
        return heuristic_classify(crop_rgb)
    try:
        pil_img = Image.fromarray(crop_rgb)
        tensor = CLASSIFIER_TRANSFORM(pil_img).unsqueeze(0)
        with torch.no_grad():
            logits = clf_bundle["model"](tensor)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs))
            conf = float(probs[idx])
        return DEFECT_CLASSES[idx], round(conf, 3)
    except Exception:
        return heuristic_classify(crop_rgb)


# ====================================================================
# GEMINI VISION DEFECT EXPLAINER
# ====================================================================
def get_gemini_explanation(pil_image: Image.Image, defect_type: str, confidence: float,
                            api_key: str, model_name: str):
    """Calls Gemini Vision to produce a probable root cause + corrective action.
    Returns (root_cause, corrective_action) or a graceful fallback message."""
    if not api_key:
        return ("No Gemini API key provided.",
                "Add a key in the sidebar to enable AI root-cause analysis.")
    if not GEMINI_SDK_AVAILABLE:
        return ("google-generativeai package not installed.",
                "Run: pip install google-generativeai")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = (
            f"You are a senior manufacturing quality engineer. An automated vision "
            f"system flagged a '{defect_type}' defect (classifier confidence "
            f"{confidence:.0%}) on a product surface, shown in the attached image.\n\n"
            "Respond in exactly two short sections:\n"
            "ROOT_CAUSE: <one or two sentences on the most probable process root cause>\n"
            "CORRECTIVE_ACTION: <one or two sentences of a concrete corrective/preventive action>"
        )
        response = model.generate_content([prompt, pil_image])
        text = response.text.strip()

        root_cause, corrective_action = "Unable to parse response.", text
        if "ROOT_CAUSE:" in text and "CORRECTIVE_ACTION:" in text:
            root_part = text.split("ROOT_CAUSE:")[1].split("CORRECTIVE_ACTION:")[0].strip()
            action_part = text.split("CORRECTIVE_ACTION:")[1].strip()
            root_cause, corrective_action = root_part, action_part
        return root_cause, corrective_action
    except Exception as e:
        return (f"Gemini request failed: {e}", "Check API key / model name / network access.")


# ====================================================================
# SPC HELPERS
# ====================================================================
def compute_xbar_r(values: pd.Series, subgroup_size: int):
    n = subgroup_size
    values = values.dropna().reset_index(drop=True)
    n_groups = len(values) // n
    if n_groups < 2:
        return None
    trimmed = values.iloc[: n_groups * n]
    groups = np.array(trimmed).reshape(n_groups, n)
    means = groups.mean(axis=1)
    ranges = groups.max(axis=1) - groups.min(axis=1)

    x_double_bar = means.mean()
    r_bar = ranges.mean()
    A2, D3, D4 = SPC_CONSTANTS.get(n, SPC_CONSTANTS[5])

    xbar_ucl = x_double_bar + A2 * r_bar
    xbar_lcl = x_double_bar - A2 * r_bar
    r_ucl = D4 * r_bar
    r_lcl = D3 * r_bar

    return {
        "means": means, "ranges": ranges,
        "x_double_bar": x_double_bar, "r_bar": r_bar,
        "xbar_ucl": xbar_ucl, "xbar_lcl": xbar_lcl,
        "r_ucl": r_ucl, "r_lcl": r_lcl,
        "n": n, "n_groups": n_groups,
    }


def compute_p_chart(inspections_df: pd.DataFrame):
    df = inspections_df.groupby("lot_id", as_index=False).agg(
        units_inspected=("units_inspected", "sum"),
        units_defective=("units_defective", "sum"),
    )
    df = df.sort_values("lot_id")
    df["p"] = df["units_defective"] / df["units_inspected"].replace(0, np.nan)
    p_bar = df["units_defective"].sum() / max(df["units_inspected"].sum(), 1)
    df["ucl"] = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / df["units_inspected"].replace(0, np.nan))
    df["lcl"] = (p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / df["units_inspected"].replace(0, np.nan))).clip(lower=0)
    return df, p_bar


# ====================================================================
# SIDEBAR
# ====================================================================
def sidebar_controls():
    st.sidebar.markdown("## ⚙️ System Configuration")

    st.sidebar.markdown("### 🔎 Detection Model (YOLOv11)")
    yolo_choice = st.sidebar.radio(
        "Weights source", ["Pretrained demo (yolo11n.pt)", "Upload custom .pt"],
        label_visibility="collapsed",
    )
    yolo_weights_path = "yolo11n.pt"
    if yolo_choice == "Upload custom .pt":
        up = st.sidebar.file_uploader("Custom YOLOv11 weights (.pt)", type=["pt"])
        if up is not None:
            yolo_weights_path = str(Path("uploaded_yolo.pt").resolve())
            with open(yolo_weights_path, "wb") as f:
                f.write(up.read())

    conf_thresh = st.sidebar.slider("Detection confidence threshold", 0.05, 0.95, 0.35, 0.05)

    st.sidebar.markdown("### 🧬 Fine-grained Classifier (EfficientNet-B2)")
    clf_upload = st.sidebar.file_uploader("Trained classifier weights (.pth)", type=["pth", "pt"])
    clf_weights_path = None
    if clf_upload is not None:
        clf_weights_path = str(Path("uploaded_classifier.pth").resolve())
        with open(clf_weights_path, "wb") as f:
            f.write(clf_upload.read())

    st.sidebar.markdown("### 🤖 Gemini Vision Explainer")
    gemini_key = st.sidebar.text_input("Gemini API key (optional)", type="password",
                                        help="Leave blank to disable AI root-cause analysis.")
    gemini_model = st.sidebar.text_input("Gemini model name", value="gemini-2.0-flash")
    enable_gemini = st.sidebar.checkbox("Enable AI defect explanation", value=bool(gemini_key))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Status")
    def pill(ok, label):
        cls = "vg-ok" if ok else "vg-bad"
        mark = "●" if ok else "○"
        st.sidebar.markdown(f'<span class="vg-pill {cls}">{mark} {label}</span>', unsafe_allow_html=True)
    pill(CV2_AVAILABLE, "OpenCV")
    pill(YOLO_AVAILABLE, "Ultralytics")
    pill(TORCH_AVAILABLE, "PyTorch")
    pill(GEMINI_SDK_AVAILABLE, "Gemini SDK")
    pill(bool(gemini_key) and enable_gemini, "Gemini Active")

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear defect database"):
        clear_database()
        st.sidebar.success("Database cleared.")

    return {
        "yolo_weights_path": yolo_weights_path,
        "conf_thresh": conf_thresh,
        "clf_weights_path": clf_weights_path,
        "gemini_key": gemini_key,
        "gemini_model": gemini_model,
        "enable_gemini": enable_gemini,
    }


# ====================================================================
# CORE FRAME-PROCESSING PIPELINE (shared by webcam + video tab)
# ====================================================================
def process_frame(frame_bgr, source_name, cfg, yolo_model, clf_bundle):
    """Runs YOLO -> crop -> EfficientNet-B2 -> (optional) Gemini -> DB log.
    Returns annotated frame + list of new detection records."""
    dets = run_yolo_inference(yolo_model, frame_bgr, cfg["conf_thresh"])
    h, w = frame_bgr.shape[:2]
    records = []

    for d in dets:
        x1, y1, x2, y2 = [max(0, int(v)) for v in d["xyxy"]]
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        crop_rgb = crop_bgr[:, :, ::-1]

        label, cls_conf = classify_defect(crop_rgb, clf_bundle)
        d["final_label"] = label

        bbox_norm = ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)
        area_pct = 100.0 * ((x2 - x1) * (y2 - y1)) / (w * h)

        img_name = f"{source_name}_{int(time.time()*1000)}.jpg"
        img_path = str(ARCHIVE_DIR / img_name)
        Image.fromarray(crop_rgb).save(img_path, quality=90)

        root_cause, corrective_action = "", ""
        if cfg["enable_gemini"] and cfg["gemini_key"]:
            root_cause, corrective_action = get_gemini_explanation(
                Image.fromarray(crop_rgb), label, cls_conf, cfg["gemini_key"], cfg["gemini_model"]
            )

        log_defect(source_name, label, d["conf"], cls_conf, bbox_norm, area_pct,
                   img_path, root_cause, corrective_action)
        records.append({**d, "label": label, "cls_conf": cls_conf,
                         "root_cause": root_cause, "corrective_action": corrective_action})

    log_inspection(source_name, units_inspected=1, units_defective=1 if dets else 0)
    annotated = draw_detections(frame_bgr, dets)
    return annotated, records


# ====================================================================
# TAB 1 — LIVE DETECTION (webcam / video upload)
# ====================================================================
def tab_live_detection(cfg):
    st.markdown("#### 🎥 Camera Input")

    if not CV2_AVAILABLE:
        st.error("OpenCV is not installed — camera/video processing is unavailable. "
                 "Install it with `pip install opencv-python-headless`.")
        return

    yolo_model = load_yolo_model(cfg["yolo_weights_path"]) if YOLO_AVAILABLE else None
    clf_bundle = load_classifier(cfg["clf_weights_path"]) if TORCH_AVAILABLE else None
    if not YOLO_AVAILABLE:
        st.warning("Ultralytics is not installed — run `pip install ultralytics` to enable YOLOv11 detection.")

    mode = st.radio("Source", ["📸 Webcam snapshot", "🎞️ Live webcam stream", "📁 Upload video file"],
                     horizontal=True)

    # ---------------- Webcam single-shot (works everywhere, incl. cloud) ----------------
    if mode == "📸 Webcam snapshot":
        st.caption("Best for cloud/browser deployments where continuous camera access is restricted.")
        shot = st.camera_input("Capture a frame")
        if shot is not None:
            pil_img = Image.open(shot).convert("RGB")
            frame_bgr = np.array(pil_img)[:, :, ::-1].copy()
            with st.spinner("Running YOLOv11 + EfficientNet-B2 + Gemini pipeline…"):
                annotated, records = process_frame(frame_bgr, "webcam_snapshot", cfg, yolo_model, clf_bundle)
            st.image(annotated[:, :, ::-1], caption="Annotated frame", use_container_width=True)
            _render_detection_records(records)

    # ---------------- Live webcam loop (local execution with an attached camera) ----------------
    elif mode == "🎞️ Live webcam stream":
        st.caption("Requires a local camera accessible to the machine running Streamlit "
                   "(`cv2.VideoCapture(0)`). Not available on most hosted/cloud deployments.")
        cam_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
        frame_skip = st.slider("Process every Nth frame", 1, 15, 5)
        col_a, col_b = st.columns(2)
        start = col_a.button("▶️ Start stream")
        stop = col_b.button("⏹ Stop stream")

        if start:
            st.session_state["run_camera"] = True
        if stop:
            st.session_state["run_camera"] = False

        frame_slot = st.empty()
        info_slot = st.empty()

        if st.session_state.get("run_camera"):
            cap = cv2.VideoCapture(int(cam_index))
            frame_count = 0
            try:
                while st.session_state.get("run_camera") and cap.isOpened():
                    ok, frame_bgr = cap.read()
                    if not ok:
                        info_slot.error("Could not read from camera.")
                        break
                    frame_count += 1
                    if frame_count % frame_skip == 0:
                        annotated, records = process_frame(frame_bgr, "webcam_live", cfg, yolo_model, clf_bundle)
                        frame_slot.image(annotated[:, :, ::-1], channels="RGB", use_container_width=True)
                        if records:
                            info_slot.warning(f"⚠️ {len(records)} defect(s) detected in this frame.")
                        else:
                            info_slot.success("No defects detected.")
                    else:
                        frame_slot.image(frame_bgr[:, :, ::-1], channels="RGB", use_container_width=True)
                    time.sleep(0.03)
            finally:
                cap.release()

    # ---------------- Video file upload ----------------
    else:
        video_file = st.file_uploader("Upload inspection video", type=["mp4", "avi", "mov", "mkv"])
        frame_skip = st.slider("Process every Nth frame", 1, 30, 10, key="video_skip")
        if video_file is not None:
            tmp_path = Path("uploaded_video") 
            tmp_path = tmp_path.with_suffix(Path(video_file.name).suffix or ".mp4")
            with open(tmp_path, "wb") as f:
                f.write(video_file.read())

            cap = cv2.VideoCapture(str(tmp_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            progress = st.progress(0)
            frame_slot = st.empty()
            summary_slot = st.empty()
            run = st.button("▶️ Process video")

            if run:
                frame_count = 0
                total_defects = 0
                all_records = []
                while cap.isOpened():
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                    frame_count += 1
                    if frame_count % frame_skip == 0:
                        annotated, records = process_frame(frame_bgr, video_file.name, cfg, yolo_model, clf_bundle)
                        total_defects += len(records)
                        all_records.extend(records)
                        frame_slot.image(annotated[:, :, ::-1], channels="RGB", use_container_width=True)
                    progress.progress(min(frame_count / total_frames, 1.0))
                cap.release()
                summary_slot.success(f"Done — {frame_count} frames scanned, {total_defects} defect(s) logged.")
                _render_detection_records(all_records)


def _render_detection_records(records):
    if not records:
        return
    st.markdown("##### 🧾 Detections in this frame")
    for i, r in enumerate(records):
        with st.expander(f"Defect {i+1}: {r['label'].upper()} — det {r['conf']:.0%} / cls {r['cls_conf']:.0%}"):
            if r.get("root_cause"):
                st.markdown(f"**Root cause:** {r['root_cause']}")
                st.markdown(f"**Corrective action:** {r['corrective_action']}")
            else:
                st.caption("Enable Gemini in the sidebar for AI root-cause analysis.")


# ====================================================================
# TAB 2 — TRAINING PIPELINE
# ====================================================================
def tab_training():
    st.markdown("#### 🏋️ YOLOv11 Custom Defect Training Pipeline")
    st.caption(
        "Trains a YOLOv11 model on your own defect dataset. Annotations must be in "
        "standard YOLO format (one `.txt` per image: `class x_center y_center width height`, "
        "all normalized 0–1)."
    )

    with st.expander("📁 Expected dataset layout", expanded=False):
        st.code(
            """dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/

# data.yaml
path: dataset
train: images/train
val: images/val
names:
  0: scratch
  1: dent
  2: discoloration
  3: contamination
""",
            language="yaml",
        )

    col1, col2 = st.columns(2)
    with col1:
        data_yaml = st.text_input("Path to data.yaml", value="dataset/data.yaml")
        base_model = st.selectbox(
            "Base model",
            ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
            index=0,
        )
        imgsz = st.selectbox("Image size", [320, 480, 640, 960], index=2)
    with col2:
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=100)
        batch = st.number_input("Batch size", min_value=1, max_value=256, value=16)
        device = st.selectbox("Device", ["cpu", "0", "0,1"], index=0)

    run_name = st.text_input("Run name", value="defect_qc_v1")

    st.warning(
        "Training runs synchronously inside this Streamlit process and will block the UI "
        "until it completes. For long production runs, prefer the equivalent CLI command:\n\n"
        f"`yolo detect train data={data_yaml} model={base_model} epochs={epochs} "
        f"imgsz={imgsz} batch={batch} device={device} name={run_name}`"
    )

    if st.button("🚀 Start training"):
        if not YOLO_AVAILABLE:
            st.error("Ultralytics is not installed. Run `pip install ultralytics`.")
            return
        if not os.path.exists(data_yaml):
            st.error(f"data.yaml not found at '{data_yaml}'. Update the path above.")
            return

        log_box = st.empty()
        buf = io.StringIO()
        with st.spinner("Training YOLOv11 — this may take a while…"):
            try:
                model = YOLO(base_model)
                with redirect_stdout(buf):
                    model.train(
                        data=data_yaml, epochs=int(epochs), imgsz=int(imgsz),
                        batch=int(batch), device=device, project=str(RUNS_DIR), name=run_name,
                    )
                st.success("Training complete.")
            except Exception as e:
                st.error(f"Training failed: {e}")
            finally:
                log_box.code(buf.getvalue()[-4000:] or "(no console output captured)")

        results_png = RUNS_DIR / run_name / "results.png"
        if results_png.exists():
            st.image(str(results_png), caption="Training curves", use_container_width=True)
        best_pt = RUNS_DIR / run_name / "weights" / "best.pt"
        if best_pt.exists():
            st.info(f"Best weights saved to: `{best_pt}` — upload this file in the sidebar to use it live.")


# ====================================================================
# TAB 3 — DEFECT DATABASE
# ====================================================================
def tab_database():
    st.markdown("#### 🗄️ Defect Database & Image Archive")
    df = fetch_defects_df()
    if df.empty:
        st.info("No defects logged yet. Run detection on the Live Detection tab.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total defects", len(df))
    c2.metric("Unique lots", df["lot_id"].nunique())
    c3.metric("Most common type", df["defect_type"].mode().iloc[0] if not df.empty else "—")
    c4.metric("Avg. classifier confidence", f"{df['cls_confidence'].mean():.0%}")

    with st.expander("🔍 Filter"):
        types = st.multiselect("Defect type", DEFECT_CLASSES, default=DEFECT_CLASSES)
        df = df[df["defect_type"].isin(types)]

    st.dataframe(
        df[["id", "timestamp", "lot_id", "source", "defect_type",
            "det_confidence", "cls_confidence", "area_pct", "root_cause", "corrective_action"]],
        use_container_width=True, height=320,
    )

    st.markdown("##### 🖼️ Image Archive Browser")
    row_ids = df["id"].tolist()
    if row_ids:
        selected = st.selectbox("Select a logged defect image", row_ids)
        row = df[df["id"] == selected].iloc[0]
        if row["image_path"] and os.path.exists(row["image_path"]):
            cols = st.columns([1, 2])
            cols[0].image(row["image_path"], caption=f'{row["defect_type"]} ({row["cls_confidence"]:.0%})',
                          use_container_width=True)
            with cols[1]:
                st.markdown(f"**Timestamp:** {row['timestamp']}")
                st.markdown(f"**Lot:** {row['lot_id']}  |  **Source:** {row['source']}")
                st.markdown(f"**Detection confidence:** {row['det_confidence']:.0%}")
                st.markdown(f"**Root cause:** {row['root_cause'] or '—'}")
                st.markdown(f"**Corrective action:** {row['corrective_action'] or '—'}")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export defect log (CSV)", csv, "defect_log.csv", "text/csv")


# ====================================================================
# TAB 4 — REAL-TIME DASHBOARD
# ====================================================================
def tab_dashboard():
    st.markdown("#### 📊 Real-Time QC Dashboard")
    df = fetch_defects_df()
    insp_df = fetch_inspections_df()

    if df.empty:
        st.info("No data yet — run detections to populate the dashboard.")
        return

    total_defects = len(df)
    total_inspected = int(insp_df["units_inspected"].sum()) if not insp_df.empty else total_defects
    defect_rate = total_defects / max(total_inspected, 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total inspected units", total_inspected)
    c2.metric("Total defects", total_defects)
    c3.metric("Overall defect rate", f"{defect_rate:.1%}")

    col1, col2 = st.columns(2)

    # --- Defect rate per hour ---
    with col1:
        st.markdown("**Defect rate per hour**")
        hourly = df.groupby(df["timestamp"].dt.floor("h")).size().reset_index(name="defects")
        if not insp_df.empty:
            insp_hourly = insp_df.groupby(insp_df["timestamp"].dt.floor("h"))["units_inspected"].sum().reset_index()
            hourly = hourly.merge(insp_hourly, on="timestamp", how="left")
            hourly["rate"] = hourly["defects"] / hourly["units_inspected"].replace(0, np.nan)
        else:
            hourly["rate"] = hourly["defects"]
        fig = px.bar(hourly, x="timestamp", y="defects", template="plotly_dark",
                     color_discrete_sequence=["#f5a623"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # --- Defect type distribution ---
    with col2:
        st.markdown("**Defect type distribution**")
        type_counts = df["defect_type"].value_counts().reset_index()
        type_counts.columns = ["defect_type", "count"]
        fig = px.pie(type_counts, names="defect_type", values="count", hole=0.45,
                     template="plotly_dark",
                     color="defect_type", color_discrete_map=DEFECT_COLORS)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # --- Heatmap on product silhouette ---
    st.markdown("**Defect location heatmap (product silhouette)**")
    fig = go.Figure()
    # generic product silhouette (rounded rectangle) drawn with shapes
    fig.add_shape(type="rect", x0=0.05, y0=0.05, x1=0.95, y1=0.95,
                  line=dict(color="#5b616a", width=3), fillcolor="rgba(90,97,106,0.12)")
    fig.add_trace(
        go.Histogram2dContour(
            x=df["bbox_x"], y=1 - df["bbox_y"],
            colorscale=[[0, "rgba(0,0,0,0)"], [0.3, "#3a2a12"], [0.6, "#c9741f"], [1.0, "#ff3b3b"]],
            contours=dict(showlines=False),
            showscale=True, opacity=0.85, ncontours=12,
        )
    )
    fig.add_trace(
        go.Scatter(x=df["bbox_x"], y=1 - df["bbox_y"], mode="markers",
                   marker=dict(size=5, color="#e7ecef", opacity=0.5), name="defects")
    )
    fig.update_xaxes(range=[0, 1], visible=False)
    fig.update_yaxes(range=[0, 1], visible=False, scaleanchor="x")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=420,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ====================================================================
# TAB 5 — SPC CHARTS
# ====================================================================
def tab_spc():
    st.markdown("#### 📈 Statistical Process Control")
    df = fetch_defects_df()
    insp_df = fetch_inspections_df()

    if df.empty:
        st.info("No data yet — run detections to populate SPC charts.")
        return

    st.markdown("##### X-bar & R Chart — defect size (% of frame area)")
    subgroup_size = st.slider("Subgroup size (n)", 2, 10, 5)
    spc = compute_xbar_r(df["area_pct"], subgroup_size)

    if spc is None:
        st.warning(f"Need at least {2*subgroup_size} logged defects for a stable X-bar/R chart. "
                   f"Currently have {len(df)}.")
    else:
        idx = list(range(1, spc["n_groups"] + 1))

        fig_x = go.Figure()
        fig_x.add_trace(go.Scatter(x=idx, y=spc["means"], mode="lines+markers", name="X-bar",
                                   line=dict(color="#f5a623")))
        fig_x.add_hline(y=spc["x_double_bar"], line=dict(color="#3ddc84", dash="dash"), annotation_text="CL")
        fig_x.add_hline(y=spc["xbar_ucl"], line=dict(color="#ff4d4d", dash="dot"), annotation_text="UCL")
        fig_x.add_hline(y=spc["xbar_lcl"], line=dict(color="#ff4d4d", dash="dot"), annotation_text="LCL")
        fig_x.update_layout(template="plotly_dark", title="X-bar Chart", height=320,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_x, use_container_width=True)

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=idx, y=spc["ranges"], mode="lines+markers", name="Range",
                                   line=dict(color="#35a6f5")))
        fig_r.add_hline(y=spc["r_bar"], line=dict(color="#3ddc84", dash="dash"), annotation_text="CL")
        fig_r.add_hline(y=spc["r_ucl"], line=dict(color="#ff4d4d", dash="dot"), annotation_text="UCL")
        fig_r.add_hline(y=spc["r_lcl"], line=dict(color="#ff4d4d", dash="dot"), annotation_text="LCL")
        fig_r.update_layout(template="plotly_dark", title="R Chart", height=320,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("##### P Chart — proportion defective per lot")
    if insp_df.empty:
        st.warning("No inspection tallies logged yet.")
    else:
        p_df, p_bar = compute_p_chart(insp_df)
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=p_df["lot_id"], y=p_df["p"], mode="lines+markers",
                                   name="p", line=dict(color="#f5a623")))
        fig_p.add_trace(go.Scatter(x=p_df["lot_id"], y=p_df["ucl"], mode="lines",
                                   name="UCL", line=dict(color="#ff4d4d", dash="dot")))
        fig_p.add_trace(go.Scatter(x=p_df["lot_id"], y=p_df["lcl"], mode="lines",
                                   name="LCL", line=dict(color="#ff4d4d", dash="dot")))
        fig_p.add_hline(y=p_bar, line=dict(color="#3ddc84", dash="dash"), annotation_text="p̄")
        fig_p.update_layout(template="plotly_dark", title="P Chart (per hourly lot)", height=360,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis_tickangle=-45)
        st.plotly_chart(fig_p, use_container_width=True)


# ====================================================================
# MAIN
# ====================================================================
def main():
    init_db()
    inject_css()
    header()

    cfg = sidebar_controls()

    tabs = st.tabs([
        "🎥 Live Detection", "🏋️ Training Pipeline", "🗄️ Defect Database",
        "📊 Dashboard", "📈 SPC Charts",
    ])
    with tabs[0]:
        tab_live_detection(cfg)
    with tabs[1]:
        tab_training()
    with tabs[2]:
        tab_database()
    with tabs[3]:
        tab_dashboard()
    with tabs[4]:
        tab_spc()

    st.markdown("---")
    st.caption("VisionGuard AI — demo QC pipeline. Bring your own trained YOLOv11 / "
              "EfficientNet-B2 weights and Gemini API key for production-grade accuracy.")


if __name__ == "__main__":
    main()
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import onnxruntime as ort
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
import base64

import torch
import torch.nn as nn
import torchvision.transforms as T
import mysql.connector
from mysql.connector import Error as MySQLError
import json
import datetime

app = Flask(__name__)
CORS(app)

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH    = "x3d_lvh.onnx"       # ONNX model (for fast inference)
PT_MODEL_PATH = "best_x3d_weighted.pth"  # PyTorch model (for Grad-CAM++ XAI)

# ============================================================
# MYSQL CONFIGURATION  — update these values
# ============================================================
DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",        
    "password": "",            
    "database": "healthapp"  
}

def get_db():
    """Return a fresh MySQL connection."""
    conn = mysql.connector.connect(**DB_CONFIG)
    return conn

def init_db():
    """Create database and table if they don't exist."""
    try:
        # Connect without specifying database first
        cfg_no_db = {k: v for k, v in DB_CONFIG.items() if k != "database"}
        conn = mysql.connector.connect(**cfg_no_db)
        cur  = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}`")
        conn.commit()
        cur.close()
        conn.close()

        # Now connect to the database and create table
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS lvh (
                id              INT AUTO_INCREMENT PRIMARY KEY,
                patient_id      VARCHAR(100)   NOT NULL,
                doctor_name     VARCHAR(150),
                doctor_comment  TEXT,
                video_filename  VARCHAR(255),
                prediction      TINYINT(1)     NOT NULL,
                prediction_label VARCHAR(50),
                lvh_probability FLOAT          NOT NULL,
                confidence      FLOAT,
                risk_level      VARCHAR(30),
                threshold       FLOAT,
                peak_frame      INT,
                high_act_count  INT,
                spread          VARCHAR(30),
                num_frames      INT,
                frame_importance JSON,
                top3_frames     JSON,
                xai_json        LONGTEXT,
                pdf_data        LONGBLOB,
                created_at      DATETIME       DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ MySQL database & table ready")
    except MySQLError as e:
        print(f"⚠️  MySQL init warning: {e}")

# Initialise DB on startup
init_db()

# ============================================================
# MODEL SETTINGS
# ============================================================
NUM_FRAMES    = 16
FRAME_SIZE    = 224
LVH_THRESHOLD = 0.37

MEAN = [0.43216, 0.394666, 0.37645]
STD  = [0.22803,  0.22145, 0.216989]

# ============================================================
# LOAD ONNX MODEL
# ============================================================
print("Loading ONNX model...")
ort_session = ort.InferenceSession(MODEL_PATH)
print("✅ ONNX model loaded")

# ============================================================
# LAZY-LOAD PYTORCH MODEL (only when /explain is called)
# ============================================================
_pt_model = None

def get_pt_model():
    global _pt_model
    if _pt_model is not None:
        return _pt_model

    print("Loading PyTorch model for Grad-CAM++...")
    model = torch.hub.load(
        "facebookresearch/pytorchvideo", "x3d_m", pretrained=False
    )
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(PT_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    _pt_model = model
    print(f"✅ PyTorch model loaded on {device}")
    return _pt_model

# ============================================================
# VIDEO PREPROCESSING (shared)
# ============================================================
def preprocess_video(video_path, num_frames=NUM_FRAMES, frame_size=FRAME_SIZE):
    """
    Returns:
      numpy_tensor  – (1, C, T, H, W) float32  [for ONNX]
      raw_frames    – list of (H, W, 3) uint8 RGB images
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError("Empty or unreadable video file")

    indices      = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    raw_frames   = [cv2.resize(frames[i], (frame_size, frame_size)) for i in indices]

    processed = []
    for f in raw_frames:
        f_f = f.astype(np.float32) / 255.0
        f_f = (f_f - MEAN) / STD
        processed.append(f_f)

    video_tensor = np.stack(processed)               # (T, H, W, C)
    video_tensor = np.transpose(video_tensor, (3, 0, 1, 2))  # (C, T, H, W)
    video_tensor = np.expand_dims(video_tensor, 0)   # (1, C, T, H, W)
    return video_tensor.astype(np.float32), raw_frames

# ============================================================
# GRAD-CAM++ ENGINE (PyTorch only)
# ============================================================
class GradCAMPlusPlus3D:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd(module, inp, out):
            self.activations = out.detach()
        def bwd(module, gin, gout):
            self.gradients = gout[0].detach()
        self.target_layer.register_forward_hook(fwd)
        self.target_layer.register_full_backward_hook(bwd)

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score  = output[0, target_class]
        score.backward()

        grads    = self.gradients
        acts     = self.activations
        grads_sq = grads ** 2
        grads_cu = grads ** 3
        denom    = 2.0 * grads_sq + (acts * grads_cu).sum(dim=(2,3,4), keepdim=True) + 1e-7
        alpha    = grads_sq / denom
        relu_g   = torch.relu(score.exp() * grads)
        weights  = (alpha * relu_g).sum(dim=(2,3,4), keepdim=True)
        cam_3d   = torch.relu((weights * acts).sum(dim=1, keepdim=True))

        cam_np = cam_3d.squeeze().detach().cpu().numpy()
        probs  = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        return cam_np, probs

def run_gradcam(video_path):
    """
    Runs Grad-CAM++ on the video.
    Returns:
      heatmap_b64_list  – list of 16 base64-encoded PNG overlays
      frame_importance  – list of 16 floats
      top3_frames       – list of top-3 frame indices (0-based)
      probs             – [no_lvh_prob, lvh_prob]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_pt_model()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((FRAME_SIZE, FRAME_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

    # Load video
    cap = cv2.VideoCapture(video_path)
    raw_all = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_all.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not raw_all:
        raise RuntimeError("Empty video")

    idx        = np.linspace(0, len(raw_all) - 1, NUM_FRAMES).astype(int)
    raw_frames = [cv2.resize(raw_all[i], (FRAME_SIZE, FRAME_SIZE)) for i in idx]
    tensors    = [transform(f) for f in raw_frames]
    clip       = torch.stack(tensors).permute(1, 0, 2, 3).unsqueeze(0).to(device)

    # Target layer: last res-block of block-4
    target_layer = model.blocks[4].res_blocks[-1].branch2.conv_b
    cam_engine   = GradCAMPlusPlus3D(model, target_layer)

    # Quick prediction to choose target class
    with torch.no_grad():
        logits_check = model(clip.clone())
        probs_check  = torch.softmax(logits_check, dim=1)[0]
    lvh_prob    = probs_check[1].item()
    target_cls  = 1 if lvh_prob >= LVH_THRESHOLD else 0

    # Run Grad-CAM++
    cam_3d, probs_arr = cam_engine.generate(
        clip.clone().requires_grad_(True), target_cls
    )

    # Normalise
    cam_min, cam_max = cam_3d.min(), cam_3d.max()
    cam_3d = (cam_3d - cam_min) / (cam_max - cam_min + 1e-8)

    # Per-frame importance
    frame_importance = cam_3d.mean(axis=(1, 2)).tolist()
    top3_idx         = np.argsort(frame_importance)[-3:][::-1].tolist()

    # Build blended heatmap images
    heatmap_b64 = []
    for fi in range(NUM_FRAMES):
        cam_r   = cv2.resize(cam_3d[fi], (FRAME_SIZE, FRAME_SIZE))
        hmap    = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
        hmap    = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(raw_frames[fi], 0.50, hmap, 0.50, 0)
        _, buf  = cv2.imencode(".png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        heatmap_b64.append(base64.b64encode(buf).decode("utf-8"))

    return heatmap_b64, frame_importance, top3_idx, probs_arr.tolist()


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": True, "threshold": LVH_THRESHOLD})


@app.route("/predict", methods=["POST"])
def predict():
    """Fast inference using ONNX model."""
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        video_file = request.files["video"]
        if video_file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            video_tensor, _ = preprocess_video(tmp_path)
            input_name = ort_session.get_inputs()[0].name
            outputs    = ort_session.run(None, {input_name: video_tensor})
            logits     = outputs[0][0]

            exp_l  = np.exp(logits - np.max(logits))
            probs  = exp_l / exp_l.sum()

            lvh_prob   = float(probs[1])
            prediction = int(lvh_prob >= LVH_THRESHOLD)

            return jsonify({
                "success":           True,
                "lvh_probability":   round(lvh_prob, 4),
                "prediction":        prediction,
                "prediction_label":  "LVH Detected" if prediction else "No LVH",
                "threshold":         LVH_THRESHOLD,
                "confidence":        round(float(max(probs)) * 100, 2),
            })
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/explain", methods=["POST"])
def explain():
    """
    Grad-CAM++ explainability endpoint.
    Runs PyTorch model + Grad-CAM++ and returns:
      - heatmap_frames : list[str]  base64 PNG overlays for all 16 frames
      - frame_importance: list[float]
      - top3_frames    : list[int]  0-based indices of top 3 frames
      - lvh_probability: float
      - prediction     : int
      - peak_frame     : int  (1-based)
      - high_act_count : int
      - spread         : "Focal" | "Diffuse"
    """
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        video_file = request.files["video"]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            heatmaps, frame_importance, top3_idx, probs = run_gradcam(tmp_path)

            lvh_prob    = float(probs[1])
            prediction  = int(lvh_prob >= LVH_THRESHOLD)
            fi_arr      = np.array(frame_importance)
            peak_frame  = int(np.argmax(fi_arr)) + 1
            high_act    = int((fi_arr > fi_arr.mean()).sum())
            spread      = "Focal" if high_act <= 4 else "Diffuse"

            risk = ("High" if lvh_prob >= 0.65 else
                    "Borderline" if lvh_prob >= 0.40 else "Low")

            return jsonify({
                "success":          True,
                "heatmap_frames":   heatmaps,           # list of 16 base64 PNGs
                "frame_importance": frame_importance,    # list of 16 floats
                "top3_frames":      top3_idx,            # 0-based indices
                "lvh_probability":  round(lvh_prob, 4),
                "prediction":       prediction,
                "prediction_label": "LVH Detected" if prediction else "No LVH",
                "threshold":        LVH_THRESHOLD,
                "peak_frame":       peak_frame,
                "high_act_count":   high_act,
                "spread":           spread,
                "risk_level":       risk,
                "confidence":       round(max(probs) * 100, 2),
                "num_frames":       NUM_FRAMES,
            })
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/report", methods=["POST"])
def generate_report():
    """Generate and download PDF without saving to DB."""
    try:
        import datetime
        data            = request.get_json()
        patient_id      = data.get("patient_id", "—")
        doctor_name     = data.get("doctor_name", "")
        doctor_comment  = data.get("doctor_comment", "")
        filename        = data.get("filename", "echocardiogram")
        report_date     = datetime.datetime.now().strftime("%d %B %Y  %H:%M")

        buf = _build_pdf(
            patient_id, doctor_name, doctor_comment, filename,
            float(data["lvh_probability"]), int(data["prediction"]),
            data["prediction_label"], float(data["threshold"]),
            float(data["confidence"]), int(data["peak_frame"]),
            int(data["high_act_count"]), data["spread"], data["risk_level"],
            int(data["num_frames"]), data["frame_importance"],
            data["top3_frames"], data["heatmap_frames"], report_date
        )
        return send_file(buf, mimetype="application/pdf",
                         as_attachment=True, download_name="lvh_clinical_report.pdf")
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500
        threshold       = float(data["threshold"])
        confidence      = float(data["confidence"])
        peak_frame      = int(data["peak_frame"])
        high_act_count  = int(data["high_act_count"])
        spread          = data["spread"]
        risk_level      = data["risk_level"]
        num_frames      = int(data["num_frames"])
        frame_importance= data["frame_importance"]
        top3_frames     = data["top3_frames"]          # 0-based
        heatmap_frames  = data["heatmap_frames"]       # base64 PNGs
        patient_id      = data.get("patient_id", "—")
        doctor_name     = data.get("doctor_name", "")
        doctor_comment  = data.get("doctor_comment", "")
        filename        = data.get("filename", "echocardiogram")
        report_date     = datetime.datetime.now().strftime("%d %B %Y  %H:%M")

        # ── colour palette ──────────────────────────────────────
        NAVY   = colors.HexColor("#1a237e")
        PURPLE = colors.HexColor("#4a148c")
        if lvh_prob >= 0.65:
            RISK_COL = colors.HexColor("#c62828")
            RISK_BG  = colors.HexColor("#ffebee")
        elif lvh_prob >= 0.40:
            RISK_COL = colors.HexColor("#e65100")
            RISK_BG  = colors.HexColor("#fff3e0")
        else:
            RISK_COL = colors.HexColor("#2e7d32")
            RISK_BG  = colors.HexColor("#e8f5e9")

        GREY_BG  = colors.HexColor("#f5f5f5")
        DARK_TXT = colors.HexColor("#212121")
        MID_TXT  = colors.HexColor("#555555")

        # ── styles ──────────────────────────────────────────────
        styles = getSampleStyleSheet()

        def style(name, parent="Normal", **kw):
            s = ParagraphStyle(name, parent=styles[parent], **kw)
            return s

        S_title   = style("S_title",   "Title",   fontSize=20, textColor=colors.white,   alignment=TA_CENTER, spaceAfter=2)
        S_sub     = style("S_sub",     "Normal",  fontSize=10, textColor=colors.HexColor("#90caf9"), alignment=TA_CENTER)
        S_h2      = style("S_h2",      "Heading2",fontSize=12, textColor=NAVY,  spaceBefore=10, spaceAfter=4)
        S_body    = style("S_body",    "Normal",  fontSize=9,  textColor=DARK_TXT, leading=14)
        S_bodyB   = style("S_bodyB",   "Normal",  fontSize=9,  textColor=DARK_TXT, leading=14, fontName="Helvetica-Bold")
        S_small   = style("S_small",   "Normal",  fontSize=8,  textColor=MID_TXT,  leading=12)
        S_disc    = style("S_disc",    "Normal",  fontSize=8,  textColor=colors.HexColor("#eceff1"), leading=13, alignment=TA_CENTER)
        S_diag    = style("S_diag",    "Normal",  fontSize=18, textColor=RISK_COL, fontName="Helvetica-Bold", alignment=TA_CENTER)
        S_prob    = style("S_prob",    "Normal",  fontSize=28, textColor=RISK_COL, fontName="Helvetica-Bold", alignment=TA_CENTER)
        S_label   = style("S_label",   "Normal",  fontSize=8,  textColor=MID_TXT,  alignment=TA_CENTER)
        S_val     = style("S_val",     "Normal",  fontSize=13, textColor=NAVY,     fontName="Helvetica-Bold", alignment=TA_CENTER)

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=15*mm, rightMargin=15*mm,
            topMargin=12*mm, bottomMargin=12*mm,
            title="LVH Detection Clinical Report"
        )
        W = A4[0] - 30*mm   # usable width

        story = []

        # ════════════════════════════════════════════════════
        # 1. HEADER BANNER
        # ════════════════════════════════════════════════════
        header_tbl = Table(
            [[Paragraph("&#127880;  CARDIAC AI SCREENING REPORT  —  LVH DETECTION", S_title)],
             [Paragraph("Gradient-weighted Class Activation Mapping (Grad-CAM++)  •  X3D Echocardiogram Analysis", S_sub)]],
            colWidths=[W]
        )
        header_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), NAVY),
            ("TOPPADDING",    (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("LEFTPADDING",   (0,0), (-1,-1), 12),
            ("RIGHTPADDING",  (0,0), (-1,-1), 12),
            ("ROUNDEDCORNERS", [6]),
        ]))
        story.append(header_tbl)
        story.append(Spacer(1, 6*mm))

        # ── meta row ────────────────────────────────────────
        meta_tbl = Table(
            [[Paragraph(f"<b>Patient ID:</b> {patient_id}", S_body),
              Paragraph(f"<b>File:</b> {filename}", S_body),
              Paragraph(f"<b>Report Date:</b> {report_date}", S_body)]],
            colWidths=[W/3, W/3, W/3]
        )
        meta_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), GREY_BG),
            ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#e0e0e0")),
            ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING", (0,0), (-1,-1), 10),
        ]))
        story.append(meta_tbl)
        story.append(Spacer(1, 3*mm))

        # ── doctor comment row ───────────────────────────────
        if doctor_name or doctor_comment:
            doc_rows = []
            if doctor_name:
                doc_rows.append([
                    Paragraph("<b>Reviewing Doctor:</b>", S_body),
                    Paragraph(doctor_name, S_body)
                ])
            if doctor_comment:
                doc_rows.append([
                    Paragraph("<b>Doctor's Comment:</b>", S_body),
                    Paragraph(doctor_comment, S_body)
                ])
            doc_tbl = Table(doc_rows, colWidths=[W*0.22, W*0.78])
            doc_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#e8eaf6")),
                ("BOX", (0,0), (-1,-1), 0.8, colors.HexColor("#7986cb")),
                ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#c5cae9")),
                ("TOPPADDING", (0,0), (-1,-1), 6),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                ("LEFTPADDING", (0,0), (-1,-1), 10),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
            ]))
            story.append(doc_tbl)
            story.append(Spacer(1, 3*mm))

        story.append(Spacer(1, 2*mm))

        # ════════════════════════════════════════════════════
        # 2. DIAGNOSIS  |  PROBABILITY  |  STATS
        # ════════════════════════════════════════════════════
        icon = "WARNING: " if prediction == 1 else "OK: "

        diag_inner = Table(
            [[Paragraph(icon, style("ic", fontSize=22, alignment=TA_CENTER))],
             [Paragraph(prediction_label, S_diag)],
             [Paragraph(f"Risk Level: <b>{risk_level}</b>", style("rl", fontSize=10, textColor=RISK_COL, alignment=TA_CENTER))],
             [Paragraph(f"AI Confidence: {confidence}%", S_label)]],
            colWidths=[W*0.30]
        )
        diag_inner.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), RISK_BG),
            ("BOX",           (0,0), (-1,-1), 1.5, RISK_COL),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ]))

        prob_inner = Table(
            [[Paragraph("LVH PROBABILITY", style("lp", fontSize=9, textColor=MID_TXT, alignment=TA_CENTER, fontName="Helvetica-Bold"))],
             [Paragraph(f"{round(lvh_prob*100, 1)}%", S_prob)],
             [Paragraph(f"Threshold: {round(threshold*100)}%", S_label)],
             [Paragraph(f"{'Above' if lvh_prob >= threshold else 'Below'} decision threshold", S_small)]],
            colWidths=[W*0.30]
        )
        prob_inner.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), colors.white),
            ("BOX",           (0,0), (-1,-1), 0.5, colors.HexColor("#e0e0e0")),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ]))

        fi_arr   = np.array(frame_importance)
        max_fi   = fi_arr.max()
        stats_rows = [
            ["Frames Analysed",     str(num_frames)],
            ["Peak Activity Frame", f"Frame {peak_frame}"],
            ["High-Activity Frames",f"{high_act_count} / {num_frames}"],
            ["Activation Pattern",  spread],
            ["Decision Threshold",  f"{round(threshold*100)}%"],
            ["XAI Method",          "Grad-CAM++"],
        ]
        stats_data = [[Paragraph(r[0], S_small), Paragraph(f"<b>{r[1]}</b>", style("sv", fontSize=9, textColor=NAVY, fontName="Helvetica-Bold"))]
                      for r in stats_rows]
        stats_inner = Table(stats_data, colWidths=[W*0.20, W*0.16])
        stats_inner.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), GREY_BG),
            ("BOX",           (0,0), (-1,-1), 0.5, colors.HexColor("#e0e0e0")),
            ("ROWBACKGROUNDS",(0,0), (-1,-1), [colors.white, GREY_BG]),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))

        top_row = Table(
            [[diag_inner, prob_inner, stats_inner]],
            colWidths=[W*0.32, W*0.32, W*0.36],
            hAlign="LEFT"
        )
        top_row.setStyle(TableStyle([
            ("VALIGN",       (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING",  (0,0), (-1,-1), 3),
            ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ]))
        story.append(top_row)
        story.append(Spacer(1, 5*mm))

        # ════════════════════════════════════════════════════
        # 3. FRAME-BY-FRAME BAR CHART (drawn with ReportLab)
        # ════════════════════════════════════════════════════
        story.append(Paragraph("FRAME-BY-FRAME CARDIAC ACTIVITY", S_h2))
        story.append(Paragraph(
            "Which moments of the heartbeat were most important to the AI decision.",
            S_small))
        story.append(Spacer(1, 2*mm))

        chart_h   = 40*mm
        chart_w   = W
        bar_w     = chart_w / (num_frames * 1.4)
        gap       = bar_w * 0.4
        max_bar_h = chart_h - 8*mm
        mean_fi   = fi_arr.mean()

        drw = Drawing(chart_w, chart_h + 6*mm)

        for i, v in enumerate(frame_importance):
            x   = i * (bar_w + gap) + gap/2
            bh  = max(1, (v / (max_fi + 1e-8)) * max_bar_h)
            rank = top3_frames.index(i) if i in top3_frames else -1
            if rank == 0:
                col = colors.HexColor("#c62828")
            elif rank == 1:
                col = colors.HexColor("#ef5350")
            elif rank == 2:
                col = colors.HexColor("#ef9a9a")
            else:
                col = colors.HexColor("#b0bec5")
            drw.add(Rect(x, 6*mm, bar_w, bh, fillColor=col, strokeColor=colors.white, strokeWidth=0.5))

        # mean line
        mean_y = (mean_fi / (max_fi + 1e-8)) * max_bar_h + 6*mm
        from reportlab.graphics.shapes import Line, String
        drw.add(Line(0, mean_y, chart_w, mean_y,
                     strokeColor=colors.HexColor("#78909c"),
                     strokeWidth=1, strokeDashArray=[3,3]))

        story.append(drw)

        # legend
        legend_data = [[
            Paragraph("<font color='#c62828'>&#9632;</font> Rank #1 — Most relevant", S_small),
            Paragraph("<font color='#ef5350'>&#9632;</font> Rank #2", S_small),
            Paragraph("<font color='#ef9a9a'>&#9632;</font> Rank #3", S_small),
            Paragraph("<font color='#b0bec5'>&#9632;</font> Lower relevance", S_small),
        ]]
        leg_tbl = Table(legend_data, colWidths=[W/4]*4)
        leg_tbl.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)]))
        story.append(leg_tbl)
        story.append(Spacer(1, 5*mm))

        # ════════════════════════════════════════════════════
        # 4. TOP-3 HEATMAP FRAMES
        # ════════════════════════════════════════════════════
        story.append(Paragraph("GRAD-CAM++ HEATMAP OVERLAYS — TOP 3 FRAMES", S_h2))

        rank_cols  = ["#c62828", "#ef5350", "#ef9a9a"]
        rank_labels= ["MOST RELEVANT", "2ND MOST RELEVANT", "3RD MOST RELEVANT"]
        img_w      = W / 3 - 4*mm
        img_cells  = []
        cap_cells  = []

        for rank, fi in enumerate(top3_frames[:3]):
            img_bytes = base64.b64decode(heatmap_frames[fi])
            img_buf   = io.BytesIO(img_bytes)
            rl_img    = RLImage(img_buf, width=img_w, height=img_w)
            pct       = round((frame_importance[fi] / (max_fi + 1e-8)) * 100)
            img_cells.append(rl_img)
            cap_cells.append(Paragraph(
                f"<font color='{rank_cols[rank]}'><b>{rank_labels[rank]}</b></font><br/>"
                f"Frame {fi+1}  |  Activity: {pct}% of peak",
                style(f"cap{rank}", fontSize=8, alignment=TA_CENTER, textColor=DARK_TXT)
            ))

        hmap_tbl = Table(
            [img_cells, cap_cells],
            colWidths=[img_w + 3*mm] * 3
        )
        hmap_tbl.setStyle(TableStyle([
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ("TOPPADDING",    (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ("BOX",           (0,0), (0,-1), 2, colors.HexColor(rank_cols[0])),
            ("BOX",           (1,0), (1,-1), 2, colors.HexColor(rank_cols[1])),
            ("BOX",           (2,0), (2,-1), 2, colors.HexColor(rank_cols[2])),
        ]))
        story.append(hmap_tbl)
        story.append(Spacer(1, 5*mm))

        # ════════════════════════════════════════════════════
        # 5. HEATMAP COLOUR LEGEND
        # ════════════════════════════════════════════════════
        story.append(Paragraph("HOW TO READ THE HEATMAP COLOURS", S_h2))

        colour_guide = [
            ("#c62828", "RED / ORANGE", "The AI is focusing HERE strongly. These areas show heart wall features most linked to LVH."),
            ("#f9a825", "YELLOW",        "Moderate AI attention. Moderately relevant features detected in this region."),
            ("#1565c0", "BLUE / DARK",   "Low AI attention. The AI considers these areas less relevant for LVH diagnosis."),
        ]
        cg_cells = []
        for col, name, desc in colour_guide:
            cell = Table(
                [[Paragraph(f"<font color='{col}'><b>&#9632;  {name}</b></font>", S_small)],
                 [Paragraph(desc, S_small)]],
                colWidths=[W/3 - 4*mm]
            )
            cell.setStyle(TableStyle([
                ("BOX",           (0,0), (-1,-1), 1.5, colors.HexColor(col)),
                ("BACKGROUND",    (0,0), (-1,-1), colors.white),
                ("TOPPADDING",    (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                ("LEFTPADDING",   (0,0), (-1,-1), 7),
            ]))
            cg_cells.append(cell)

        cg_tbl = Table([cg_cells], colWidths=[W/3]*3)
        cg_tbl.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2)]))
        story.append(cg_tbl)
        story.append(Spacer(1, 5*mm))

        # ════════════════════════════════════════════════════
        # 6. CLINICAL INTERPRETATION
        # ════════════════════════════════════════════════════
        story.append(Paragraph("CLINICAL INTERPRETATION", S_h2))

        direction = "above" if lvh_prob >= threshold else "below"
        if prediction == 1:
            interp = [
                ("<b>What the AI found:</b>",
                 f"The AI detected cardiac features consistent with Left Ventricular Hypertrophy (LVH) "
                 f"in {high_act_count} of {num_frames} analysed frames ({round(lvh_prob*100, 1)}% probability). "
                 f"Activation was {spread.lower()}, peaking at Frame {peak_frame}."),
                ("<b>What this means:</b>",
                 "LVH means the muscle wall of the left ventricle appears thicker than normal. "
                 "This can be caused by high blood pressure, heart valve disease, or other conditions."),
                ("<b>Recommended next steps:</b>",
                 "• Formal measurement of septal thickness (IVSd) and posterior wall thickness (LVPWd)<br/>"
                 "• Left ventricular mass index (LVMI) calculation<br/>"
                 "• Blood pressure evaluation and management review<br/>"
                 "• Cardiology specialist referral if not already under care"),
            ]
        else:
            interp = [
                ("<b>What the AI found:</b>",
                 f"No dominant LVH-associated patterns detected. LVH probability ({round(lvh_prob*100, 1)}%) "
                 f"is {direction} the {round(threshold*100)}% decision threshold. "
                 "Cardiac wall regions appear within normal range."),
                ("<b>What this means:</b>",
                 "The left ventricle does not show the typical thickening pattern associated with LVH. "
                 "This is a reassuring result."),
                ("<b>Recommended next steps:</b>",
                 "• Continue routine cardiac monitoring as clinically indicated<br/>"
                 "• Maintain blood pressure within target range<br/>"
                 "• Repeat echocardiogram if symptoms change or risk factors worsen"),
            ]

        interp_data = []
        for heading, body in interp:
            interp_data.append([
                Paragraph(heading, S_bodyB),
                Paragraph(body, S_body)
            ])

        interp_tbl = Table(interp_data, colWidths=[W*0.22, W*0.78])
        interp_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), RISK_BG),
            ("BOX",           (0,0), (-1,-1), 0.5, RISK_COL),
            ("INNERGRID",     (0,0), (-1,-1), 0.3, colors.HexColor("#e0e0e0")),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        story.append(interp_tbl)
        story.append(Spacer(1, 5*mm))

        # ════════════════════════════════════════════════════
        # 7. DISCLAIMER FOOTER
        # ════════════════════════════════════════════════════
        disc_tbl = Table(
            [[Paragraph("<b>WARNING:  IMPORTANT NOTICE</b>", style("dh", fontSize=9, textColor=colors.HexColor("#ffcc02"), alignment=TA_CENTER))],
             [Paragraph(
                 "This report is generated by an AI system and is intended as a clinical decision SUPPORT tool only. "
                 "It does NOT replace professional medical judgement. All findings must be reviewed and confirmed by a "
                 "qualified cardiologist before any clinical decision, treatment, or diagnosis is made.",
                 S_disc
             )]],
            colWidths=[W]
        )
        disc_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#37474f")),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 12),
            ("RIGHTPADDING",  (0,0), (-1,-1), 12),
            ("ROUNDEDCORNERS",[4]),
        ]))
        story.append(disc_tbl)

        # ── build ────────────────────────────────────────────
        doc.build(story)
        buf.seek(0)

        return send_file(
            buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="lvh_clinical_report.pdf"
        )

    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/save-report", methods=["POST"])
def save_report():
    """
    Generates the PDF (same as /report) AND saves everything to MySQL.
    POST body: XAI JSON  +  patient_id, doctor_name, doctor_comment, filename
    Returns: { success, record_id, pdf_base64 }
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, Image as RLImage
        )
        from reportlab.graphics.shapes import Drawing, Rect, Line

        data             = request.get_json()
        patient_id       = data.get("patient_id", "Unknown").strip() or "Unknown"
        doctor_name      = data.get("doctor_name", "").strip()
        doctor_comment   = data.get("doctor_comment", "").strip()
        filename         = data.get("filename", "echocardiogram")

        lvh_prob         = float(data["lvh_probability"])
        prediction       = int(data["prediction"])
        prediction_label = data["prediction_label"]
        threshold        = float(data["threshold"])
        confidence       = float(data["confidence"])
        peak_frame       = int(data["peak_frame"])
        high_act_count   = int(data["high_act_count"])
        spread           = data["spread"]
        risk_level       = data["risk_level"]
        num_frames       = int(data["num_frames"])
        frame_importance = data["frame_importance"]
        top3_frames      = data["top3_frames"]
        heatmap_frames   = data["heatmap_frames"]
        report_date      = datetime.datetime.now().strftime("%d %B %Y  %H:%M")

        # ── build PDF (same logic as /report, includes doctor section) ──
        pdf_buf = _build_pdf(
            patient_id, doctor_name, doctor_comment, filename,
            lvh_prob, prediction, prediction_label, threshold, confidence,
            peak_frame, high_act_count, spread, risk_level,
            num_frames, frame_importance, top3_frames, heatmap_frames,
            report_date
        )

        pdf_bytes = pdf_buf.getvalue()

        # ── save to MySQL ───────────────────────────────────────
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO reports (
                patient_id, doctor_name, doctor_comment, video_filename,
                prediction, prediction_label, lvh_probability, confidence,
                risk_level, threshold, peak_frame, high_act_count, spread,
                num_frames, frame_importance, top3_frames, xai_json, pdf_data
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            patient_id, doctor_name, doctor_comment, filename,
            prediction, prediction_label, lvh_prob, confidence,
            risk_level, threshold, peak_frame, high_act_count, spread,
            num_frames,
            json.dumps(frame_importance),
            json.dumps(top3_frames),
            json.dumps({k: v for k, v in data.items() if k != "heatmap_frames"}),
            pdf_bytes
        ))
        conn.commit()
        record_id = cur.lastrowid
        cur.close()
        conn.close()

        return jsonify({
            "success":    True,
            "record_id":  record_id,
            "pdf_base64": base64.b64encode(pdf_bytes).decode("utf-8"),
            "message":    f"Report saved with ID {record_id}"
        })

    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/reports", methods=["GET"])
def list_reports():
    """Return all saved reports (without PDF blob) for a records list."""
    try:
        conn = get_db()
        cur  = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT id, patient_id, doctor_name, doctor_comment,
                   video_filename, prediction, prediction_label,
                   lvh_probability, confidence, risk_level, threshold,
                   peak_frame, high_act_count, spread, created_at
            FROM reports ORDER BY created_at DESC
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        # Convert datetime to string for JSON
        for r in rows:
            if isinstance(r.get("created_at"), datetime.datetime):
                r["created_at"] = r["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        return jsonify({"success": True, "reports": rows})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/reports/<int:report_id>/pdf", methods=["GET"])
def download_report_pdf(report_id):
    """Stream the saved PDF for a given report ID."""
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("SELECT pdf_data, patient_id, created_at FROM reports WHERE id=%s", (report_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            return jsonify({"error": "Report not found"}), 404
        pdf_data, patient_id, created_at = row
        fname = f"lvh_report_{patient_id}_{report_id}.pdf"
        return send_file(
            io.BytesIO(pdf_data),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=fname
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# SHARED PDF BUILDER  (used by /report and /save-report)
# ============================================================
def _build_pdf(patient_id, doctor_name, doctor_comment, filename,
               lvh_prob, prediction, prediction_label, threshold, confidence,
               peak_frame, high_act_count, spread, risk_level,
               num_frames, frame_importance, top3_frames, heatmap_frames,
               report_date):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage
    )
    from reportlab.graphics.shapes import Drawing, Rect, Line

    NAVY    = colors.HexColor("#1a237e")
    PURPLE  = colors.HexColor("#4a148c")
    if lvh_prob >= 0.65:
        RISK_COL = colors.HexColor("#c62828"); RISK_BG = colors.HexColor("#ffebee")
    elif lvh_prob >= 0.40:
        RISK_COL = colors.HexColor("#e65100"); RISK_BG = colors.HexColor("#fff3e0")
    else:
        RISK_COL = colors.HexColor("#2e7d32"); RISK_BG = colors.HexColor("#e8f5e9")

    GREY_BG  = colors.HexColor("#f5f5f5")
    DARK_TXT = colors.HexColor("#212121")
    MID_TXT  = colors.HexColor("#555555")

    styles = getSampleStyleSheet()
    def style(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=styles[parent], **kw)

    S_title  = style("S_title",  "Title",   fontSize=20, textColor=colors.white,   alignment=TA_CENTER)
    S_sub    = style("S_sub",    "Normal",  fontSize=10, textColor=colors.HexColor("#90caf9"), alignment=TA_CENTER)
    S_h2     = style("S_h2",     "Heading2",fontSize=12, textColor=NAVY, spaceBefore=10, spaceAfter=4)
    S_body   = style("S_body",   "Normal",  fontSize=9,  textColor=DARK_TXT, leading=14)
    S_bodyB  = style("S_bodyB",  "Normal",  fontSize=9,  textColor=DARK_TXT, leading=14, fontName="Helvetica-Bold")
    S_small  = style("S_small",  "Normal",  fontSize=8,  textColor=MID_TXT,  leading=12)
    S_disc   = style("S_disc",   "Normal",  fontSize=8,  textColor=colors.HexColor("#eceff1"), leading=13, alignment=TA_CENTER)
    S_diag   = style("S_diag",   "Normal",  fontSize=18, textColor=RISK_COL, fontName="Helvetica-Bold", alignment=TA_CENTER)
    S_prob   = style("S_prob",   "Normal",  fontSize=28, textColor=RISK_COL, fontName="Helvetica-Bold", alignment=TA_CENTER)
    S_label  = style("S_label",  "Normal",  fontSize=8,  textColor=MID_TXT,  alignment=TA_CENTER)
    S_val    = style("S_val",    "Normal",  fontSize=13, textColor=NAVY,     fontName="Helvetica-Bold", alignment=TA_CENTER)
    S_comment= style("S_comment","Normal",  fontSize=9,  textColor=DARK_TXT, leading=14,
                     backColor=colors.HexColor("#fffde7"), leftIndent=6)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=12*mm, bottomMargin=12*mm,
        title="LVH Detection Clinical Report"
    )
    W     = A4[0] - 30*mm
    story = []

    # ── 1. Header ──────────────────────────────────────────
    hdr = Table(
        [[Paragraph("&#127880;  CARDIAC AI SCREENING REPORT  —  LVH DETECTION", S_title)],
         [Paragraph("Gradient-weighted Class Activation Mapping (Grad-CAM++)  •  X3D Echocardiogram Analysis", S_sub)]],
        colWidths=[W]
    )
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), NAVY),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 12),
        ("RIGHTPADDING",  (0,0),(-1,-1), 12),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 4*mm))

    # ── 2. Meta row ────────────────────────────────────────
    meta = Table([[
        Paragraph(f"<b>Patient ID:</b>  {patient_id}", S_body),
        Paragraph(f"<b>Doctor:</b>  {doctor_name or '—'}", S_body),
        Paragraph(f"<b>File:</b>  {filename}", S_body),
        Paragraph(f"<b>Date:</b>  {report_date}", S_body),
    ]], colWidths=[W*0.22, W*0.23, W*0.30, W*0.25])
    meta.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), GREY_BG),
        ("BOX",           (0,0),(-1,-1), 0.5, colors.HexColor("#e0e0e0")),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
    ]))
    story.append(meta)
    story.append(Spacer(1, 4*mm))

    # ── 3. Doctor comment box (if provided) ────────────────
    if doctor_comment:
        story.append(Paragraph("DOCTOR'S COMMENT", S_h2))
        cmt_tbl = Table(
            [[Paragraph(f"<i>{doctor_comment}</i>", S_comment)]],
            colWidths=[W]
        )
        cmt_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), colors.HexColor("#fffde7")),
            ("BOX",           (0,0),(-1,-1), 1.2, colors.HexColor("#f9a825")),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
            ("RIGHTPADDING",  (0,0),(-1,-1), 10),
            ("TOPPADDING",    (0,0),(-1,-1), 8),
            ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ]))
        story.append(cmt_tbl)
        story.append(Spacer(1, 4*mm))

    # ── 4. Diagnosis | Probability | Stats ─────────────────
    diag_inner = Table([
        [Paragraph("&#9888;" if prediction == 1 else "&#10003;", style("ic2", fontSize=22, alignment=TA_CENTER))],
        [Paragraph(prediction_label, S_diag)],
        [Paragraph(f"Risk Level: <b>{risk_level}</b>", style("rl2", fontSize=10, textColor=RISK_COL, alignment=TA_CENTER))],
        [Paragraph(f"AI Confidence: {confidence}%", S_label)],
    ], colWidths=[W*0.30])
    diag_inner.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), RISK_BG),
        ("BOX",(0,0),(-1,-1),1.5,RISK_COL),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))

    prob_inner = Table([
        [Paragraph("LVH PROBABILITY", style("lp2", fontSize=9, textColor=MID_TXT, alignment=TA_CENTER, fontName="Helvetica-Bold"))],
        [Paragraph(f"{round(lvh_prob*100,1)}%", S_prob)],
        [Paragraph(f"Threshold: {round(threshold*100)}%", S_label)],
        [Paragraph(f"{'Above' if lvh_prob>=threshold else 'Below'} threshold", S_small)],
    ], colWidths=[W*0.30])
    prob_inner.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.white),
        ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#e0e0e0")),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))

    fi_arr = np.array(frame_importance)
    stats_rows = [
        ["Frames Analysed",      str(num_frames)],
        ["Peak Activity Frame",  f"Frame {peak_frame}"],
        ["High-Activity Frames", f"{high_act_count} / {num_frames}"],
        ["Activation Pattern",   spread],
        ["Decision Threshold",   f"{round(threshold*100)}%"],
        ["XAI Method",           "Grad-CAM++"],
    ]
    stats_inner = Table(
        [[Paragraph(r[0], S_small),
          Paragraph(f"<b>{r[1]}</b>", style(f"sv2{i}", fontSize=9, textColor=NAVY, fontName="Helvetica-Bold"))]
         for i, r in enumerate(stats_rows)],
        colWidths=[W*0.20, W*0.16]
    )
    stats_inner.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),GREY_BG),
        ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#e0e0e0")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white,GREY_BG]),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("LEFTPADDING",(0,0),(-1,-1),8),
    ]))

    top_row = Table([[diag_inner, prob_inner, stats_inner]],
                    colWidths=[W*0.32, W*0.32, W*0.36])
    top_row.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),
    ]))
    story.append(top_row)
    story.append(Spacer(1, 5*mm))

    # ── 5. Temporal bar chart ───────────────────────────────
    story.append(Paragraph("FRAME-BY-FRAME CARDIAC ACTIVITY", S_h2))
    story.append(Paragraph("Which moments of the heartbeat were most important to the AI.", S_small))
    story.append(Spacer(1, 2*mm))

    max_fi   = fi_arr.max()
    chart_h  = 40*mm
    bar_w    = W / (num_frames * 1.4)
    gap      = bar_w * 0.4
    max_bar_h= chart_h - 8*mm
    mean_fi  = fi_arr.mean()

    drw = Drawing(W, chart_h + 6*mm)
    for i, v in enumerate(frame_importance):
        x   = i * (bar_w + gap) + gap/2
        bh  = max(1, (v / (max_fi + 1e-8)) * max_bar_h)
        rank = top3_frames.index(i) if i in top3_frames else -1
        col = (colors.HexColor("#c62828") if rank==0 else
               colors.HexColor("#ef5350") if rank==1 else
               colors.HexColor("#ef9a9a") if rank==2 else
               colors.HexColor("#b0bec5"))
        drw.add(Rect(x, 6*mm, bar_w, bh, fillColor=col, strokeColor=colors.white, strokeWidth=0.5))
    mean_y = (mean_fi / (max_fi + 1e-8)) * max_bar_h + 6*mm
    drw.add(Line(0, mean_y, W, mean_y, strokeColor=colors.HexColor("#78909c"),
                 strokeWidth=1, strokeDashArray=[3,3]))
    story.append(drw)

    leg = Table([[
        Paragraph("<font color='#c62828'>&#9632;</font> Rank #1", S_small),
        Paragraph("<font color='#ef5350'>&#9632;</font> Rank #2", S_small),
        Paragraph("<font color='#ef9a9a'>&#9632;</font> Rank #3", S_small),
        Paragraph("<font color='#b0bec5'>&#9632;</font> Lower relevance", S_small),
    ]], colWidths=[W/4]*4)
    story.append(leg)
    story.append(Spacer(1, 5*mm))

    # ── 6. Top-3 heatmaps ──────────────────────────────────
    story.append(Paragraph("GRAD-CAM++ HEATMAP OVERLAYS — TOP 3 FRAMES", S_h2))
    rank_cols  = ["#c62828","#ef5350","#ef9a9a"]
    rank_labels= ["MOST RELEVANT","2ND MOST RELEVANT","3RD MOST RELEVANT"]
    img_w = W / 3 - 4*mm
    img_cells, cap_cells = [], []
    for rank, fi in enumerate(top3_frames[:3]):
        ib  = base64.b64decode(heatmap_frames[fi])
        img = RLImage(io.BytesIO(ib), width=img_w, height=img_w)
        pct = round((frame_importance[fi] / (max_fi + 1e-8)) * 100)
        img_cells.append(img)
        cap_cells.append(Paragraph(
            f"<font color='{rank_cols[rank]}'><b>{rank_labels[rank]}</b></font><br/>"
            f"Frame {fi+1}  |  {pct}% of peak",
            style(f"cap2{rank}", fontSize=8, alignment=TA_CENTER, textColor=DARK_TXT)
        ))
    hmap_tbl = Table([img_cells, cap_cells], colWidths=[img_w+3*mm]*3)
    hmap_tbl.setStyle(TableStyle([
        ("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("BOX",(0,0),(0,-1),2,colors.HexColor(rank_cols[0])),
        ("BOX",(1,0),(1,-1),2,colors.HexColor(rank_cols[1])),
        ("BOX",(2,0),(2,-1),2,colors.HexColor(rank_cols[2])),
    ]))
    story.append(hmap_tbl)
    story.append(Spacer(1, 5*mm))

    # ── 7. Colour legend ───────────────────────────────────
    story.append(Paragraph("HOW TO READ THE HEATMAP COLOURS", S_h2))
    cg_cells = []
    for col, name, desc in [
        ("#c62828","RED / ORANGE","Strong AI focus — heart wall features linked to LVH."),
        ("#f9a825","YELLOW","Moderate AI attention — relevant features detected."),
        ("#1565c0","BLUE / DARK","Low attention — less relevant for LVH diagnosis."),
    ]:
        cell = Table([
            [Paragraph(f"<font color='{col}'><b>&#9632;  {name}</b></font>", S_small)],
            [Paragraph(desc, S_small)],
        ], colWidths=[W/3-4*mm])
        cell.setStyle(TableStyle([
            ("BOX",(0,0),(-1,-1),1.5,colors.HexColor(col)),
            ("BACKGROUND",(0,0),(-1,-1),colors.white),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),7),
        ]))
        cg_cells.append(cell)
    cg_tbl = Table([cg_cells], colWidths=[W/3]*3)
    cg_tbl.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2)]))
    story.append(cg_tbl)
    story.append(Spacer(1, 5*mm))

    # ── 8. Clinical interpretation ─────────────────────────
    story.append(Paragraph("CLINICAL INTERPRETATION", S_h2))
    direction = "above" if lvh_prob >= threshold else "below"
    if prediction == 1:
        interp = [
            ("<b>What the AI found:</b>",
             f"The AI detected LVH-consistent features in {high_act_count}/{num_frames} frames "
             f"({round(lvh_prob*100,1)}% probability). Activation was {spread.lower()}, peaking at Frame {peak_frame}."),
            ("<b>What this means:</b>",
             "LVH means the left ventricle wall appears thicker than normal, commonly caused by hypertension or valvular disease."),
            ("<b>Recommended next steps:</b>",
             "• Formal measurement of IVSd and LVPWd<br/>• LVMI calculation<br/>• BP evaluation<br/>• Cardiology referral"),
        ]
    else:
        interp = [
            ("<b>What the AI found:</b>",
             f"No dominant LVH patterns detected. LVH probability ({round(lvh_prob*100,1)}%) is {direction} the {round(threshold*100)}% threshold."),
            ("<b>What this means:</b>",
             "The left ventricle does not show typical LVH thickening. This is a reassuring result."),
            ("<b>Recommended next steps:</b>",
             "• Continue routine cardiac monitoring<br/>• Maintain BP within target range<br/>• Repeat echo if symptoms change"),
        ]
    interp_tbl = Table(
        [[Paragraph(h, S_bodyB), Paragraph(b, S_body)] for h, b in interp],
        colWidths=[W*0.22, W*0.78]
    )
    interp_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),RISK_BG),
        ("BOX",(0,0),(-1,-1),0.5,RISK_COL),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#e0e0e0")),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1),8),
    ]))
    story.append(interp_tbl)
    story.append(Spacer(1, 5*mm))

    # ── 9. Disclaimer ──────────────────────────────────────
    disc_tbl = Table([
        [Paragraph("<b>WARNING:  IMPORTANT NOTICE</b>",
                   style("dh2", fontSize=9, textColor=colors.HexColor("#ffcc02"), alignment=TA_CENTER))],
        [Paragraph(
            "This report is generated by an AI system and is intended as a clinical decision SUPPORT tool only. "
            "It does NOT replace professional medical judgement. All findings must be reviewed and confirmed by a "
            "qualified cardiologist before any clinical decision, treatment, or diagnosis is made.",
            S_disc
        )],
    ], colWidths=[W])
    disc_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#37474f")),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12),
    ]))
    story.append(disc_tbl)

    doc.build(story)
    buf.seek(0)
    return buf


@app.route("/update-threshold", methods=["POST"])
def update_threshold():
    global LVH_THRESHOLD
    try:
        data          = request.get_json()
        new_threshold = float(data.get("threshold", LVH_THRESHOLD))
        if not 0 <= new_threshold <= 1:
            return jsonify({"error": "Threshold must be between 0 and 1"}), 400
        LVH_THRESHOLD = new_threshold
        return jsonify({"success": True, "threshold": LVH_THRESHOLD})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
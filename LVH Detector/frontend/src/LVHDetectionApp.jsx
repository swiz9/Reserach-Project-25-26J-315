import React, { useState, useRef } from "react";
import {
  Upload, Activity, AlertCircle, CheckCircle,
  Loader2, Video, Brain, BarChart2, ChevronDown, ChevronUp,
  Info, FileDown
} from "lucide-react";

const API_URL = "http://localhost:5000";

// ── colour helpers ──────────────────────────────────────────
function riskColour(lvhProb) {
  if (lvhProb >= 0.65) return { text: "#c62828", bg: "#ffebee", border: "#ef9a9a" };
  if (lvhProb >= 0.40) return { text: "#e65100", bg: "#fff3e0", border: "#ffcc80" };
  return { text: "#2e7d32", bg: "#e8f5e9", border: "#a5d6a7" };
}

// ── probability bar ─────────────────────────────────────────
function ProbBar({ value, threshold }) {
  const pct      = Math.round(value * 100);
  const threshPct = Math.round(threshold * 100);
  const colour   = value >= threshold ? "#c62828" : "#2e7d32";

  return (
    <div style={{ position: "relative", marginTop: 8 }}>
      {/* gradient track */}
      <div style={{
        height: 18, borderRadius: 9,
        background: "linear-gradient(to right,#2e7d32,#f9a825,#c62828)",
        position: "relative", overflow: "visible"
      }}>
        {/* needle */}
        <div style={{
          position: "absolute", left: `${pct}%`, top: -6,
          transform: "translateX(-50%)",
          width: 4, height: 30, borderRadius: 2,
          background: colour, zIndex: 2
        }} />
        {/* threshold dashed line */}
        <div style={{
          position: "absolute", left: `${threshPct}%`, top: -8,
          transform: "translateX(-50%)",
          borderLeft: "2px dashed #333", height: 34, zIndex: 1
        }} />
      </div>
      <div style={{
        display: "flex", justifyContent: "space-between",
        fontSize: 11, color: "#666", marginTop: 4
      }}>
        <span>0%</span>
        <span style={{ color: "#333", fontWeight: 600 }}>
          Threshold {threshPct}%
        </span>
        <span>100%</span>
      </div>
    </div>
  );
}

// ── frame importance bar chart ───────────────────────────────
function FrameBarChart({ importance, top3 }) {
  const max = Math.max(...importance);
  return (
    <div>
      <div style={{
        display: "flex", alignItems: "flex-end",
        gap: 3, height: 80
      }}>
        {importance.map((v, i) => {
          const rank = top3.indexOf(i);
          const col  = rank === 0 ? "#c62828"
                     : rank === 1 ? "#ef5350"
                     : rank === 2 ? "#ef9a9a"
                     : "#b0bec5";
          return (
            <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center" }}>
              {rank >= 0 && rank <= 2 && (
                <span style={{ fontSize: 9, color: "#c62828" }}>
                  {"★".repeat(3 - rank)}
                </span>
              )}
              <div style={{
                width: "100%",
                height: `${(v / max) * 64}px`,
                background: col,
                borderRadius: "2px 2px 0 0",
                transition: "height 0.3s"
              }} />
            </div>
          );
        })}
      </div>
      {/* x-axis labels */}
      <div style={{ display: "flex", gap: 3, marginTop: 2 }}>
        {importance.map((_, i) => (
          <div key={i} style={{ flex: 1, textAlign: "center", fontSize: 8, color: "#888" }}>
            {i + 1}
          </div>
        ))}
      </div>
      <p style={{ fontSize: 10, color: "#666", marginTop: 4 }}>
        Frame number (1 – {importance.length})
      </p>
    </div>
  );
}

// ── heatmap strip ────────────────────────────────────────────
function HeatmapStrip({ frames, importance, top3 }) {
  const [selected, setSelected] = useState(top3[0]);
  const max = Math.max(...importance);

  return (
    <div>
      {/* top-3 large view */}
      <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
        {top3.map((fi, rank) => {
          const rankCol = rank === 0 ? "#c62828" : rank === 1 ? "#ef5350" : "#ef9a9a";
          const rankLbl = ["MOST RELEVANT", "2ND MOST RELEVANT", "3RD MOST RELEVANT"][rank];
          const pct     = Math.round((importance[fi] / max) * 100);
          return (
            <div key={fi} style={{
              flex: 1, border: `3px solid ${rankCol}`,
              borderRadius: 10, overflow: "hidden"
            }}>
              <img
                src={`data:image/png;base64,${frames[fi]}`}
                alt={`Frame ${fi + 1}`}
                style={{ width: "100%", display: "block" }}
              />
              <div style={{
                padding: "6px 8px", background: "#fff",
                fontSize: 11, fontWeight: 700, color: rankCol
              }}>
                {rankLbl} — Frame {fi + 1} ({pct}% of peak)
              </div>
            </div>
          );
        })}
      </div>

      {/* scrollable all-frames strip */}
      <p style={{ fontSize: 12, color: "#555", marginBottom: 6, fontWeight: 600 }}>
        All frames — click to inspect
      </p>
      <div style={{
        display: "flex", gap: 4, overflowX: "auto", paddingBottom: 6
      }}>
        {frames.map((b64, i) => {
          const isTop  = top3.includes(i);
          const isSel  = selected === i;
          return (
            <div
              key={i}
              onClick={() => setSelected(i)}
              style={{
                minWidth: 52, cursor: "pointer",
                border: `2px solid ${isSel ? "#1a237e" : isTop ? "#c62828" : "#ddd"}`,
                borderRadius: 6, overflow: "hidden",
                opacity: isSel ? 1 : 0.75,
                transition: "all 0.15s",
                flexShrink: 0
              }}
            >
              <img
                src={`data:image/png;base64,${b64}`}
                alt={`F${i + 1}`}
                style={{ width: 52, display: "block" }}
              />
              <div style={{
                fontSize: 9, textAlign: "center", padding: "2px 0",
                background: isSel ? "#1a237e" : "#f5f5f5",
                color: isSel ? "white" : "#555"
              }}>
                F{i + 1}
              </div>
            </div>
          );
        })}
      </div>

      {/* selected frame enlarged */}
      {selected !== null && (
        <div style={{ marginTop: 12, textAlign: "center" }}>
          <p style={{ fontSize: 12, color: "#555", marginBottom: 6 }}>
            Frame {selected + 1} — activation intensity:{" "}
            <strong>{Math.round((importance[selected] / max) * 100)}% of peak</strong>
          </p>
          <img
            src={`data:image/png;base64,${frames[selected]}`}
            alt={`Frame ${selected + 1}`}
            style={{
              maxWidth: 300, borderRadius: 10,
              border: "2px solid #ccc"
            }}
          />
        </div>
      )}
    </div>
  );
}

// ── colour legend ────────────────────────────────────────────
function HeatmapLegend() {
  const items = [
    { col: "#c62828", label: "RED / ORANGE", desc: "Strong AI focus — heart wall features linked to LVH." },
    { col: "#f9a825", label: "YELLOW",       desc: "Moderate AI attention — relevant features detected." },
    { col: "#1565c0", label: "BLUE / DARK",  desc: "Low attention — less relevant for LVH diagnosis." },
  ];
  return (
    <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
      {items.map(({ col, label, desc }) => (
        <div key={label} style={{
          flex: 1, border: `2px solid ${col}`, borderRadius: 8,
          padding: "8px 10px", background: "#fff"
        }}>
          <p style={{ color: col, fontWeight: 700, fontSize: 11, margin: "0 0 4px" }}>
            ■ {label}
          </p>
          <p style={{ fontSize: 11, color: "#444", margin: 0 }}>{desc}</p>
        </div>
      ))}
    </div>
  );
}

// ── clinical interpretation ──────────────────────────────────
function ClinicalInterpretation({ data }) {
  const { prediction, lvh_probability: prob, threshold,
          high_act_count, num_frames, peak_frame, spread } = data;
  const direction = prob >= threshold ? "above" : "below";

  return (
    <div style={{
      background: prediction === 1 ? "#ffebee" : "#e8f5e9",
      border: `1px solid ${prediction === 1 ? "#ef9a9a" : "#a5d6a7"}`,
      borderRadius: 10, padding: 16
    }}>
      <p style={{
        fontWeight: 700, fontSize: 13, marginBottom: 10,
        color: prediction === 1 ? "#b71c1c" : "#1b5e20"
      }}>
        🩺 Clinical Interpretation
      </p>

      {prediction === 1 ? (
        <>
          <p style={{ fontSize: 12, color: "#333", marginBottom: 8 }}>
            <strong>What the AI found:</strong> The model detected features consistent with
            Left Ventricular Hypertrophy in {high_act_count} of {num_frames} frames
            ({Math.round(prob * 100)}% probability). Activation was {spread.toLowerCase()},
            peaking at Frame {peak_frame}.
          </p>
          <p style={{ fontSize: 12, color: "#333", marginBottom: 8 }}>
            <strong>What this means:</strong> LVH means the left ventricle wall appears
            thicker than normal, commonly caused by hypertension or valvular disease.
          </p>
          <p style={{ fontSize: 12, color: "#333" }}>
            <strong>Recommended next steps:</strong><br />
            • Formal measurement of IVSd and LVPWd<br />
            • Left ventricular mass index (LVMI) calculation<br />
            • Blood pressure evaluation and management review<br />
            • Cardiology specialist referral if not already under care
          </p>
        </>
      ) : (
        <>
          <p style={{ fontSize: 12, color: "#333", marginBottom: 8 }}>
            <strong>What the AI found:</strong> No dominant LVH patterns detected. LVH
            probability ({Math.round(prob * 100)}%) is {direction} the{" "}
            {Math.round(threshold * 100)}% threshold. Cardiac wall regions appear within
            normal range.
          </p>
          <p style={{ fontSize: 12, color: "#333", marginBottom: 8 }}>
            <strong>What this means:</strong> The left ventricle does not show the typical
            thickening pattern associated with LVH. This is a reassuring result.
          </p>
          <p style={{ fontSize: 12, color: "#333" }}>
            <strong>Recommended next steps:</strong><br />
            • Continue routine cardiac monitoring as indicated<br />
            • Maintain blood pressure within target range<br />
            • Repeat echocardiogram if symptoms change
          </p>
        </>
      )}
    </div>
  );
}

// ── scan stats card ──────────────────────────────────────────
function ScanStats({ data }) {
  const stats = [
    ["📹 Frames Analysed",     `${data.num_frames}`],
    ["🎯 Peak Activity Frame",  `Frame ${data.peak_frame}`],
    ["📊 High-Activity Frames", `${data.high_act_count} / ${data.num_frames}`],
    ["🔬 Activation Pattern",   data.spread],
    ["📐 Decision Threshold",   `${Math.round(data.threshold * 100)}%`],
    ["🤖 XAI Method",           "Grad-CAM++"],
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
      {stats.map(([label, val]) => (
        <div key={label} style={{
          background: "#f5f5f5", borderRadius: 8,
          padding: "8px 12px", border: "1px solid #e0e0e0"
        }}>
          <p style={{ fontSize: 10, color: "#666", margin: 0 }}>{label}</p>
          <p style={{ fontSize: 13, fontWeight: 700, color: "#1a237e", margin: "2px 0 0" }}>
            {val}
          </p>
        </div>
      ))}
    </div>
  );
}

// ════════════════════════════════════════════════════════════
// MAIN APP
// ════════════════════════════════════════════════════════════
export default function LVHDetectionApp() {
  const [file,          setFile]          = useState(null);
  const [videoPreview,  setVideoPreview]  = useState(null);
  const [loading,       setLoading]       = useState(false);
  const [xaiLoading,    setXaiLoading]    = useState(false);
  const [pdfLoading,    setPdfLoading]    = useState(false);
  const [saveLoading,   setSaveLoading]   = useState(false);
  const [result,        setResult]        = useState(null);
  const [xaiResult,     setXaiResult]     = useState(null);
  const [error,         setError]         = useState(null);
  const [dragActive,    setDragActive]    = useState(false);
  const [xaiOpen,       setXaiOpen]       = useState(false);
  const [savedRecord,   setSavedRecord]   = useState(null);
  // Doctor fields
  const [patientId,     setPatientId]     = useState("");
  const [doctorName,    setDoctorName]    = useState("");
  const [doctorComment, setDoctorComment] = useState("");
  const fileInputRef = useRef(null);

  // ── drag & drop ────────────────────────────────────────────
  const handleDrag = (e) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(e.type === "dragenter" || e.type === "dragover");
  };
  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) handleFileSelect(e.dataTransfer.files[0]);
  };
  const handleFileSelect = (f) => {
    if (!f) return;
    if (!f.name.match(/\.(mp4|avi|mov|mkv)$/i)) {
      setError("Please upload a valid video file (MP4, AVI, MOV, or MKV)");
      return;
    }
    setFile(f);
    setError(null); setResult(null); setXaiResult(null); setXaiOpen(false);
    setVideoPreview(URL.createObjectURL(f));
  };

  // ── predict ────────────────────────────────────────────────
  const handlePredict = async () => {
    if (!file) { setError("Please select a video file first"); return; }
    setLoading(true); setError(null); setResult(null);
    const fd = new FormData();
    fd.append("video", file);
    try {
      const res  = await fetch(`${API_URL}/predict`, { method: "POST", body: fd });
      const data = await res.json();
      if (data.success) setResult(data);
      else setError(data.error || "Prediction failed");
    } catch {
      setError("Failed to connect to server. Make sure the Flask backend is running.");
    } finally { setLoading(false); }
  };

  // ── explain ────────────────────────────────────────────────
  const handleExplain = async () => {
    if (!file) return;
    setXaiLoading(true); setError(null); setXaiOpen(false);
    const fd = new FormData();
    fd.append("video", file);
    try {
      const res  = await fetch(`${API_URL}/explain`, { method: "POST", body: fd });
      const data = await res.json();
      if (data.success) { setXaiResult(data); setXaiOpen(true); }
      else setError(data.error || "Explainability analysis failed");
    } catch {
      setError("Failed to connect to server for XAI analysis.");
    } finally { setXaiLoading(false); }
  };

  // ── export PDF only ──────────────────────────────────────
  const handleExportPDF = async () => {
    if (!xaiResult) return;
    setPdfLoading(true); setError(null);
    try {
      const payload = {
        ...xaiResult,
        patient_id:     patientId || "—",
        doctor_name:    doctorName || "",
        doctor_comment: doctorComment || "",
        filename:       file?.name || "echocardiogram",
      };
      const res = await fetch(`${API_URL}/report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json();
        setError(err.error || "PDF generation failed");
        return;
      }
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href     = url;
      a.download = `lvh_report_${patientId || "patient"}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setError("Failed to generate PDF. Make sure the Flask backend is running.");
    } finally { setPdfLoading(false); }
  };

  // ── save report to DB + download PDF ─────────────────────
  const handleSaveReport = async () => {
    if (!xaiResult) return;
    if (!patientId.trim()) { setError("Please enter a Patient ID before saving."); return; }
    setSaveLoading(true); setError(null); setSavedRecord(null);
    try {
      const payload = {
        ...xaiResult,
        patient_id:     patientId.trim(),
        doctor_name:    doctorName.trim(),
        doctor_comment: doctorComment.trim(),
        filename:       file?.name || "echocardiogram",
      };
      const res = await fetch(`${API_URL}/save-report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.success) {
        setSavedRecord({ id: data.record_id, message: data.message });
        const pdfBytes = Uint8Array.from(atob(data.pdf_base64), c => c.charCodeAt(0));
        const blob = new Blob([pdfBytes], { type: "application/pdf" });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement("a");
        a.href     = url;
        a.download = `lvh_report_${patientId}_ID${data.record_id}.pdf`;
        a.click();
        URL.revokeObjectURL(url);
      } else {
        setError(data.error || "Failed to save report");
      }
    } catch {
      setError("Failed to connect to server. Make sure the Flask backend is running.");
    } finally { setSaveLoading(false); }
  };

  const reset = () => {
    setFile(null); setResult(null); setXaiResult(null);
    setError(null); setVideoPreview(null); setXaiOpen(false);
    setSavedRecord(null); setPatientId(""); setDoctorName(""); setDoctorComment("");
  };

  // ── risk colours for result card ───────────────────────────
  const rc = result ? riskColour(result.lvh_probability) : null;

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg,#e8eaf6 0%,#e3f2fd 100%)",
      padding: "32px 16px", fontFamily: "system-ui,sans-serif"
    }}>
      <div style={{ maxWidth: 860, margin: "0 auto" }}>

        {/* ── HEADER ── */}
        <div style={{ textAlign: "center", marginBottom: 28 }}>
          <div style={{
            display: "inline-flex", alignItems: "center", justifyContent: "center",
            width: 64, height: 64, borderRadius: "50%",
            background: "#3949ab", marginBottom: 12
          }}>
            <Activity color="white" size={32} />
          </div>
          <h1 style={{ fontSize: 32, fontWeight: 800, color: "#1a237e", margin: "0 0 6px" }}>
            LVH Detection System
          </h1>
          <p style={{ color: "#546e7a", fontSize: 15, margin: 0 }}>
            Upload an echocardiogram video to detect Left Ventricular Hypertrophy
          </p>
        </div>

        {/* ── MAIN CARD ── */}
        <div style={{
          background: "#fff", borderRadius: 20,
          boxShadow: "0 8px 32px rgba(0,0,0,0.10)", padding: 32
        }}>

          {/* ── upload zone ── */}
          {!file && (
            <div
              onDragEnter={handleDrag} onDragLeave={handleDrag}
              onDragOver={handleDrag} onDrop={handleDrop}
              style={{
                border: `2px dashed ${dragActive ? "#3949ab" : "#cfd8dc"}`,
                borderRadius: 14, padding: "48px 24px", textAlign: "center",
                background: dragActive ? "#e8eaf6" : "#fafafa",
                transition: "all 0.2s", cursor: "pointer"
              }}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={48} color="#90a4ae" style={{ marginBottom: 12 }} />
              <h3 style={{ fontSize: 18, color: "#455a64", margin: "0 0 6px" }}>
                Upload Echocardiogram Video
              </h3>
              <p style={{ color: "#90a4ae", marginBottom: 16 }}>
                Drag & drop or click to browse
              </p>
              <span style={{
                background: "#3949ab", color: "white",
                padding: "10px 24px", borderRadius: 8, fontSize: 14, fontWeight: 600
              }}>
                Select Video
              </span>
              <p style={{ fontSize: 12, color: "#b0bec5", marginTop: 14 }}>
                Supported: MP4 · AVI · MOV · MKV
              </p>
              <input ref={fileInputRef} type="file"
                accept="video/mp4,video/avi,video/mov,video/mkv"
                style={{ display: "none" }}
                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              />
            </div>
          )}

          {/* ── video preview ── */}
          {file && videoPreview && (
            <div style={{ marginBottom: 20 }}>
              <div style={{
                display: "flex", alignItems: "center",
                justifyContent: "space-between", marginBottom: 10
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <Video size={20} color="#3949ab" />
                  <div>
                    <p style={{ fontWeight: 600, color: "#1a237e", margin: 0 }}>{file.name}</p>
                    <p style={{ fontSize: 12, color: "#90a4ae", margin: 0 }}>
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button onClick={reset} style={{
                  background: "none", border: "none",
                  color: "#ef5350", cursor: "pointer", fontSize: 13, fontWeight: 600
                }}>
                  Remove
                </button>
              </div>
              <video src={videoPreview} controls
                style={{ width: "100%", borderRadius: 10, border: "1px solid #e0e0e0" }}
              />
            </div>
          )}

          {/* ── action buttons ── */}
          {file && !result && (
            <button onClick={handlePredict} disabled={loading} style={{
              width: "100%", padding: "14px 0", borderRadius: 10,
              border: "none", cursor: loading ? "not-allowed" : "pointer",
              background: loading ? "#b0bec5" : "#3949ab",
              color: "white", fontSize: 16, fontWeight: 700,
              display: "flex", alignItems: "center", justifyContent: "center", gap: 10
            }}>
              {loading
                ? <><Loader2 size={20} className="spin" style={{ animation: "spin 1s linear infinite" }} /> Analysing…</>
                : <><Activity size={20} /> Analyse for LVH</>
              }
            </button>
          )}

          {/* ── error ── */}
          {error && (
            <div style={{
              marginTop: 16, padding: 14,
              background: "#ffebee", border: "1px solid #ef9a9a",
              borderRadius: 10, display: "flex", gap: 10, alignItems: "flex-start"
            }}>
              <AlertCircle size={18} color="#c62828" style={{ flexShrink: 0, marginTop: 2 }} />
              <div>
                <p style={{ fontWeight: 700, color: "#b71c1c", margin: "0 0 2px" }}>Error</p>
                <p style={{ fontSize: 13, color: "#c62828", margin: 0 }}>{error}</p>
              </div>
            </div>
          )}

          {/* ── RESULT CARD ── */}
          {result && rc && (
            <div style={{ marginTop: 20 }}>
              <div style={{
                border: `2px solid ${rc.border}`, borderRadius: 14,
                background: rc.bg, padding: 20
              }}>
                {/* diagnosis headline */}
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
                  {result.prediction === 1
                    ? <AlertCircle size={36} color={rc.text} />
                    : <CheckCircle size={36} color={rc.text} />
                  }
                  <div>
                    <h2 style={{ fontSize: 22, fontWeight: 800, color: rc.text, margin: 0 }}>
                      {result.prediction_label}
                    </h2>
                    <p style={{ fontSize: 13, color: "#555", margin: 0 }}>
                      {result.prediction === 1
                        ? "Left Ventricular Hypertrophy detected"
                        : "No signs of LVH detected"}
                    </p>
                  </div>
                </div>

                {/* probability gauge */}
                <div style={{ background: "white", borderRadius: 10, padding: "12px 16px", marginBottom: 12 }}>
                  <p style={{ fontSize: 12, color: "#666", margin: "0 0 4px" }}>LVH Probability</p>
                  <p style={{ fontSize: 28, fontWeight: 800, color: rc.text, margin: 0 }}>
                    {Math.round(result.lvh_probability * 100)}%
                  </p>
                  <ProbBar value={result.lvh_probability} threshold={result.threshold} />
                </div>

                {/* metrics row */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  {[
                    ["AI Confidence", `${result.confidence}%`],
                    ["Threshold", `${Math.round(result.threshold * 100)}%`],
                  ].map(([label, val]) => (
                    <div key={label} style={{
                      background: "white", borderRadius: 10, padding: "10px 14px"
                    }}>
                      <p style={{ fontSize: 11, color: "#666", margin: 0 }}>{label}</p>
                      <p style={{ fontSize: 20, fontWeight: 700, color: "#1a237e", margin: 0 }}>
                        {val}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* ── XAI BUTTON ── */}
              <button
                onClick={handleExplain}
                disabled={xaiLoading}
                style={{
                  width: "100%", marginTop: 14, padding: "13px 0",
                  borderRadius: 10, border: "2px solid #7b1fa2",
                  cursor: xaiLoading ? "not-allowed" : "pointer",
                  background: xaiLoading ? "#f3e5f5" : "#f3e5f5",
                  color: "#6a1b9a", fontSize: 15, fontWeight: 700,
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 10
                }}
              >
                {xaiLoading
                  ? <><Loader2 size={18} style={{ animation: "spin 1s linear infinite" }} /> Running Grad-CAM++ Analysis…</>
                  : <><Brain size={18} /> Explain AI Decision (Grad-CAM++)</>
                }
              </button>

              {/* reset */}
              <button onClick={reset} style={{
                width: "100%", marginTop: 10, padding: "12px 0",
                borderRadius: 10, border: "none", cursor: "pointer",
                background: "#eceff1", color: "#546e7a",
                fontSize: 14, fontWeight: 600
              }}>
                Analyse Another Video
              </button>
            </div>
          )}
        </div>

        {/* ════════════════════════════════════════════════════
            XAI PANEL
        ════════════════════════════════════════════════════ */}
        {xaiResult && (
          <div style={{
            marginTop: 24, background: "#fff",
            borderRadius: 20, boxShadow: "0 8px 32px rgba(0,0,0,0.10)",
            overflow: "hidden"
          }}>
            {/* XAI header */}
            <div
              onClick={() => setXaiOpen(o => !o)}
              style={{
                background: "linear-gradient(135deg,#4a148c,#7b1fa2)",
                padding: "16px 24px", cursor: "pointer",
                display: "flex", alignItems: "center", justifyContent: "space-between"
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <Brain color="white" size={24} />
                <div>
                  <p style={{ color: "white", fontWeight: 700, fontSize: 16, margin: 0 }}>
                    🫀 Cardiac AI Screening Report — Grad-CAM++ XAI
                  </p>
                  <p style={{ color: "#ce93d8", fontSize: 12, margin: 0 }}>
                    Gradient-weighted Class Activation Mapping • X3D Echocardiogram Analysis
                  </p>
                </div>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <button
                  onClick={(e) => { e.stopPropagation(); handleExportPDF(); }}
                  disabled={pdfLoading}
                  style={{
                    display: "flex", alignItems: "center", gap: 6,
                    background: pdfLoading ? "#ce93d8" : "white",
                    color: "#4a148c", border: "none", borderRadius: 8,
                    padding: "7px 14px", fontWeight: 700, fontSize: 13,
                    cursor: pdfLoading ? "not-allowed" : "pointer",
                    whiteSpace: "nowrap"
                  }}
                >
                  {pdfLoading
                    ? <><Loader2 size={15} style={{ animation: "spin 1s linear infinite" }} /> Generating…</>
                    : <><FileDown size={15} /> Export PDF</>
                  }
                </button>
                {xaiOpen ? <ChevronUp color="white" size={20} /> : <ChevronDown color="white" size={20} />}
              </div>
            </div>

            {xaiOpen && (
              <div style={{ padding: 24 }}>

                {/* ── diagnosis + risk badge ── */}
                {(() => {
                  const xrc = riskColour(xaiResult.lvh_probability);
                  return (
                    <div style={{
                      border: `2px solid ${xrc.border}`, borderRadius: 12,
                      background: xrc.bg, padding: 16, marginBottom: 20,
                      display: "flex", alignItems: "center", gap: 16
                    }}>
                      <div style={{
                        fontSize: 40, width: 56, textAlign: "center"
                      }}>
                        {xaiResult.prediction === 1 ? "⚠️" : "✅"}
                      </div>
                      <div>
                        <h3 style={{ color: xrc.text, fontWeight: 800, fontSize: 20, margin: 0 }}>
                          {xaiResult.prediction_label}
                        </h3>
                        <p style={{ color: xrc.text, margin: "2px 0 0", fontSize: 13 }}>
                          Risk Level: <strong>{xaiResult.risk_level}</strong> &nbsp;|&nbsp;
                          LVH Probability: <strong>{Math.round(xaiResult.lvh_probability * 100)}%</strong> &nbsp;|&nbsp;
                          Confidence: <strong>{xaiResult.confidence}%</strong>
                        </p>
                      </div>
                    </div>
                  );
                })()}

                {/* ── scan stats ── */}
                <h4 style={{ color: "#1a237e", fontWeight: 700, marginBottom: 10 }}>
                  📋 Scan Summary
                </h4>
                <ScanStats data={xaiResult} />

                {/* ── temporal bar chart ── */}
                <h4 style={{ color: "#1a237e", fontWeight: 700, margin: "20px 0 6px" }}>
                  📊 Frame-by-Frame Cardiac Activity
                </h4>
                <p style={{ fontSize: 12, color: "#666", marginBottom: 8 }}>
                  Which moments of the heartbeat were most important to the AI
                </p>
                <div style={{
                  background: "#fafafa", borderRadius: 10,
                  padding: "14px 16px", border: "1px solid #e0e0e0"
                }}>
                  <FrameBarChart
                    importance={xaiResult.frame_importance}
                    top3={xaiResult.top3_frames}
                  />
                </div>

                {/* ── heatmap frames ── */}
                <h4 style={{ color: "#1a237e", fontWeight: 700, margin: "20px 0 8px" }}>
                  🔬 Grad-CAM++ Heatmap Overlays
                </h4>
                <HeatmapStrip
                  frames={xaiResult.heatmap_frames}
                  importance={xaiResult.frame_importance}
                  top3={xaiResult.top3_frames}
                />

                {/* ── colour legend ── */}
                <h4 style={{ color: "#1a237e", fontWeight: 700, margin: "20px 0 4px" }}>
                  🎨 How to Read the Heatmap Colours
                </h4>
                <HeatmapLegend />

                {/* ── clinical interpretation ── */}
                <div style={{ marginTop: 20 }}>
                  <ClinicalInterpretation data={xaiResult} />
                </div>

                {/* ── disclaimer ── */}
                <div style={{
                  marginTop: 20, background: "#37474f",
                  borderRadius: 10, padding: "14px 18px"
                }}>
                  <p style={{ color: "#ffcc02", fontWeight: 700, fontSize: 13, margin: "0 0 6px" }}>
                    ⚠️ Important Notice
                  </p>
                  <p style={{ color: "#eceff1", fontSize: 12, margin: 0, lineHeight: 1.7 }}>
                    This report is generated by an AI system and is intended as a clinical
                    decision <strong>support</strong> tool only. It does <strong>not</strong> replace
                    professional medical judgement. All findings must be reviewed and confirmed
                    by a qualified cardiologist before any clinical decision, treatment, or
                    diagnosis is made.
                  </p>
                </div>

                {/* ── Doctor Comment & Submit ── */}
                <div style={{
                  marginTop: 24, background: "#f3f4ff",
                  border: "2px solid #7986cb", borderRadius: 14, padding: 20
                }}>
                  <h4 style={{ color: "#1a237e", fontWeight: 700, fontSize: 15, margin: "0 0 16px", display: "flex", alignItems: "center", gap: 8 }}>
                    🩺 Doctor Review &amp; Save to Database
                  </h4>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 14 }}>
                    <div>
                      <label style={{ fontSize: 12, fontWeight: 700, color: "#3949ab", display: "block", marginBottom: 4 }}>
                        Patient ID <span style={{ color: "#c62828" }}>*</span>
                      </label>
                      <input
                        type="text"
                        value={patientId}
                        onChange={e => setPatientId(e.target.value)}
                        placeholder="e.g. PT-2024-001"
                        style={{
                          width: "100%", boxSizing: "border-box",
                          padding: "9px 12px", borderRadius: 8,
                          border: "1.5px solid #9fa8da", fontSize: 13,
                          outline: "none", fontFamily: "inherit",
                          background: "white"
                        }}
                      />
                    </div>
                    <div>
                      <label style={{ fontSize: 12, fontWeight: 700, color: "#3949ab", display: "block", marginBottom: 4 }}>
                        Doctor Name
                      </label>
                      <input
                        type="text"
                        value={doctorName}
                        onChange={e => setDoctorName(e.target.value)}
                        placeholder="e.g. Dr. Smith"
                        style={{
                          width: "100%", boxSizing: "border-box",
                          padding: "9px 12px", borderRadius: 8,
                          border: "1.5px solid #9fa8da", fontSize: 13,
                          outline: "none", fontFamily: "inherit",
                          background: "white"
                        }}
                      />
                    </div>
                  </div>

                  <div style={{ marginBottom: 16 }}>
                    <label style={{ fontSize: 12, fontWeight: 700, color: "#3949ab", display: "block", marginBottom: 4 }}>
                      Doctor's Comment
                    </label>
                    <textarea
                      value={doctorComment}
                      onChange={e => setDoctorComment(e.target.value)}
                      placeholder="Enter clinical notes, observations, or recommendations for this patient..."
                      rows={4}
                      style={{
                        width: "100%", boxSizing: "border-box",
                        padding: "9px 12px", borderRadius: 8,
                        border: "1.5px solid #9fa8da", fontSize: 13,
                        outline: "none", fontFamily: "inherit",
                        resize: "vertical", background: "white",
                        lineHeight: 1.6
                      }}
                    />
                  </div>

                  {/* Save + PDF buttons */}
                  <div style={{ display: "flex", gap: 12 }}>
                    <button
                      onClick={handleSaveReport}
                      disabled={saveLoading}
                      style={{
                        flex: 2, padding: "12px 0", borderRadius: 10, border: "none",
                        background: saveLoading ? "#b0bec5" : "#1a237e",
                        color: "white", fontSize: 14, fontWeight: 700,
                        cursor: saveLoading ? "not-allowed" : "pointer",
                        display: "flex", alignItems: "center", justifyContent: "center", gap: 8
                      }}
                    >
                      {saveLoading
                        ? <><span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>⏳</span> Saving &amp; Generating PDF…</>
                        : <>💾 Submit &amp; Save Report to DB</>
                      }
                    </button>
                    <button
                      onClick={handleExportPDF}
                      disabled={pdfLoading}
                      style={{
                        flex: 1, padding: "12px 0", borderRadius: 10,
                        border: "1.5px solid #7986cb",
                        background: pdfLoading ? "#e8eaf6" : "white",
                        color: "#3949ab", fontSize: 14, fontWeight: 700,
                        cursor: pdfLoading ? "not-allowed" : "pointer",
                        display: "flex", alignItems: "center", justifyContent: "center", gap: 8
                      }}
                    >
                      {pdfLoading ? "Generating…" : "📄 Preview PDF"}
                    </button>
                  </div>

                  {/* Success banner */}
                  {savedRecord && (
                    <div style={{
                      marginTop: 14, padding: "12px 16px",
                      background: "#e8f5e9", border: "1.5px solid #a5d6a7",
                      borderRadius: 10, display: "flex", alignItems: "center", gap: 10
                    }}>
                      <span style={{ fontSize: 22 }}>✅</span>
                      <div>
                        <p style={{ fontWeight: 700, color: "#1b5e20", margin: 0, fontSize: 13 }}>
                          Report Saved Successfully!
                        </p>
                        <p style={{ color: "#2e7d32", margin: 0, fontSize: 12 }}>
                          {savedRecord.message} — PDF downloaded automatically.
                        </p>
                      </div>
                    </div>
                  )}
                </div>

              </div>
            )}
          </div>
        )}

        {/* ── footer ── */}
        <p style={{ textAlign: "center", fontSize: 12, color: "#90a4ae", marginTop: 24 }}>
          This system uses an X3D deep learning model trained on echocardiogram videos.<br />
          <strong>For research purposes only. Not for clinical diagnosis.</strong>
        </p>
      </div>

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
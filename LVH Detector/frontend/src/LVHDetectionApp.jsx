import React, { useState } from "react";
import {
  Upload,
  Activity,
  AlertCircle,
  CheckCircle,
  Loader2,
  Video,
} from "lucide-react";

export default function LVHDetectionApp() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [videoPreview, setVideoPreview] = useState(null);

  const API_URL = "http://localhost:5000"; // Update this if deployed

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (selectedFile) => {
    if (!selectedFile) return;

    const validTypes = ["video/mp4", "video/avi", "video/mov", "video/mkv"];
    if (
      !validTypes.includes(selectedFile.type) &&
      !selectedFile.name.match(/\.(mp4|avi|mov|mkv)$/i)
    ) {
      setError("Please upload a valid video file (MP4, AVI, MOV, or MKV)");
      return;
    }

    setFile(selectedFile);
    setError(null);
    setResult(null);

    // Create video preview
    const url = URL.createObjectURL(selectedFile);
    setVideoPreview(url);
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleSubmit = async () => {
    if (!file) {
      setError("Please select a video file first");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("video", file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResult(data);
      } else {
        setError(data.error || "Prediction failed");
      }
    } catch (err) {
      setError(
        "Failed to connect to server. Make sure the Flask backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setVideoPreview(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Activity className="w-12 h-12 text-indigo-600" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            LVH Detection System
          </h1>
          <p className="text-gray-600">
            Upload an echocardiogram video to detect Left Ventricular
            Hypertrophy
          </p>
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          {/* Upload Area */}
          {!file && (
            <div
              className={`border-2 border-dashed rounded-xl p-12 text-center transition-all ${
                dragActive
                  ? "border-indigo-500 bg-indigo-50"
                  : "border-gray-300 hover:border-indigo-400"
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <Upload className="w-16 h-16 mx-auto mb-4 text-gray-400" />
              <h3 className="text-xl font-semibold text-gray-700 mb-2">
                Upload Video File
              </h3>
              <p className="text-gray-500 mb-6">
                Drag and drop or click to browse
              </p>
              <input
                type="file"
                accept="video/mp4,video/avi,video/mov,video/mkv"
                onChange={handleFileInput}
                className="hidden"
                id="file-input"
              />
              <label
                htmlFor="file-input"
                className="inline-block bg-indigo-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-indigo-700 cursor-pointer transition-colors"
              >
                Select Video
              </label>
              <p className="text-sm text-gray-400 mt-4">
                Supported formats: MP4, AVI, MOV, MKV
              </p>
            </div>
          )}

          {/* Video Preview */}
          {file && videoPreview && (
            <div className="mb-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <Video className="w-6 h-6 text-indigo-600" />
                  <div>
                    <p className="font-medium text-gray-900">{file.name}</p>
                    <p className="text-sm text-gray-500">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button
                  onClick={resetForm}
                  className="text-gray-500 hover:text-gray-700 text-sm font-medium"
                >
                  Remove
                </button>
              </div>
              <video
                src={videoPreview}
                controls
                className="w-full rounded-lg border border-gray-200"
              />
            </div>
          )}

          {/* Analyze Button */}
          {file && !result && (
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="w-full bg-indigo-600 text-white py-4 rounded-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-3"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Analyzing Video...
                </>
              ) : (
                <>
                  <Activity className="w-5 h-5" />
                  Analyze for LVH
                </>
              )}
            </button>
          )}

          {/* Error Message */}
          {error && (
            <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-medium text-red-900">Error</p>
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="mt-6 space-y-4">
              <div
                className={`p-6 rounded-xl border-2 ${
                  result.prediction === 1
                    ? "bg-red-50 border-red-200"
                    : "bg-green-50 border-green-200"
                }`}
              >
                <div className="flex items-center gap-3 mb-4">
                  {result.prediction === 1 ? (
                    <AlertCircle className="w-8 h-8 text-red-600" />
                  ) : (
                    <CheckCircle className="w-8 h-8 text-green-600" />
                  )}
                  <div>
                    <h3 className="text-2xl font-bold text-gray-900">
                      {result.prediction_label}
                    </h3>
                    <p className="text-sm text-gray-600">
                      {result.prediction === 1
                        ? "Left Ventricular Hypertrophy detected"
                        : "No signs of LVH detected"}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 mt-4">
                  <div className="bg-white p-4 rounded-lg">
                    <p className="text-sm text-gray-600 mb-1">
                      LVH Probability
                    </p>
                    <p className="text-2xl font-bold text-gray-900">
                      {(result.lvh_probability * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div className="bg-white p-4 rounded-lg">
                    <p className="text-sm text-gray-600 mb-1">Confidence</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {result.confidence}%
                    </p>
                  </div>
                </div>
              </div>

              <button
                onClick={resetForm}
                className="w-full bg-gray-100 text-gray-700 py-3 rounded-lg font-medium hover:bg-gray-200 transition-colors"
              >
                Analyze Another Video
              </button>
            </div>
          )}
        </div>

        {/* Info Footer */}
        <div className="mt-8 text-center text-sm text-gray-600">
          <p>
            This system uses an X3D deep learning model trained on
            echocardiogram videos.
          </p>
          <p className="mt-2">
            <strong>Note:</strong> This tool is for research purposes only and
            should not be used for clinical diagnosis.
          </p>
        </div>
      </div>
    </div>
  );
}

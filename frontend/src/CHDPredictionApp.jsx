import React, { useState } from "react";
import {
  Heart,
  Activity,
  AlertCircle,
  CheckCircle,
  TrendingUp,
  Info,
} from "lucide-react";

const FactorBar = ({ icon, label, value, isPositive }) => {
  const absValue = Math.abs(value);
  const percentage = Math.min(absValue * 100, 100);

  return (
    <div className="space-y-2">
      <div className="flex items-start gap-2">
        <div className={isPositive ? "text-green-600" : "text-red-600"}>
          {icon}
        </div>
        <div className="flex-1">
          <div className="text-sm text-gray-700">{label}</div>
          <div className="relative h-2 mt-2 overflow-hidden bg-gray-200 rounded-full">
            <div
              className={`absolute top-0 h-full transition-all ${
                isPositive ? "bg-green-500" : "bg-red-500"
              }`}
              style={{ width: `${percentage}%` }}
            />
          </div>
        </div>
        <div
          className={`text-sm font-semibold ${
            isPositive ? "text-green-600" : "text-red-600"
          }`}
        >
          {value > 0 ? "+" : ""}
          {value.toFixed(2)}
        </div>
      </div>
    </div>
  );
};

export default function CHDPredictionApp() {
  const [formData, setFormData] = useState({
    gender: 0,
    age: 50,
    currentSmoker: 0,
    cigsPerDay: 0,
    BPMeds: 0,
    prevalentStroke: 0,
    prevalentHyp: 0,
    diabetes: 0,
    totChol: 200,
    sysBP: 120,
    diaBP: 80,
    BMI: 25,
    heartRate: 70,
    glucose: 85,
  });

  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);

  const API_URL = "http://localhost:5000/api";

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: parseFloat(value) || 0 }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setShowExplanation(false);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      alert("Error: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleExplain = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      setExplanation(data);
      setShowExplanation(true);
    } catch (error) {
      alert("Error: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  if (showExplanation && explanation) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
        <div className="container px-4 py-8 mx-auto max-w-7xl">
          <div className="mb-8 text-center">
            <div className="flex items-center justify-center gap-3 mb-3">
              <Activity className="w-10 h-10 text-indigo-600" />
              <h1 className="text-4xl font-bold text-gray-800">
                CHD Risk Prediction System
              </h1>
            </div>
            <p className="text-lg text-gray-600">
              AI-powered clinical decision support system with explainable
              predictions
            </p>
          </div>
          <div className="p-6 bg-white shadow-lg rounded-xl">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-gray-800">
                Model Explanation (SHAP)
              </h2>
              <button
                onClick={() => setShowExplanation(false)}
                className="px-4 py-2 text-sm font-medium text-gray-700 transition bg-gray-100 rounded-lg hover:bg-gray-200"
              >
                Back to Results
              </button>
            </div>
            <div className="mb-6">
              <img
                src={`data:image/png;base64,${explanation.shap_plot}`}
                alt="SHAP Feature Importance"
                className="w-full rounded-lg shadow-md"
              />
            </div>
            <div className="p-4 rounded-lg bg-gray-50">
              <h3 className="mb-3 font-semibold">Feature Impact Values</h3>
              <div className="space-y-2">
                {Object.entries(explanation.feature_importance).map(
                  ([feature, value]) => (
                    <div
                      key={feature}
                      className="flex items-center justify-between"
                    >
                      <span className="font-medium">{feature}</span>
                      <span
                        className={`font-mono ${
                          value > 0 ? "text-red-600" : "text-green-600"
                        }`}
                      >
                        {value > 0 ? "+" : ""}
                        {value.toFixed(4)}
                      </span>
                    </div>
                  )
                )}
              </div>
            </div>
            <div className="p-3 mt-4 text-sm text-yellow-800 border border-yellow-200 rounded-lg bg-yellow-50">
              <strong>Note:</strong> Positive values increase risk prediction,
              negative values decrease it.
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="container px-4 py-8 mx-auto max-w-7xl">
        <div className="mb-8 text-center">
          <div className="flex items-center justify-center gap-3 mb-3">
            <Activity className="w-10 h-10 text-indigo-600" />
            <h1 className="text-4xl font-bold text-gray-800">
              CHD Risk Prediction System
            </h1>
          </div>
          <p className="text-lg text-gray-600">
            AI-powered clinical decision support system with explainable
            predictions
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* LEFT PANEL - Patient Information Form */}
          <div className="p-6 bg-white shadow-lg rounded-xl">
            <h2 className="mb-6 text-xl font-bold text-gray-800">
              Patient Information
            </h2>

            <div className="space-y-6">
              {/* Demographics Section */}
              <div>
                <h3 className="flex items-center gap-2 mb-4 text-base font-semibold text-gray-700">
                  <Info className="w-4 h-4" />
                  Demographics
                </h3>
                <div className="space-y-4">
                  <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                      Gender
                    </label>
                    <select
                      name="gender"
                      value={formData.gender}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value={0}>Female</option>
                      <option value={1}>Male</option>
                    </select>
                  </div>
                  <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                      Age (years)
                    </label>
                    <input
                      type="number"
                      name="age"
                      value={formData.age}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>
                </div>
              </div>

              {/* Lifestyle Section */}
              <div>
                <h3 className="flex items-center gap-2 mb-4 text-base font-semibold text-gray-700">
                  <Activity className="w-4 h-4" />
                  Lifestyle
                </h3>
                <div className="space-y-4">
                  <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                      Current Smoker
                    </label>
                    <select
                      name="currentSmoker"
                      value={formData.currentSmoker}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                      Cigarettes Per Day
                    </label>
                    <input
                      type="number"
                      name="cigsPerDay"
                      value={formData.cigsPerDay}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>
                  <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                      BMI
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      name="BMI"
                      value={formData.BMI}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>
                </div>
              </div>

              {/* Medical History Section */}
              <div>
                <h3 className="mb-4 text-base font-semibold text-gray-700">
                  Medical History
                </h3>
                <div className="space-y-4">
                  <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                      BP Medication
                    </label>
                    <select
                      name="BPMeds"
                      value={formData.BPMeds}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                      Prevalent Stroke
                    </label>
                    <select
                      name="prevalentStroke"
                      value={formData.prevalentStroke}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                      Hypertension
                    </label>
                    <select
                      name="prevalentHyp"
                      value={formData.prevalentHyp}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                      Diabetes
                    </label>
                    <select
                      name="diabetes"
                      value={formData.diabetes}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Blood Pressure Section */}
              <div>
                <h3 className="mb-4 text-base font-semibold text-gray-700">
                  Blood Pressure
                </h3>
                <div className="space-y-4">
                  <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                      Systolic BP (mmHg)
                    </label>
                    <input
                      type="number"
                      name="sysBP"
                      value={formData.sysBP}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>
                  <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                      Diastolic BP (mmHg)
                    </label>
                    <input
                      type="number"
                      name="diaBP"
                      value={formData.diaBP}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>
                  <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                      Heart Rate (bpm)
                    </label>
                    <input
                      type="number"
                      name="heartRate"
                      value={formData.heartRate}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>
                </div>
              </div>

              {/* Lab Results Section */}
              <div>
                <h3 className="mb-4 text-base font-semibold text-gray-700">
                  Lab Results
                </h3>
                <div className="space-y-4">
                  <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                      Total Cholesterol (mg/dL)
                    </label>
                    <input
                      type="number"
                      name="totChol"
                      value={formData.totChol}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>
                  <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                      Glucose (mg/dL)
                    </label>
                    <input
                      type="number"
                      name="glucose"
                      value={formData.glucose}
                      onChange={handleInputChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>
                </div>
              </div>

              <button
                onClick={handlePredict}
                disabled={loading}
                className="flex items-center justify-center w-full gap-2 px-6 py-3 font-medium text-white transition bg-indigo-600 rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {loading ? "Processing..." : "Predict Risk"}
                <TrendingUp className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* RIGHT PANEL - AI Risk Assessment */}
          <div className="p-6 bg-white shadow-lg rounded-xl">
            <div className="flex items-center gap-2 mb-6">
              <CheckCircle className="w-5 h-5 text-indigo-600" />
              <h2 className="text-xl font-semibold text-gray-800">
                AI Risk Assessment
              </h2>
            </div>

            {!prediction ? (
              <div className="flex flex-col items-center justify-center h-96">
                <Activity className="w-20 h-20 mb-4 text-gray-300" />
                <p className="text-lg text-center text-gray-400">
                  Enter patient data and click "Predict Risk" to see results
                </p>
              </div>
            ) : (
              <>
                <div
                  className={`rounded-xl p-6 mb-6 ${
                    prediction.risk_level === "Low" ||
                    prediction.risk_level === "Moderate"
                      ? "bg-yellow-50 border-2 border-yellow-200"
                      : "bg-red-50 border-2 border-red-200"
                  }`}
                >
                  <div className="text-center">
                    <div className="mb-2 text-sm text-gray-600">Risk Level</div>
                    <div
                      className={`text-5xl font-bold mb-2 ${
                        prediction.risk_level === "Low" ||
                        prediction.risk_level === "Moderate"
                          ? "text-yellow-600"
                          : "text-red-600"
                      }`}
                    >
                      {prediction.risk_level.toUpperCase()}
                    </div>
                    <div className="mb-1 text-3xl font-semibold text-gray-800">
                      {prediction.risk_percentage.toFixed(1)}% probability
                    </div>
                    <div className="text-sm text-gray-600">
                      of CHD event in 10 years
                    </div>
                  </div>
                </div>

                <div
                  className={`flex items-center gap-2 p-4 mb-6 border-l-4 rounded ${
                    prediction.prediction === 1
                      ? "border-red-500 bg-red-50"
                      : "border-green-500 bg-green-50"
                  }`}
                >
                  {prediction.prediction === 1 ? (
                    <>
                      <AlertCircle className="w-5 h-5 text-red-600" />
                      <span className="font-medium text-red-800">
                        CHD Event Risk Detected
                      </span>
                    </>
                  ) : (
                    <>
                      <CheckCircle className="w-5 h-5 text-green-600" />
                      <span className="font-medium text-green-800">
                        No CHD Event Expected
                      </span>
                    </>
                  )}
                </div>

                <div>
                  <h3 className="mb-4 text-lg font-semibold text-gray-800">
                    Key Contributing Factors
                  </h3>
                  <div className="space-y-4">
                    <FactorBar
                      icon={<TrendingUp className="w-4 h-4" />}
                      label={`Age (${formData.age} years) impact on risk`}
                      value={formData.age > 60 ? 0.45 : -0.15}
                      isPositive={formData.age <= 60}
                    />
                    <FactorBar
                      icon={<Activity className="w-4 h-4" />}
                      label={`Systolic BP (${formData.sysBP} mmHg) ${
                        formData.sysBP > 140 ? "increases" : "normal level"
                      }`}
                      value={formData.sysBP > 140 ? 0.35 : -0.1}
                      isPositive={formData.sysBP <= 140}
                    />
                    <FactorBar
                      icon={<AlertCircle className="w-4 h-4" />}
                      label={`Cholesterol (${formData.totChol} mg/dL) ${
                        formData.totChol > 240 ? "elevated" : "normal range"
                      }`}
                      value={formData.totChol > 240 ? 0.28 : -0.08}
                      isPositive={formData.totChol <= 240}
                    />
                    <FactorBar
                      icon={<Info className="w-4 h-4" />}
                      label={`Smoking status (${
                        formData.currentSmoker === 1
                          ? "Active smoker"
                          : "Non-smoker"
                      })`}
                      value={formData.currentSmoker === 1 ? 0.32 : -0.2}
                      isPositive={formData.currentSmoker === 0}
                    />
                    <FactorBar
                      icon={<Heart className="w-4 h-4" />}
                      label={`Diabetes (${
                        formData.diabetes === 1 ? "Present" : "Absent"
                      })`}
                      value={formData.diabetes === 1 ? 0.25 : -0.12}
                      isPositive={formData.diabetes === 0}
                    />
                  </div>
                </div>

                
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

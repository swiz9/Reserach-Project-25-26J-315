import React, { useState } from 'react';
import { Activity, AlertCircle, CheckCircle, TrendingUp, TrendingDown, Minus } from 'lucide-react';

const HeartFailurePredictor = () => {
  const [formData, setFormData] = useState({
    age: '',
    ejection_fraction: '',
    serum_creatinine: '',
    serum_sodium: '',
    time: '',
    anaemia: '0',
    high_blood_pressure: '0',
    diabetes: '0',
    sex: '1',
    creatinine_phosphokinase: '',
    platelets: ''
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const featureDescriptions = {
    age: { label: 'Age', unit: 'years', range: '40-95' },
    ejection_fraction: { label: 'Ejection Fraction', unit: '%', range: '14-80' },
    serum_creatinine: { label: 'Serum Creatinine', unit: 'mg/dL', range: '0.5-9.4' },
    serum_sodium: { label: 'Serum Sodium', unit: 'mEq/L', range: '113-148' },
    anaemia: { label: 'Anaemia', unit: '', range: 'Yes/No' },
    high_blood_pressure: { label: 'High Blood Pressure', unit: '', range: 'Yes/No' },
    diabetes: { label: 'Diabetes', unit: '', range: 'Yes/No' },
    sex: { label: 'Sex', unit: '', range: 'Male/Female' },
    creatinine_phosphokinase: { label: 'Creatinine Phosphokinase', unit: 'mcg/L', range: '23-7861' },
    platelets: { label: 'Platelets', unit: 'kiloplatelets/mL', range: '25100-850000' }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const generateMockPrediction = (data) => {
    // Simulate model prediction based on risk factors
    let riskScore = 0;
    
    // Age factor
    if (parseFloat(data.age) > 65) riskScore += 0.2;
    
    // Ejection fraction (lower is worse)
    if (parseFloat(data.ejection_fraction) < 30) riskScore += 0.3;
    else if (parseFloat(data.ejection_fraction) < 40) riskScore += 0.15;
    
    // Serum creatinine (higher is worse)
    if (parseFloat(data.serum_creatinine) > 1.5) riskScore += 0.25;
    
    // Serum sodium (lower is worse)
    if (parseFloat(data.serum_sodium) < 135) riskScore += 0.15;
    
    // Comorbidities
    if (data.anaemia === '1') riskScore += 0.1;
    if (data.high_blood_pressure === '1') riskScore += 0.08;
    if (data.diabetes === '1') riskScore += 0.07;
    
    // Add some randomness
    riskScore = Math.min(0.95, Math.max(0.05, riskScore + (Math.random() * 0.1 - 0.05)));
    
    const topFactors = [
      {
        feature: 'ejection_fraction',
        value: data.ejection_fraction,
        impact: parseFloat(data.ejection_fraction) < 35 ? 'strongly increases risk' : 'moderately reduces risk',
        shap_value: parseFloat(data.ejection_fraction) < 35 ? 0.65 : -0.45,
        description: `Heart pumping ability (${data.ejection_fraction}%) ${parseFloat(data.ejection_fraction) < 35 ? 'strongly increases risk' : 'moderately reduces risk'}`
      },
      {
        feature: 'serum_creatinine',
        value: data.serum_creatinine,
        impact: parseFloat(data.serum_creatinine) > 1.5 ? 'moderately increases risk' : 'slightly reduces risk',
        shap_value: parseFloat(data.serum_creatinine) > 1.5 ? 0.35 : -0.2,
        description: `Kidney function (${data.serum_creatinine} mg/dL) ${parseFloat(data.serum_creatinine) > 1.5 ? 'moderately increases risk' : 'slightly reduces risk'}`
      },
      {
        feature: 'age',
        value: data.age,
        impact: parseFloat(data.age) > 65 ? 'moderately increases risk' : 'slightly reduces risk',
        shap_value: parseFloat(data.age) > 65 ? 0.28 : -0.15,
        description: `Age (${data.age} years) ${parseFloat(data.age) > 65 ? 'moderately increases risk' : 'slightly reduces risk'}`
      },
      {
        feature: 'serum_sodium',
        value: data.serum_sodium,
        impact: parseFloat(data.serum_sodium) < 135 ? 'slightly increases risk' : 'slightly reduces risk',
        shap_value: parseFloat(data.serum_sodium) < 135 ? 0.18 : -0.12,
        description: `Blood sodium level (${data.serum_sodium} mEq/L) ${parseFloat(data.serum_sodium) < 135 ? 'slightly increases risk' : 'slightly reduces risk'}`
      },
      {
        feature: 'anaemia',
        value: data.anaemia === '1' ? 'Yes' : 'No',
        impact: data.anaemia === '1' ? 'slightly increases risk' : 'has minimal impact',
        shap_value: data.anaemia === '1' ? 0.12 : -0.05,
        description: `Anaemia presence (${data.anaemia === '1' ? 'Yes' : 'No'}) ${data.anaemia === '1' ? 'slightly increases risk' : 'has minimal impact'}`
      }
    ];

    return {
      probability: riskScore,
      prediction: riskScore >= 0.5 ? 1 : 0,
      topFactors: topFactors.sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    };
  };

  const handleSubmit = () => {
    setLoading(true);

    // Simulate API call delay
    setTimeout(() => {
      const result = generateMockPrediction(formData);
      setPrediction(result);
      setLoading(false);
    }, 1500);
  };

  const getRiskLevel = (probability) => {
    if (probability < 0.4) return { level: 'LOW', color: 'text-green-600', bgColor: 'bg-green-50', borderColor: 'border-green-200' };
    if (probability < 0.7) return { level: 'MODERATE', color: 'text-yellow-600', bgColor: 'bg-yellow-50', borderColor: 'border-yellow-200' };
    return { level: 'HIGH', color: 'text-red-600', bgColor: 'bg-red-50', borderColor: 'border-red-200' };
  };

  const getImpactIcon = (shap_value) => {
    if (Math.abs(shap_value) < 0.1) return <Minus className="w-4 h-4 text-gray-400" />;
    return shap_value > 0 ? <TrendingUp className="w-4 h-4 text-red-500" /> : <TrendingDown className="w-4 h-4 text-green-500" />;
  };

  return (
    <div className="min-h-screen p-6 bg-gradient-to-br from-blue-50 to-indigo-50">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8 text-center">
          <div className="flex items-center justify-center mb-4">
            <Activity className="w-12 h-12 mr-3 text-indigo-600" />
            <h1 className="text-4xl font-bold text-gray-800">Heart Failure Risk Predictor</h1>
          </div>
          <p className="text-gray-600">AI-powered clinical decision support system with explainable predictions</p>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Input Form */}
          <div className="p-6 bg-white shadow-lg rounded-xl">
            <h2 className="flex items-center mb-6 text-2xl font-bold text-gray-800">
              <AlertCircle className="w-6 h-6 mr-2 text-indigo-600" />
              Patient Clinical Data
            </h2>
            
            <div className="space-y-4">
              {/* Numeric Inputs */}
              {['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'creatinine_phosphokinase', 'platelets'].map(field => (
                <div key={field}>
                  <label className="block mb-1 text-sm font-medium text-gray-700">
                    {featureDescriptions[field].label}
                    {featureDescriptions[field].unit && ` (${featureDescriptions[field].unit})`}
                  </label>
                  <input
                    type="number"
                    step="any"
                    name={field}
                    value={formData[field]}
                    onChange={handleInputChange}
                    placeholder={`Range: ${featureDescriptions[field].range}`}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  />
                </div>
              ))}

              {/* Binary Inputs */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block mb-1 text-sm font-medium text-gray-700">Sex</label>
                  <select
                    name="sex"
                    value={formData.sex}
                    onChange={handleInputChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  >
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                  </select>
                </div>

                <div>
                  <label className="block mb-1 text-sm font-medium text-gray-700">Anaemia</label>
                  <select
                    name="anaemia"
                    value={formData.anaemia}
                    onChange={handleInputChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  >
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block mb-1 text-sm font-medium text-gray-700">High Blood Pressure</label>
                  <select
                    name="high_blood_pressure"
                    value={formData.high_blood_pressure}
                    onChange={handleInputChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  >
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>

                <div>
                  <label className="block mb-1 text-sm font-medium text-gray-700">Diabetes</label>
                  <select
                    name="diabetes"
                    value={formData.diabetes}
                    onChange={handleInputChange}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  >
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>
              </div>

              <button
                onClick={handleSubmit}
                disabled={loading}
                className="w-full py-3 font-semibold text-white transition-colors bg-indigo-600 rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {loading ? 'Analyzing...' : 'Predict Risk'}
              </button>
            </div>
          </div>

          {/* Results Panel */}
          <div className="p-6 bg-white shadow-lg rounded-xl">
            <h2 className="flex items-center mb-6 text-2xl font-bold text-gray-800">
              <CheckCircle className="w-6 h-6 mr-2 text-indigo-600" />
              AI Risk Assessment
            </h2>

            {!prediction ? (
              <div className="flex flex-col items-center justify-center text-gray-400 h-96">
                <Activity className="w-16 h-16 mb-4" />
                <p className="text-lg">Enter patient data and click "Predict Risk" to see results</p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Risk Level */}
                <div className={`border-2 ${getRiskLevel(prediction.probability).borderColor} ${getRiskLevel(prediction.probability).bgColor} rounded-lg p-6`}>
                  <div className="text-center">
                    <p className="mb-2 text-sm font-medium text-gray-600">Risk Level</p>
                    <p className={`text-4xl font-bold ${getRiskLevel(prediction.probability).color} mb-2`}>
                      {getRiskLevel(prediction.probability).level}
                    </p>
                    <p className="text-2xl font-semibold text-gray-700">
                      {(prediction.probability * 100).toFixed(1)}% probability
                    </p>
                    <p className="mt-2 text-sm text-gray-500">of death event</p>
                  </div>
                </div>

                {/* Clinical Decision */}
                <div className={`border-l-4 p-4 ${prediction.prediction === 1 ? 'border-red-500 bg-red-50' : 'border-green-500 bg-green-50'}`}>
                  <p className="font-semibold text-gray-700">
                    {prediction.prediction === 1 ? '⚠️ Death Event Likely' : '✅ No Death Event Expected'}
                  </p>
                </div>

                {/* SHAP Explanations */}
                <div>
                  <h3 className="mb-4 text-lg font-bold text-gray-800">Key Contributing Factors</h3>
                  <div className="space-y-3">
                    {prediction.topFactors.map((factor, index) => (
                      <div key={index} className="p-4 transition-shadow border border-gray-200 rounded-lg hover:shadow-md">
                        <div className="flex items-start justify-between">
                          <div className="flex items-start flex-1 space-x-3">
                            {getImpactIcon(factor.shap_value)}
                            <div className="flex-1">
                              <p className="text-sm font-medium text-gray-800">{factor.description}</p>
                              <div className="h-2 mt-2 overflow-hidden bg-gray-200 rounded-full">
                                <div
                                  className={`h-full ${factor.shap_value > 0 ? 'bg-red-500' : 'bg-green-500'}`}
                                  style={{ width: `${Math.min(100, Math.abs(factor.shap_value) * 100)}%` }}
                                />
                              </div>
                            </div>
                          </div>
                          <span className={`text-xs font-semibold px-2 py-1 rounded ${factor.shap_value > 0 ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                            {factor.shap_value > 0 ? '+' : ''}{factor.shap_value.toFixed(2)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Clinical Recommendation */}
                <div className="p-4 border border-blue-200 rounded-lg bg-blue-50">
                  <p className="mb-2 text-sm font-semibold text-blue-900">💡 Clinical Recommendation</p>
                  <p className="text-sm text-blue-800">
                    {prediction.probability > 0.7
                      ? 'High-risk patient requiring immediate clinical attention and intensive monitoring.'
                      : prediction.probability > 0.4
                      ? 'Moderate-risk patient. Consider enhanced monitoring and preventive interventions.'
                      : 'Low-risk patient. Continue standard care protocols with routine follow-up.'}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default HeartFailurePredictor;
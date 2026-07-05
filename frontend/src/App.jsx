import { useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const FIELDS = [
  { key: "Cycle", label: "Cycle" },
  { key: "Capacity_Ah", label: "Capacity (Ah)" },
  { key: "Internal_Resistance_Ohm", label: "Internal Resistance (Ohm)" },
  { key: "Temperature_C", label: "Temperature (°C)" },
  { key: "Voltage_Max_V", label: "Voltage Max (V)" },
  { key: "Voltage_Min_V", label: "Voltage Min (V)" },
  { key: "Charge_Time_s", label: "Charge Time (s)" },
  { key: "Discharge_Time_s", label: "Discharge Time (s)" },
  { key: "Ambient_Temp_C", label: "Ambient Temperature (°C)" },
];

const EXAMPLE_VALUES = {
  Cycle: "10",
  Capacity_Ah: "1.95",
  Internal_Resistance_Ohm: "0.047",
  Temperature_C: "32.8",
  Voltage_Max_V: "4.19",
  Voltage_Min_V: "3.2",
  Charge_Time_s: "3600",
  Discharge_Time_s: "3000",
  Ambient_Temp_C: "32.8",
};

export default function App() {
  const [values, setValues] = useState(
    Object.fromEntries(FIELDS.map((f) => [f.key, ""]))
  );
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  function handleChange(key, value) {
    setValues((prev) => ({ ...prev, [key]: value }));
  }

  function fillExample() {
    setValues(EXAMPLE_VALUES);
    setPrediction(null);
    setError(null);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setPrediction(null);
    setLoading(true);

    const features = FIELDS.map((f) => Number(values[f.key]));

    if (features.some((v) => Number.isNaN(v))) {
      setError("All fields must be valid numbers.");
      setLoading(false);
      return;
    }

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setError(body.detail || `Request failed with status ${response.status}`);
        return;
      }

      const data = await response.json();
      setPrediction(data.prediction);
    } catch {
      setError("Could not reach the prediction API.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <div className="card">
        <h1 className="title">Battery Health Predictor</h1>
        <p className="subtitle">
          Enter the 9 battery cycle measurements to estimate State of Health (SOH).
        </p>

        <button type="button" className="example-btn" onClick={fillExample}>
          Fill with example values
        </button>

        <form onSubmit={handleSubmit}>
          <div className="grid">
            {FIELDS.map((f) => (
              <div key={f.key} className="field">
                <label htmlFor={f.key}>{f.label}</label>
                <input
                  id={f.key}
                  type="number"
                  step="any"
                  value={values[f.key]}
                  onChange={(e) => handleChange(f.key, e.target.value)}
                  required
                />
              </div>
            ))}
          </div>

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? "Predicting..." : "Predict"}
          </button>
        </form>

        {prediction !== null && (
          <div className="result success">
            Predicted SOH
            <strong>{Array.isArray(prediction) ? prediction[0] : prediction}</strong>
          </div>
        )}
        {error && <div className="result error">{error}</div>}
      </div>
    </div>
  );
}

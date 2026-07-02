import { useState } from "react";

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
    <div style={{ maxWidth: 480, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>Battery Health Predictor</h1>
      <form onSubmit={handleSubmit}>
        {FIELDS.map((f) => (
          <div key={f.key} style={{ marginBottom: 12 }}>
            <label style={{ display: "block", marginBottom: 4 }}>{f.label}</label>
            <input
              type="number"
              step="any"
              value={values[f.key]}
              onChange={(e) => handleChange(f.key, e.target.value)}
              required
              style={{ width: "100%", padding: 8 }}
            />
          </div>
        ))}
        <button type="submit" disabled={loading} style={{ padding: "8px 16px" }}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {prediction !== null && (
        <p style={{ marginTop: 16 }}>
          Predicted SOH: <strong>{Array.isArray(prediction) ? prediction[0] : prediction}</strong>
        </p>
      )}
      {error && <p style={{ marginTop: 16, color: "red" }}>{error}</p>}
    </div>
  );
}

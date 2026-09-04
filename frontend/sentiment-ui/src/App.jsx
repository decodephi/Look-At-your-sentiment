import { useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";


function App() {

  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");


  const predictSentiment = async () => {

    if (!text.trim()) {
      setError("Please enter a sentence.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Unable to classify this text.");

      setResult(data);

    } catch (err) {

      setError(
        err.message ||
        "Unable to connect to the API."
      );

    } finally {

      setLoading(false);

    }
  };


  return (
    <div className="container">

      <div className="card">

        <p className="eyebrow">IMDB review intelligence</p>
        <h1>Read the feeling<br /><em>between the lines.</em></h1>

        <p>
          Paste a review, reaction, or thought. The production model will weigh its tone in seconds.
        </p>


        <textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="The cinematography was stunning, but the story never found its rhythm..."
          rows="6"
        />


        <button
          onClick={predictSentiment}
          disabled={loading}
        >
          {loading ? "Reading tone..." : "Analyze sentiment"}
        </button>


        {error && (
          <div className="error">
            {error}
          </div>
        )}


        {result && (
          <div className="result">

            <h2>Model read</h2>

            <p>
              <strong>Sentiment:</strong>{" "}
              {result.sentiment}
            </p>

            <p>
              <strong>Confidence:</strong>{" "}
              {result.confidence !== null
                ? `${(result.confidence * 100).toFixed(2)}%`
                : "N/A"
              }
            </p>

          </div>
        )}

      </div>

    </div>
  );
}


export default App;

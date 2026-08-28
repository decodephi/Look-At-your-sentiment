import { useState } from "react";
import axios from "axios";
import "./App.css";


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

      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        {
          text: text
        }
      );

      setResult(response.data);

    } catch (err) {

      setError(
        err.response?.data?.detail ||
        "Unable to connect to the API."
      );

    } finally {

      setLoading(false);

    }
  };


  return (
    <div className="container">

      <div className="card">

        <h1>Sentiment Analyzer</h1>

        <p>
          Enter a sentence and let the ML model
          classify its sentiment.
        </p>


        <textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Enter your sentence..."
          rows="6"
        />


        <button
          onClick={predictSentiment}
          disabled={loading}
        >
          {loading ? "Analyzing..." : "Analyze Sentiment"}
        </button>


        {error && (
          <div className="error">
            {error}
          </div>
        )}


        {result && (
          <div className="result">

            <h2>Result</h2>

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

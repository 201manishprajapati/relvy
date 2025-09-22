import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("prompt", prompt);

    const res = await fetch("http://localhost:8000/analyze/", {
      method: "POST",
      body: formData
    });
    setResult(await res.json());
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>Log Analysis App</h2>
      <input type="file" onChange={e => setFile(e.target.files[0])} />
      <br/><br/>
      <input
        type="text"
        value={prompt}
        onChange={e => setPrompt(e.target.value)}
        placeholder="Describe incident..."
        style={{ width: "300px" }}
      />
      <br/><br/>
      <button onClick={handleSubmit}>Analyze</button>

      {result && (
        <div style={{ marginTop: "20px" }}>
          <h3>Filtered Logs</h3>
          <pre>{JSON.stringify(result.filtered_logs, null, 2)}</pre>
          <h3>Highlighted Logs</h3>
          <pre>{JSON.stringify(result.highlighted_logs, null, 2)}</pre>
          <h3>Cost</h3>
          <pre>{JSON.stringify(result.cost, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;

import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const DEFAULT_CONFIG = {
  generation_model: "gpt-4o-mini",
  top_k: 5,
  score_threshold: 0.3,
  use_reranking: true,
  temperature: 0.3,
  top_p: 0.9,
  limit: 5,
};

const SAMPLE_QUESTIONS = [
  "Theo Luật Phòng, chống ma túy 2021, chất ma túy được hiểu là gì?",
  "Các hình thức cai nghiện ma túy theo Luật Phòng, chống ma túy 2021 gồm những gì?",
  "Trong vụ án ma túy tại TP.HCM được Thanh Niên nêu, những nghệ sĩ nào nằm trong số các bị can?",
  "Bài VietnamNet mô tả ma túy ảnh hưởng đến não bộ người trẻ như thế nào?",
];

function App() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [models, setModels] = useState([DEFAULT_CONFIG.generation_model]);
  const [health, setHealth] = useState(null);
  const [question, setQuestion] = useState(SAMPLE_QUESTIONS[0]);
  const [chatResult, setChatResult] = useState(null);
  const [evalResult, setEvalResult] = useState(null);
  const [chatLoading, setChatLoading] = useState(false);
  const [evalLoading, setEvalLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadInitialState() {
      try {
        const [configResponse, healthResponse] = await Promise.all([
          fetch("/api/config"),
          fetch("/api/health"),
        ]);
        const configData = await configResponse.json();
        const healthData = await healthResponse.json();
        setModels(configData.models || models);
        setConfig((current) => ({
          ...current,
          ...(configData.defaults || {}),
          limit: current.limit,
        }));
        setHealth(healthData);
      } catch (err) {
        setError(`Không kết nối được backend: ${err.message}`);
      }
    }

    loadInitialState();
  }, []);

  const requestConfig = useMemo(
    () => ({
      generation_model: config.generation_model,
      top_k: Number(config.top_k),
      score_threshold: Number(config.score_threshold),
      use_reranking: Boolean(config.use_reranking),
      temperature: Number(config.temperature),
      top_p: Number(config.top_p),
    }),
    [config],
  );

  function updateConfig(key, value) {
    setConfig((current) => ({ ...current, [key]: value }));
  }

  async function runChat() {
    setChatLoading(true);
    setError("");
    setChatResult(null);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...requestConfig, question }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Chat request failed");
      }
      setChatResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setChatLoading(false);
    }
  }

  async function runRagas() {
    setEvalLoading(true);
    setError("");
    setEvalResult(null);

    try {
      const response = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...requestConfig, limit: Number(config.limit) }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Evaluation request failed");
      }
      setEvalResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setEvalLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="control-panel">
        <div className="brand">
          <div>
            <h1>DrugLaw RAG Demo</h1>
            <p>FastAPI + React dashboard cho chatbot và RAGAS evaluation.</p>
          </div>
          <StatusPill health={health} />
        </div>

        <section className="panel-section">
          <h2>Model</h2>
          <label className="field">
            <span>Generation model</span>
            <select
              value={config.generation_model}
              onChange={(event) => updateConfig("generation_model", event.target.value)}
            >
              {models.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </label>
          <div className="two-cols">
            <NumberField
              label="Temperature"
              value={config.temperature}
              min={0}
              max={1.5}
              step={0.1}
              onChange={(value) => updateConfig("temperature", value)}
            />
            <NumberField
              label="Top-p"
              value={config.top_p}
              min={0.1}
              max={1}
              step={0.05}
              onChange={(value) => updateConfig("top_p", value)}
            />
          </div>
        </section>

        <section className="panel-section">
          <h2>Retrieval Config</h2>
          <label className="switch-row">
            <span>
              <strong>Reranking</strong>
              <small>{config.use_reranking ? "Hybrid + rerank" : "Hybrid không rerank"}</small>
            </span>
            <input
              type="checkbox"
              checked={config.use_reranking}
              onChange={(event) => updateConfig("use_reranking", event.target.checked)}
            />
          </label>
          <RangeField
            label="Top K"
            value={config.top_k}
            min={1}
            max={12}
            step={1}
            onChange={(value) => updateConfig("top_k", value)}
          />
          <RangeField
            label="Fallback threshold"
            value={config.score_threshold}
            min={0}
            max={1}
            step={0.05}
            onChange={(value) => updateConfig("score_threshold", value)}
          />
        </section>

        <section className="panel-section">
          <h2>RAGAS</h2>
          <RangeField
            label="Số golden cases"
            value={config.limit}
            min={1}
            max={20}
            step={1}
            onChange={(value) => updateConfig("limit", value)}
          />
          <button className="secondary-action" onClick={runRagas} disabled={evalLoading}>
            {evalLoading ? "Đang chạy RAGAS..." : "Evaluate bằng RAGAS"}
          </button>
        </section>

        <ConfigSummary config={config} />
        <HealthBox health={health} />
      </aside>

      <section className="workspace">
        <section className="dashboard-hero">
          <div className="hero-copy">
            <span className="eyebrow">RAG chatbot demo</span>
            <h2>Pháp luật ma túy và tin tức liên quan</h2>
            <p>
              Điều khiển retrieval, generation và RAGAS trên cùng một màn hình để demo pipeline end-to-end.
            </p>
          </div>
          <div className="stat-grid">
            <MiniStat label="OpenAI" value={health?.openai_api_key ? "Ready" : "Missing"} tone={health?.openai_api_key ? "ok" : "warn"} />
            <MiniStat label="Docs" value={health?.standardized_docs ?? "-"} />
            <MiniStat label="Golden" value={health?.golden_cases ?? "-"} />
            <MiniStat label="Mode" value={config.use_reranking ? "Rerank" : "No rerank"} />
          </div>
        </section>

        {error && <div className="error-banner">{error}</div>}

        <section className="query-band">
          <div className="query-header">
            <div>
              <h2>Chat RAG</h2>
              <p>Nhập câu hỏi, chọn cấu hình bên trái, rồi chạy pipeline.</p>
            </div>
            <button className="primary-action" onClick={runChat} disabled={chatLoading}>
              {chatLoading ? "Đang chạy..." : "Chạy truy vấn RAG"}
            </button>
          </div>

          <label className="field">
            <span>Câu hỏi mẫu</span>
            <select value={question} onChange={(event) => setQuestion(event.target.value)}>
              {SAMPLE_QUESTIONS.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>

          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            rows={4}
            placeholder="Nhập câu hỏi về pháp luật ma túy hoặc tin tức liên quan..."
          />
        </section>

        <section className="result-grid">
          <article className="answer-panel">
            <div className="section-title">
              <h2>Câu trả lời</h2>
              {chatResult && (
                <span className="meta-pill">
                  {chatResult.generation_mode} · {chatResult.retrieval_source}
                </span>
              )}
            </div>
            {chatResult ? (
              <>
                {chatResult.error && (
                  <div className="warning-box">
                    Fallback: {chatResult.error}
                  </div>
                )}
                <div className="answer-text">{chatResult.answer}</div>
              </>
            ) : (
              <EmptyState text="Chưa có kết quả. Bấm Hỏi RAG để chạy pipeline." />
            )}
          </article>

          <article className="sources-panel">
            <div className="section-title">
              <h2>Sources</h2>
              <span className="meta-pill">{chatResult?.sources?.length || 0} chunks</span>
            </div>
            {chatResult?.sources?.length ? (
              <div className="source-list">
                {chatResult.sources.map((source) => (
                  <SourceItem key={`${source.rank}-${source.preview}`} source={source} />
                ))}
              </div>
            ) : (
              <EmptyState text="Nguồn truy xuất sẽ xuất hiện ở đây." />
            )}
          </article>
        </section>

        <section className="eval-panel">
          <div className="section-title">
            <h2>RAGAS Results</h2>
            {evalResult && <span className="meta-pill">{evalResult.total_cases} cases</span>}
          </div>
          {evalResult ? <EvaluationResult result={evalResult} /> : <EmptyState text="Bấm Chạy RAGAS để hiển thị bảng điểm." />}
        </section>
      </section>
    </main>
  );
}

function StatusPill({ health }) {
  if (!health) {
    return <span className="status-pill muted">Backend</span>;
  }
  return <span className={`status-pill ${health.openai_api_key ? "ok" : "warn"}`}>
    {health.openai_api_key ? "Ready" : "No API key"}
  </span>;
}

function HealthBox({ health }) {
  return (
    <section className="health-box">
      <h2>Runtime</h2>
      <dl>
        <div>
          <dt>OpenAI key</dt>
          <dd>{health?.openai_api_key ? "Có" : "Thiếu"}</dd>
        </div>
        <div>
          <dt>Markdown docs</dt>
          <dd>{health?.standardized_docs ?? "-"}</dd>
        </div>
        <div>
          <dt>Golden cases</dt>
          <dd>{health?.golden_cases ?? "-"}</dd>
        </div>
      </dl>
    </section>
  );
}

function ConfigSummary({ config }) {
  return (
    <section className="config-summary">
      <h2>Active Config</h2>
      <div className="chip-grid">
        <span>{config.generation_model}</span>
        <span>top_k {config.top_k}</span>
        <span>{config.use_reranking ? "rerank on" : "rerank off"}</span>
        <span>threshold {config.score_threshold}</span>
      </div>
    </section>
  );
}

function MiniStat({ label, value, tone = "" }) {
  return (
    <div className={`mini-stat ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function NumberField({ label, value, min, max, step, onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function RangeField({ label, value, min, max, step, onChange }) {
  return (
    <label className="field">
      <span className="range-label">
        {label}
        <strong>{value}</strong>
      </span>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function SourceItem({ source }) {
  const metadata = source.metadata || {};
  return (
    <details className="source-item">
      <summary>
        <span>#{source.rank} {metadata.source || metadata.path || source.source}</span>
        <strong>{source.score == null ? "n/a" : source.score.toFixed(3)}</strong>
      </summary>
      <p>{source.preview}</p>
      <dl>
        <div>
          <dt>Path</dt>
          <dd>{metadata.path || "-"}</dd>
        </div>
        <div>
          <dt>Chunk</dt>
          <dd>{metadata.chunk_index ?? "-"}</dd>
        </div>
        <div>
          <dt>Reranker</dt>
          <dd>{source.reranker || "-"}</dd>
        </div>
      </dl>
    </details>
  );
}

function EvaluationResult({ result }) {
  const metrics = result.metrics || [];
  const records = result.records || [];
  return (
    <div className="eval-content">
      <div className="metric-row">
        {metrics.map((metric) => (
          <div className="metric-tile" key={metric}>
            <span>{metric}</span>
            <strong>{formatScore(result.averages?.[metric])}</strong>
          </div>
        ))}
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Question</th>
              {metrics.map((metric) => (
                <th key={metric}>{metric}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {records.map((record, index) => (
              <tr key={`${record.question}-${index}`}>
                <td>{record.question || record.user_input}</td>
                {metrics.map((metric) => (
                  <td key={metric}>{formatScore(record[metric])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="result-path">Saved report: {result.results_path}</p>
    </div>
  );
}

function EmptyState({ text }) {
  return <div className="empty-state">{text}</div>;
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(3);
}

createRoot(document.getElementById("root")).render(<App />);

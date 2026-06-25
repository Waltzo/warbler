import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from "recharts";
import { api, GpuStat, JobStatus, STATE_KO, subscribeJob, TrainConfig } from "../api";

interface Point { step: number; loss?: number; eval_loss?: number; wer?: number; cer?: number; }

function mergeMetric(points: Point[], m: any): Point[] {
  const i = points.findIndex((p) => p.step === m.step);
  const patch: Point = { step: m.step };
  if (m.event === "log" && m.loss !== undefined) patch.loss = m.loss;
  if (m.event === "eval") {
    if (m.loss !== undefined) patch.eval_loss = m.loss;
    if (m.wer !== undefined) patch.wer = m.wer;
    if (m.cer !== undefined) patch.cer = m.cer;
  }
  if (i >= 0) {
    const next = [...points];
    next[i] = { ...next[i], ...patch };
    return next;
  }
  return [...points, patch].sort((a, b) => a.step - b.step);
}

export default function JobDetail() {
  const { id } = useParams();
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [cfg, setCfg] = useState<TrainConfig | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [log, setLog] = useState("");
  const [gpus, setGpus] = useState<GpuStat[]>([]);
  const logRef = useRef<HTMLDivElement>(null);
  const stickBottom = useRef(true); // only auto-scroll when user is at the bottom

  useEffect(() => {
    if (!id) return;
    api.job(id).then(setStatus);
    api.config(id).then(setCfg).catch(() => setCfg(null));
    api.metrics(id).then((ms) => {
      let pts: Point[] = [];
      ms.forEach((m) => { pts = mergeMetric(pts, m); });
      setPoints(pts);
    });
    const unsub = subscribeJob(id, {
      onMetric: (m) => setPoints((p) => mergeMetric(p, m)),
      onLog: (t) => setLog((l) => (l + t).slice(-50000)),
      onStatus: setStatus,
    });
    return unsub;
  }, [id]);

  // Poll GPU stats while running.
  useEffect(() => {
    const load = () => api.gpus().then(setGpus);
    load();
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (stickBottom.current && logRef.current)
      logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  const onLogScroll = () => {
    const el = logRef.current;
    if (!el) return;
    // "at bottom" with a small tolerance; scrolling up disables auto-scroll.
    stickBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  };

  if (!status) return <p>불러오는 중…</p>;

  const pct = status.total_steps
    ? Math.min(100, Math.round(((status.last_step || 0) / status.total_steps) * 100))
    : 0;
  const usedGpu = gpus.find((g) => g.index === status.gpu_index);

  return (
    <div>
      <h1>{status.name} <span className={`badge ${status.state}`}>{STATE_KO[status.state] || status.state}</span></h1>
      <div className="card compact">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div className="muted">
            {status.model_type} · {status.base_model} · dataset {status.dataset_id} · GPU {status.gpu_index}
          </div>
          {status.state === "running" && (
            <button className="danger" onClick={() => api.stopJob(id!).then(setStatus)}>중지</button>
          )}
        </div>
        <div style={{ marginTop: 8 }}>
          스텝 {status.last_step ?? 0}{status.total_steps ? ` / ${status.total_steps}` : ""} ({pct}%)
          <div className="gpu-bar" style={{ marginTop: 4 }}><div style={{ width: `${pct}%` }} /></div>
        </div>
        {status.error && <p style={{ color: "#dc2626" }}>오류: {status.error}</p>}
      </div>

      {cfg && <ConfigCard cfg={cfg} />}

      <div className="grid">
        <div className="card compact">
          <h2>Loss</h2>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={points}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" /><YAxis /><Tooltip /><Legend />
              <Line type="monotone" dataKey="loss" stroke="#2563eb" dot={false} name="학습 loss" connectNulls />
              <Line type="monotone" dataKey="eval_loss" stroke="#dc2626" dot={false} name="검증 loss" connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="card compact">
          <h2>WER / CER</h2>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={points}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" /><YAxis /><Tooltip /><Legend />
              <Line type="monotone" dataKey="wer" stroke="#16a34a" dot name="WER" connectNulls />
              <Line type="monotone" dataKey="cer" stroke="#d97706" dot name="CER" connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card compact">
        <h2>GPU {status.gpu_index}</h2>
        {usedGpu ? (
          <div>
            <div className="muted">{usedGpu.name} — util {usedGpu.utilization_pct}% · mem {Math.round(usedGpu.memory_used_mb)}/{Math.round(usedGpu.memory_total_mb)}MB</div>
            <div className="gpu-bar" style={{ marginTop: 6 }}>
              <div style={{ width: `${(usedGpu.memory_used_mb / usedGpu.memory_total_mb) * 100}%` }} />
            </div>
          </div>
        ) : <p className="muted">nvidia-smi 사용 불가.</p>}
      </div>

      <div className="card compact">
        <h2>로그</h2>
        <div className="log" ref={logRef} onScroll={onLogScroll}>{log || "(출력 대기 중…)"}</div>
      </div>
    </div>
  );
}

function ConfigCard({ cfg }: { cfg: TrainConfig }) {
  const effBatch = cfg.batch_size * cfg.grad_accum;
  const rows: [string, any][] = [
    ["precision", cfg.precision],
    ["learning_rate", cfg.learning_rate],
    ["batch", `${cfg.batch_size}×${cfg.grad_accum}=${effBatch}`],
    ["epochs", cfg.num_epochs],
    ["max_steps", cfg.max_steps],
    ["warmup", cfg.warmup_steps],
    ["eval_steps", cfg.eval_steps],
    ["save_steps", cfg.save_steps],
    ["eval_ratio", cfg.eval_ratio],
    ["save_limit", cfg.save_total_limit ?? "전체"],
    ["language", cfg.language ?? "-"],
    ["task", cfg.task],
    ["LoRA", cfg.use_lora ? `r${cfg.lora_r} α${cfg.lora_alpha} d${cfg.lora_dropout}` : "off"],
  ];
  return (
    <div className="card compact">
      <h2>설정</h2>
      <div className="kv">
        {rows.map(([k, v]) => (
          <div key={k}><span className="k">{k}</span><span className="v">{String(v)}</span></div>
        ))}
      </div>
    </div>
  );
}

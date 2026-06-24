import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from "recharts";
import { api, GpuStat, JobStatus, STATE_KO, subscribeJob } from "../api";

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
  const [points, setPoints] = useState<Point[]>([]);
  const [log, setLog] = useState("");
  const [gpus, setGpus] = useState<GpuStat[]>([]);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!id) return;
    api.job(id).then(setStatus);
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
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  if (!status) return <p>불러오는 중…</p>;

  const pct = status.total_steps
    ? Math.min(100, Math.round(((status.last_step || 0) / status.total_steps) * 100))
    : 0;
  const usedGpu = gpus.find((g) => g.index === status.gpu_index);

  return (
    <div>
      <h1>{status.name} <span className={`badge ${status.state}`}>{STATE_KO[status.state] || status.state}</span></h1>
      <div className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div className="muted">
            {status.model_type} · {status.base_model} · dataset {status.dataset_id} · GPU {status.gpu_index}
          </div>
          {status.state === "running" && (
            <button className="danger" onClick={() => api.stopJob(id!).then(setStatus)}>중지</button>
          )}
        </div>
        <div style={{ marginTop: 10 }}>
          스텝 {status.last_step ?? 0}{status.total_steps ? ` / ${status.total_steps}` : ""} ({pct}%)
          <div className="gpu-bar" style={{ marginTop: 4 }}><div style={{ width: `${pct}%` }} /></div>
        </div>
        {status.error && <p style={{ color: "#dc2626" }}>오류: {status.error}</p>}
      </div>

      <div className="grid">
        <div className="card">
          <h2>Loss</h2>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={points}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" /><YAxis /><Tooltip /><Legend />
              <Line type="monotone" dataKey="loss" stroke="#2563eb" dot={false} name="학습 loss" connectNulls />
              <Line type="monotone" dataKey="eval_loss" stroke="#dc2626" dot={false} name="검증 loss" connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <h2>WER / CER</h2>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={points}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" /><YAxis /><Tooltip /><Legend />
              <Line type="monotone" dataKey="wer" stroke="#16a34a" dot name="WER" connectNulls />
              <Line type="monotone" dataKey="cer" stroke="#d97706" dot name="CER" connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
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

      <div className="card">
        <h2>로그</h2>
        <div className="log" ref={logRef}>{log || "(출력 대기 중…)"}</div>
      </div>
    </div>
  );
}

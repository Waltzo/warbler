import { useEffect, useMemo, useState } from "react";
import { api, GpuStat, InferModel, InferResult, InferTarget } from "../api";

export default function Test() {
  const [models, setModels] = useState<InferModel[]>([]);
  const [gpus, setGpus] = useState<GpuStat[]>([]);
  const [jobId, setJobId] = useState("");
  const [compareBase, setCompareBase] = useState(true);
  const [gpuIndex, setGpuIndex] = useState(0);
  const [language, setLanguage] = useState("ko");
  const [audio, setAudio] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [results, setResults] = useState<InferResult[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    api.inferModels().then((m) => { setModels(m); if (m[0]) setJobId(m[0].job_id); });
    api.gpus().then(setGpus);
    api.info().then((i) => setGpuIndex(i.default_gpu_index));
  }, []);

  const selected = useMemo(() => models.find((m) => m.job_id === jobId), [models, jobId]);

  const pickAudio = (f: File | null) => {
    setAudio(f);
    setResults([]);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(f ? URL.createObjectURL(f) : "");
  };

  const run = async () => {
    if (!audio || !selected) return;
    setErr(""); setBusy(true); setResults([]);
    try {
      const targets: InferTarget[] = [
        { kind: "finetuned", job_id: selected.job_id, label: `파인튜닝 (${selected.name})` },
      ];
      if (compareBase) {
        targets.push({
          kind: "base", model_type: selected.model_type,
          base_model: selected.base_model, label: `base (${selected.base_model})`,
        });
      }
      const r = await api.infer(audio, targets, gpuIndex, language);
      setResults(r.results);
    } catch (e: any) {
      setErr(e?.response?.data?.detail || String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h1>테스트 / 비교</h1>
      <div className="card">
        <h2>1. 모델 선택</h2>
        {models.length === 0 && <p className="muted">완료된 학습 모델 없음. 먼저 학습 끝내야 함.</p>}
        <label>파인튜닝 모델 (완료된 잡)</label>
        <select value={jobId} onChange={(e) => setJobId(e.target.value)}>
          {models.map((m) => (
            <option key={m.job_id} value={m.job_id}>
              {m.name} — {m.model_type}/{m.base_model}{m.lora ? " (LoRA)" : ""}
            </option>
          ))}
        </select>
        <label className="row" style={{ marginTop: 10 }}>
          <input type="checkbox" style={{ width: "auto" }} checked={compareBase}
            onChange={(e) => setCompareBase(e.target.checked)} />
          &nbsp;base 모델과 A/B 비교 {selected && <span className="muted">&nbsp;({selected.base_model})</span>}
        </label>
        <div className="grid" style={{ marginTop: 10 }}>
          <div>
            <label>GPU</label>
            <select value={gpuIndex} onChange={(e) => setGpuIndex(Number(e.target.value))}>
              {gpus.length === 0 && <option value={0}>GPU 0</option>}
              {gpus.map((g) => <option key={g.index} value={g.index}>GPU {g.index} — {g.name}</option>)}
            </select>
          </div>
          <div>
            <label>Language (whisper)</label>
            <input value={language} onChange={(e) => setLanguage(e.target.value)} />
          </div>
        </div>
      </div>

      <div className="card">
        <h2>2. 오디오</h2>
        <input type="file" accept="audio/*"
          onChange={(e) => pickAudio(e.target.files?.[0] || null)} />
        {audioUrl && <div style={{ marginTop: 10 }}><audio controls src={audioUrl} /></div>}
        <div style={{ marginTop: 12 }}>
          <button onClick={run} disabled={!audio || !selected || busy}>
            {busy ? "전사 중…" : "전사"}
          </button>
          {busy && <span className="muted"> (첫 호출은 모델 로드로 느릴 수 있음)</span>}
        </div>
        {err && <p style={{ color: "#dc2626" }}>{err}</p>}
      </div>

      {results.length > 0 && (
        <div className="card">
          <h2>3. 결과</h2>
          {results.map((r, i) => (
            <div key={i} style={{ marginBottom: 14 }}>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <strong>{r.label}</strong>
                {r.ms !== undefined && <span className="muted">{r.ms} ms</span>}
              </div>
              {r.error
                ? <p style={{ color: "#dc2626" }}>{r.error}</p>
                : <p style={{ fontSize: 16, background: "#f8fafc", padding: "10px 12px", borderRadius: 6 }}>{r.text || "(빈 결과)"}</p>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

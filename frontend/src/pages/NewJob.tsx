import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, DatasetInfo, GpuStat, TrainConfig } from "../api";

const DEFAULTS: TrainConfig = {
  name: "",
  model_type: "whisper",
  base_model: "openai/whisper-small",
  dataset_id: "",
  use_lora: false,
  lora_r: 16,
  lora_alpha: 32,
  lora_dropout: 0.05,
  learning_rate: 1e-5,
  batch_size: 8,
  grad_accum: 1,
  num_epochs: 3,
  max_steps: -1,
  eval_ratio: 0.1,
  eval_steps: 50,
  save_steps: 200,
  warmup_steps: 50,
  language: "korean",
  task: "transcribe",
  gpu_index: 0,
  precision: "fp16",
};

export default function NewJob() {
  const nav = useNavigate();
  const [cfg, setCfg] = useState<TrainConfig>(DEFAULTS);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [gpus, setGpus] = useState<GpuStat[]>([]);
  const [suggested, setSuggested] = useState<Record<string, string[]>>({});
  const [err, setErr] = useState("");

  useEffect(() => {
    api.datasets().then((d) => {
      setDatasets(d);
      if (d[0]) setCfg((c) => ({ ...c, dataset_id: d[0].dataset_id }));
    });
    api.gpus().then(setGpus);
    api.info().then((i) => {
      setSuggested(i.suggested_models);
      setCfg((c) => ({ ...c, gpu_index: i.default_gpu_index }));
    });
  }, []);

  const set = (k: keyof TrainConfig, v: any) => setCfg((c) => ({ ...c, [k]: v }));
  const num = (k: keyof TrainConfig) => (e: any) => set(k, Number(e.target.value));

  const submit = async () => {
    setErr("");
    try {
      const job = await api.createJob(cfg);
      nav(`/jobs/${job.job_id}`);
    } catch (e: any) {
      setErr(e?.response?.data?.detail || String(e));
    }
  };

  const modelOptions = suggested[cfg.model_type] || [];

  return (
    <div>
      <h1>새 학습</h1>
      <div className="card">
        <label>작업 이름</label>
        <input value={cfg.name} onChange={(e) => set("name", e.target.value)} />

        <div className="grid">
          <div>
            <label>모델 종류</label>
            <select value={cfg.model_type}
              onChange={(e) => set("model_type", e.target.value)}>
              <option value="whisper">whisper</option>
              <option value="wav2vec2">wav2vec2</option>
            </select>
          </div>
          <div>
            <label>베이스 모델</label>
            <input list="models" value={cfg.base_model}
              onChange={(e) => set("base_model", e.target.value)} />
            <datalist id="models">
              {modelOptions.map((m) => <option key={m} value={m} />)}
            </datalist>
          </div>
        </div>

        <div className="grid">
          <div>
            <label>데이터셋</label>
            <select value={cfg.dataset_id} onChange={(e) => set("dataset_id", e.target.value)}>
              {datasets.map((d) => (
                <option key={d.dataset_id} value={d.dataset_id}>
                  {d.dataset_id} ({d.num_samples})
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>GPU</label>
            <select value={cfg.gpu_index} onChange={num("gpu_index")}>
              {gpus.length === 0 && <option value={0}>GPU 0</option>}
              {gpus.map((g) => (
                <option key={g.index} value={g.index}>
                  GPU {g.index} — {g.name} ({Math.round(g.memory_used_mb)}/{Math.round(g.memory_total_mb)}MB)
                </option>
              ))}
            </select>
          </div>
        </div>

        {cfg.model_type === "whisper" && (
          <div className="grid">
            <div>
              <label>언어</label>
              <input value={cfg.language || ""} onChange={(e) => set("language", e.target.value)} />
            </div>
            <div>
              <label>작업</label>
              <select value={cfg.task} onChange={(e) => set("task", e.target.value)}>
                <option value="transcribe">transcribe</option>
                <option value="translate">translate</option>
              </select>
            </div>
          </div>
        )}
      </div>

      <div className="card">
        <h2>하이퍼파라미터</h2>
        <div className="grid">
          <div><label>학습률 (learning rate)</label><input type="number" step="any" value={cfg.learning_rate} onChange={num("learning_rate")} /></div>
          <div><label>배치 크기</label><input type="number" value={cfg.batch_size} onChange={num("batch_size")} /></div>
          <div><label>Grad 누적 (accumulation)</label><input type="number" value={cfg.grad_accum} onChange={num("grad_accum")} /></div>
          <div><label>에폭 (epochs)</label><input type="number" step="any" value={cfg.num_epochs} onChange={num("num_epochs")} /></div>
          <div><label>최대 스텝 (-1 = 에폭 사용)</label><input type="number" value={cfg.max_steps} onChange={num("max_steps")} /></div>
          <div><label>검증 비율 (eval ratio)</label><input type="number" step="any" value={cfg.eval_ratio} onChange={num("eval_ratio")} /></div>
          <div><label>검증 주기 (eval steps)</label><input type="number" value={cfg.eval_steps} onChange={num("eval_steps")} /></div>
          <div><label>저장 주기 (save steps)</label><input type="number" value={cfg.save_steps} onChange={num("save_steps")} /></div>
          <div><label>워밍업 스텝 (warmup)</label><input type="number" value={cfg.warmup_steps} onChange={num("warmup_steps")} /></div>
          <div>
            <label>정밀도 (precision)</label>
            <select value={cfg.precision} onChange={(e) => set("precision", e.target.value)}>
              <option value="fp16">fp16</option>
              <option value="bf16">bf16 (A100)</option>
              <option value="fp32">fp32</option>
            </select>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>
          <label className="row" style={{ margin: 0 }}>
            <input type="checkbox" style={{ width: "auto" }} checked={cfg.use_lora}
              onChange={(e) => set("use_lora", e.target.checked)} />
            &nbsp;LoRA 사용 (PEFT)
          </label>
        </h2>
        {cfg.use_lora && (
          <div className="grid">
            <div><label>LoRA r</label><input type="number" value={cfg.lora_r} onChange={num("lora_r")} /></div>
            <div><label>LoRA alpha</label><input type="number" value={cfg.lora_alpha} onChange={num("lora_alpha")} /></div>
            <div><label>LoRA dropout</label><input type="number" step="any" value={cfg.lora_dropout} onChange={num("lora_dropout")} /></div>
          </div>
        )}
      </div>

      <button onClick={submit} disabled={!cfg.name || !cfg.dataset_id}>학습 시작</button>
      {err && <p style={{ color: "#dc2626" }}>{err}</p>}
    </div>
  );
}

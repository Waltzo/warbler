import axios from "axios";

export const http = axios.create({ baseURL: "" });

export const STATE_KO: Record<string, string> = {
  pending: "대기",
  running: "실행중",
  done: "완료",
  failed: "실패",
  stopped: "중지됨",
};

export interface GpuStat {
  index: number;
  name: string;
  memory_used_mb: number;
  memory_total_mb: number;
  utilization_pct: number;
}

export interface DatasetInfo {
  dataset_id: string;
  manifest_path: string;
  audio_root: string;
  num_samples: number;
  preview: { audio: string; text: string }[];
}

export interface JobStatus {
  job_id: string;
  name: string;
  state: "pending" | "running" | "done" | "failed" | "stopped";
  model_type: string;
  base_model: string;
  dataset_id: string;
  pid?: number;
  created_at: string;
  started_at?: string;
  finished_at?: string;
  error?: string;
  last_step?: number;
  total_steps?: number;
  gpu_index?: number;
}

export interface TrainConfig {
  name: string;
  model_type: "whisper" | "wav2vec2";
  base_model: string;
  dataset_id: string;
  use_lora: boolean;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  learning_rate: number;
  batch_size: number;
  grad_accum: number;
  num_epochs: number;
  max_steps: number;
  eval_ratio: number;
  eval_steps: number;
  save_steps: number;
  warmup_steps: number;
  save_total_limit?: number | null;
  language?: string;
  task: string;
  gpu_index: number;
  precision: "fp16" | "bf16" | "fp32";
}

export interface Corpus {
  corpus_id: string;
  audio_root: string;
  num_files: number;
  segments: number;
  reviewed: number;
  model?: string;
}

export interface Segment {
  seg_id: string;
  clip: string;
  source_file: string;
  start: number;
  end: number;
  duration: number;
  draft_text: string;
  text: string;
  reviewed: boolean;
}

export const api = {
  gpus: () => http.get<GpuStat[]>("/system/gpus").then((r) => r.data),
  info: () => http.get<{ default_gpu_index: number; suggested_models: Record<string, string[]> }>("/system/info").then((r) => r.data),

  datasets: () => http.get<DatasetInfo[]>("/datasets").then((r) => r.data),
  datasetDetail: (id: string) => http.get<DatasetInfo>(`/datasets/${id}`).then((r) => r.data),
  registerDataset: (body: {
    dataset_id: string; manifest_path: string; audio_root?: string;
    audio_key?: string; text_key?: string;
  }) => http.post<DatasetInfo>("/datasets", body).then((r) => r.data),
  deleteDataset: (id: string) => http.delete(`/datasets/${id}`).then((r) => r.data),

  jobs: () => http.get<JobStatus[]>("/jobs").then((r) => r.data),
  job: (id: string) => http.get<JobStatus>(`/jobs/${id}`).then((r) => r.data),
  metrics: (id: string) => http.get<any[]>(`/jobs/${id}/metrics`).then((r) => r.data),
  config: (id: string) => http.get<TrainConfig>(`/jobs/${id}/config`).then((r) => r.data),
  createJob: (cfg: TrainConfig) => http.post<JobStatus>("/jobs", cfg).then((r) => r.data),
  stopJob: (id: string) => http.post<JobStatus>(`/jobs/${id}/stop`).then((r) => r.data),

  corpora: () => http.get<Corpus[]>("/corpus").then((r) => r.data),
  corpus: (id: string) => http.get<Corpus>(`/corpus/${id}`).then((r) => r.data),
  createCorpus: (body: { corpus_id: string; audio_root: string }) =>
    http.post<Corpus>("/corpus", body).then((r) => r.data),
  transcribe: (id: string, body: { model: string; language: string; gpu_index: number }) =>
    http.post<JobStatus>(`/corpus/${id}/transcribe`, body).then((r) => r.data),
  segments: (id: string, onlyUnreviewed: boolean, limit = 200, offset = 0) =>
    http.get<{ total: number; items: Segment[] }>(`/corpus/${id}/segments`, {
      params: { only_unreviewed: onlyUnreviewed, limit, offset },
    }).then((r) => r.data),
  segmentAudioUrl: (id: string, segId: string, v?: number) =>
    `/corpus/${id}/segments/${segId}/audio${v ? `?v=${v}` : ""}`,
  patchSegment: (id: string, segId: string,
    body: { text?: string; reviewed?: boolean; start?: number; end?: number }) =>
    http.patch<Segment>(`/corpus/${id}/segments/${segId}`, body).then((r) => r.data),
  exportCorpus: (id: string, body: { dataset_id: string; only_reviewed: boolean }) =>
    http.post<DatasetInfo>(`/corpus/${id}/export`, body).then((r) => r.data),

  inferModels: () => http.get<InferModel[]>("/infer/models").then((r) => r.data),
  infer: (audio: File, targets: InferTarget[], gpu_index: number, language: string) => {
    const fd = new FormData();
    fd.append("audio", audio);
    fd.append("targets", JSON.stringify(targets));
    fd.append("gpu_index", String(gpu_index));
    fd.append("language", language);
    return http.post<{ results: InferResult[] }>("/infer", fd).then((r) => r.data);
  },
};

export interface InferModel {
  job_id: string;
  name: string;
  model_type: "whisper" | "wav2vec2";
  base_model: string;
  lora: boolean;
}
export interface InferTarget {
  kind: "finetuned" | "base";
  label?: string;
  job_id?: string;
  model_type?: string;
  base_model?: string;
}
export interface InferResult {
  label: string;
  text?: string;
  ms?: number;
  error?: string;
}

/** Subscribe to a job's SSE stream. Returns an unsubscribe fn. */
export function subscribeJob(
  jobId: string,
  handlers: {
    onMetric?: (m: any) => void;
    onLog?: (text: string) => void;
    onStatus?: (s: JobStatus) => void;
  }
): () => void {
  const es = new EventSource(`/jobs/${jobId}/stream`);
  es.addEventListener("metric", (e) => handlers.onMetric?.(JSON.parse((e as MessageEvent).data)));
  es.addEventListener("log", (e) => handlers.onLog?.(JSON.parse((e as MessageEvent).data).text));
  es.addEventListener("status", (e) => {
    const s = JSON.parse((e as MessageEvent).data) as JobStatus;
    handlers.onStatus?.(s);
    if (["done", "failed", "stopped"].includes(s.state)) es.close();
  });
  es.onerror = () => es.close();
  return () => es.close();
}

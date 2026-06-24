import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, Corpus, GpuStat, Segment } from "../api";

export default function CorpusReview() {
  const { id } = useParams();
  const cid = id!;
  const [corpus, setCorpus] = useState<Corpus | null>(null);
  const [segs, setSegs] = useState<Segment[]>([]);
  const [total, setTotal] = useState(0);
  const [onlyUnreviewed, setOnlyUnreviewed] = useState(false);
  const [gpus, setGpus] = useState<GpuStat[]>([]);
  const [tcfg, setTcfg] = useState({ model: "large-v3", language: "ko", gpu_index: 0 });
  const [jobId, setJobId] = useState<string | null>(null);
  const [exportId, setExportId] = useState("");
  const [msg, setMsg] = useState("");

  const loadCorpus = () => api.corpus(cid).then(setCorpus);
  const loadSegs = () =>
    api.segments(cid, onlyUnreviewed).then((r) => { setSegs(r.items); setTotal(r.total); });

  useEffect(() => {
    loadCorpus();
    api.gpus().then((g) => { setGpus(g); if (g[0]) setTcfg((c) => ({ ...c, gpu_index: g[0].index })); });
  }, [cid]);
  useEffect(() => { loadSegs(); }, [cid, onlyUnreviewed]);

  // While a transcribe job runs, poll counts.
  useEffect(() => {
    if (!jobId) return;
    const t = setInterval(() => { loadCorpus(); loadSegs(); }, 3000);
    return () => clearInterval(t);
  }, [jobId]);

  const startTranscribe = async () => {
    setMsg("");
    try {
      const job = await api.transcribe(cid, tcfg);
      setJobId(job.job_id);
      setMsg(`전사 잡 시작: ${job.job_id}`);
    } catch (e: any) {
      setMsg(e?.response?.data?.detail || String(e));
    }
  };

  const save = async (s: Segment, patch: { text?: string; reviewed?: boolean }) => {
    const updated = await api.patchSegment(cid, s.seg_id, patch);
    setSegs((cur) => cur.map((x) => (x.seg_id === s.seg_id ? updated : x)));
    loadCorpus();
  };

  const doExport = async () => {
    setMsg("");
    try {
      const ds = await api.exportCorpus(cid, { dataset_id: exportId, only_reviewed: true });
      setMsg(`데이터셋 등록 완료: ${ds.dataset_id} (${ds.num_samples} samples) — New Training서 선택 가능`);
    } catch (e: any) {
      setMsg(e?.response?.data?.detail || String(e));
    }
  };

  if (!corpus) return <p>Loading…</p>;

  return (
    <div>
      <h1>Corpus: {corpus.corpus_id}</h1>
      <div className="card">
        <div className="muted">
          files {corpus.num_files} · segments {corpus.segments} · reviewed {corpus.reviewed}/{corpus.segments}
          {jobId && <> · <Link className="joblink" to={`/jobs/${jobId}`}>전사 진행 보기 →</Link></>}
        </div>
      </div>

      <div className="card">
        <h2>1. 자동 분할 + 초벌전사 (faster-whisper)</h2>
        <div className="grid">
          <div>
            <label>Model</label>
            <select value={tcfg.model} onChange={(e) => setTcfg({ ...tcfg, model: e.target.value })}>
              <option value="large-v3">large-v3</option>
              <option value="medium">medium</option>
              <option value="small">small</option>
            </select>
          </div>
          <div>
            <label>Language</label>
            <input value={tcfg.language} onChange={(e) => setTcfg({ ...tcfg, language: e.target.value })} />
          </div>
          <div>
            <label>GPU</label>
            <select value={tcfg.gpu_index}
              onChange={(e) => setTcfg({ ...tcfg, gpu_index: Number(e.target.value) })}>
              {gpus.length === 0 && <option value={0}>GPU 0</option>}
              {gpus.map((g) => (
                <option key={g.index} value={g.index}>GPU {g.index} — {g.name}</option>
              ))}
            </select>
          </div>
        </div>
        <div style={{ marginTop: 12 }}>
          <button onClick={startTranscribe}>Start transcription</button>
          <span className="muted"> (기존 segments 있으면 덮어씀)</span>
        </div>
      </div>

      <div className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2 style={{ margin: 0 }}>2. 교정 ({total} segments)</h2>
          <label className="row" style={{ margin: 0 }}>
            <input type="checkbox" style={{ width: "auto" }} checked={onlyUnreviewed}
              onChange={(e) => setOnlyUnreviewed(e.target.checked)} />
            &nbsp;미검토만
          </label>
        </div>
        <table>
          <thead><tr><th style={{ width: 220 }}>audio</th><th>text</th><th style={{ width: 90 }}>검토</th></tr></thead>
          <tbody>
            {segs.map((s) => (
              <SegmentRow key={s.seg_id} cid={cid} seg={s} onSave={save} />
            ))}
            {segs.length === 0 && (
              <tr><td colSpan={3} className="muted">아직 segment 없음. 위에서 전사 먼저 실행.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2>3. Export → 데이터셋 등록</h2>
        <p className="muted">검토완료(reviewed) segment만 manifest로 내보내 학습용 데이터셋으로 등록.</p>
        <label>Dataset ID</label>
        <input value={exportId} onChange={(e) => setExportId(e.target.value)} />
        <div style={{ marginTop: 12 }}>
          <button onClick={doExport} disabled={!exportId}>Export & Register</button>
        </div>
      </div>

      {msg && <p style={{ color: "#2563eb" }}>{msg}</p>}
    </div>
  );
}

function SegmentRow({ cid, seg, onSave }: {
  cid: string; seg: Segment;
  onSave: (s: Segment, patch: { text?: string; reviewed?: boolean }) => void;
}) {
  const [text, setText] = useState(seg.text);
  const dirty = text !== seg.text;
  return (
    <tr style={{ background: seg.reviewed ? "#f0fdf4" : undefined }}>
      <td>
        <audio controls preload="none" style={{ width: 200 }}
          src={api.segmentAudioUrl(cid, seg.seg_id)} />
        <div className="muted">{seg.duration}s</div>
      </td>
      <td>
        <input value={text} onChange={(e) => setText(e.target.value)}
          onBlur={() => dirty && onSave(seg, { text })} />
        {seg.draft_text !== text && (
          <div className="muted">초벌: {seg.draft_text}</div>
        )}
      </td>
      <td>
        <input type="checkbox" style={{ width: "auto" }} checked={seg.reviewed}
          onChange={(e) => onSave(seg, { text, reviewed: e.target.checked })} />
      </td>
    </tr>
  );
}

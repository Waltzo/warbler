import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, Corpus } from "../api";

export default function Prepare() {
  const [list, setList] = useState<Corpus[]>([]);
  const [form, setForm] = useState({ corpus_id: "", audio_root: "" });
  const [err, setErr] = useState("");

  const reload = () => api.corpora().then(setList);
  useEffect(() => {
    reload();
    const t = setInterval(reload, 4000);
    return () => clearInterval(t);
  }, []);

  const create = async () => {
    setErr("");
    try {
      await api.createCorpus(form);
      setForm({ corpus_id: "", audio_root: "" });
      reload();
    } catch (e: any) {
      setErr(e?.response?.data?.detail || String(e));
    }
  };

  return (
    <div>
      <h1>데이터 준비</h1>
      <div className="card">
        <h2>새 코퍼스 (오디오 폴더 → 라벨링 프로젝트)</h2>
        <p className="muted">
          오디오 폴더 경로 등록 → 전사 잡 실행 → 교정 → Export 하면 학습용 데이터셋으로 등록됨.
        </p>
        <label>Corpus ID</label>
        <input value={form.corpus_id}
          onChange={(e) => setForm({ ...form, corpus_id: e.target.value })} />
        <label>Audio root (서버 경로, 하위 폴더까지 스캔)</label>
        <input value={form.audio_root}
          onChange={(e) => setForm({ ...form, audio_root: e.target.value })}
          placeholder="/data/raw_audio" />
        <div style={{ marginTop: 12 }}>
          <button onClick={create} disabled={!form.corpus_id || !form.audio_root}>
            생성
          </button>
        </div>
        {err && <p style={{ color: "#dc2626" }}>{err}</p>}
      </div>

      <div className="card">
        <h2>코퍼스 목록</h2>
        <table>
          <thead><tr><th>ID</th><th>파일</th><th>세그먼트</th><th>검토완료</th><th></th></tr></thead>
          <tbody>
            {list.map((c) => (
              <tr key={c.corpus_id}>
                <td><Link className="joblink" to={`/prepare/${c.corpus_id}`}>{c.corpus_id}</Link></td>
                <td>{c.num_files}</td>
                <td>{c.segments}</td>
                <td>{c.reviewed} / {c.segments}</td>
                <td><Link className="joblink" to={`/prepare/${c.corpus_id}`}>열기 →</Link></td>
              </tr>
            ))}
            {list.length === 0 && (
              <tr><td colSpan={5} className="muted">코퍼스 없음.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

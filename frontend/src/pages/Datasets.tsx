import { useEffect, useState } from "react";
import { api, DatasetInfo } from "../api";

export default function Datasets() {
  const [list, setList] = useState<DatasetInfo[]>([]);
  const [form, setForm] = useState({ dataset_id: "", manifest_path: "", audio_root: "",
    audio_key: "", text_key: "" });
  const [preview, setPreview] = useState<DatasetInfo | null>(null);
  const [err, setErr] = useState("");

  const reload = () => api.datasets().then(setList);
  useEffect(() => { reload(); }, []);

  const submit = async () => {
    setErr("");
    try {
      const info = await api.registerDataset({
        dataset_id: form.dataset_id,
        manifest_path: form.manifest_path,
        audio_root: form.audio_root || undefined,
        audio_key: form.audio_key || undefined,
        text_key: form.text_key || undefined,
      });
      setPreview(info);
      reload();
    } catch (e: any) {
      setErr(e?.response?.data?.detail || String(e));
    }
  };

  return (
    <div>
      <h1>데이터셋</h1>
      <div className="card">
        <h2>데이터셋 등록</h2>
        <p className="muted">
          서버 경로의 manifest(.jsonl/.csv)를 등록. 각 행: <code>audio_path</code>, <code>text</code>.
        </p>
        <label>데이터셋 ID</label>
        <input value={form.dataset_id}
          onChange={(e) => setForm({ ...form, dataset_id: e.target.value })} />
        <label>Manifest 경로 (.jsonl / .csv)</label>
        <input value={form.manifest_path}
          onChange={(e) => setForm({ ...form, manifest_path: e.target.value })}
          placeholder="/data/my_set/manifest.jsonl" />
        <label>오디오 루트 (선택, 미지정 시 manifest 폴더)</label>
        <input value={form.audio_root}
          onChange={(e) => setForm({ ...form, audio_root: e.target.value })} />
        <div className="grid">
          <div>
            <label>오디오 키 (선택, 기본 audio_path/audio/path)</label>
            <input value={form.audio_key} placeholder="예: filepath"
              onChange={(e) => setForm({ ...form, audio_key: e.target.value })} />
          </div>
          <div>
            <label>텍스트 키 (선택, 기본 text/transcript/sentence)</label>
            <input value={form.text_key} placeholder="예: label"
              onChange={(e) => setForm({ ...form, text_key: e.target.value })} />
          </div>
        </div>
        <div style={{ marginTop: 12 }}>
          <button onClick={submit}>등록</button>
        </div>
        {err && <p style={{ color: "#dc2626" }}>{err}</p>}
      </div>

      {preview && (
        <div className="card">
          <h2>미리보기: {preview.dataset_id} (샘플 {preview.num_samples}개)</h2>
          <table>
            <thead><tr><th>오디오</th><th>텍스트</th></tr></thead>
            <tbody>
              {preview.preview.map((r, i) => (
                <tr key={i}><td>{r.audio}</td><td>{r.text}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="card">
        <h2>등록된 데이터셋</h2>
        <table>
          <thead><tr><th>ID</th><th>샘플 수</th><th>manifest</th><th></th></tr></thead>
          <tbody>
            {list.map((d) => (
              <tr key={d.dataset_id}>
                <td>{d.dataset_id}</td>
                <td>{d.num_samples}</td>
                <td className="muted">{d.manifest_path}</td>
                <td>
                  <button className="danger"
                    onClick={() => api.deleteDataset(d.dataset_id).then(reload)}>
                    삭제
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

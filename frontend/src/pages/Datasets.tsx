import { useEffect, useState } from "react";
import { api, DatasetInfo } from "../api";

export default function Datasets() {
  const [list, setList] = useState<DatasetInfo[]>([]);
  const [form, setForm] = useState({ dataset_id: "", manifest_path: "", audio_root: "" });
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
      });
      setPreview(info);
      reload();
    } catch (e: any) {
      setErr(e?.response?.data?.detail || String(e));
    }
  };

  return (
    <div>
      <h1>Datasets</h1>
      <div className="card">
        <h2>Register dataset</h2>
        <p className="muted">
          서버 경로의 manifest(.jsonl/.csv)를 등록. 각 행: <code>audio_path</code>, <code>text</code>.
        </p>
        <label>Dataset ID</label>
        <input value={form.dataset_id}
          onChange={(e) => setForm({ ...form, dataset_id: e.target.value })} />
        <label>Manifest path (.jsonl / .csv)</label>
        <input value={form.manifest_path}
          onChange={(e) => setForm({ ...form, manifest_path: e.target.value })}
          placeholder="/data/my_set/manifest.jsonl" />
        <label>Audio root (optional, defaults to manifest dir)</label>
        <input value={form.audio_root}
          onChange={(e) => setForm({ ...form, audio_root: e.target.value })} />
        <div style={{ marginTop: 12 }}>
          <button onClick={submit}>Register</button>
        </div>
        {err && <p style={{ color: "#dc2626" }}>{err}</p>}
      </div>

      {preview && (
        <div className="card">
          <h2>Preview: {preview.dataset_id} ({preview.num_samples} samples)</h2>
          <table>
            <thead><tr><th>audio</th><th>text</th></tr></thead>
            <tbody>
              {preview.preview.map((r, i) => (
                <tr key={i}><td>{r.audio}</td><td>{r.text}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="card">
        <h2>Registered</h2>
        <table>
          <thead><tr><th>ID</th><th>samples</th><th>manifest</th><th></th></tr></thead>
          <tbody>
            {list.map((d) => (
              <tr key={d.dataset_id}>
                <td>{d.dataset_id}</td>
                <td>{d.num_samples}</td>
                <td className="muted">{d.manifest_path}</td>
                <td>
                  <button className="danger"
                    onClick={() => api.deleteDataset(d.dataset_id).then(reload)}>
                    Delete
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

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, JobStatus, STATE_KO } from "../api";

export default function Jobs() {
  const [jobs, setJobs] = useState<JobStatus[]>([]);

  useEffect(() => {
    const load = () => api.jobs().then(setJobs);
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, []);

  return (
    <div>
      <h1>학습 작업</h1>
      <div className="card">
        <table>
          <thead>
            <tr><th>이름</th><th>상태</th><th>모델</th><th>데이터셋</th>
              <th>GPU</th><th>진행</th><th>생성일</th></tr>
          </thead>
          <tbody>
            {jobs.map((j) => (
              <tr key={j.job_id}>
                <td><Link className="joblink" to={`/jobs/${j.job_id}`}>{j.name}</Link></td>
                <td><span className={`badge ${j.state}`}>{STATE_KO[j.state] || j.state}</span></td>
                <td>{j.model_type} <span className="muted">{j.base_model}</span></td>
                <td>{j.dataset_id}</td>
                <td>{j.gpu_index ?? "-"}</td>
                <td>{j.last_step ?? 0}{j.total_steps ? ` / ${j.total_steps}` : ""}</td>
                <td className="muted">{j.created_at}</td>
              </tr>
            ))}
            {jobs.length === 0 && (
              <tr><td colSpan={7} className="muted">학습 작업 없음. "새 학습"에서 시작.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, JobStatus } from "../api";

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
      <h1>Training Jobs</h1>
      <div className="card">
        <table>
          <thead>
            <tr><th>Name</th><th>State</th><th>Model</th><th>Dataset</th>
              <th>GPU</th><th>Progress</th><th>Created</th></tr>
          </thead>
          <tbody>
            {jobs.map((j) => (
              <tr key={j.job_id}>
                <td><Link className="joblink" to={`/jobs/${j.job_id}`}>{j.name}</Link></td>
                <td><span className={`badge ${j.state}`}>{j.state}</span></td>
                <td>{j.model_type}<br /><span className="muted">{j.base_model}</span></td>
                <td>{j.dataset_id}</td>
                <td>{j.gpu_index ?? "-"}</td>
                <td>{j.last_step ?? 0}{j.total_steps ? ` / ${j.total_steps}` : ""}</td>
                <td className="muted">{j.created_at}</td>
              </tr>
            ))}
            {jobs.length === 0 && (
              <tr><td colSpan={7} className="muted">No jobs yet. Start one in “New Training”.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

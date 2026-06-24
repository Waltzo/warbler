import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import Datasets from "./pages/Datasets";
import Jobs from "./pages/Jobs";
import NewJob from "./pages/NewJob";
import JobDetail from "./pages/JobDetail";
import "./styles.css";

function App() {
  return (
    <BrowserRouter>
      <nav className="nav">
        <span className="brand">🎙️ STT Tuner</span>
        <Link to="/">Jobs</Link>
        <Link to="/new">New Training</Link>
        <Link to="/datasets">Datasets</Link>
      </nav>
      <main className="container">
        <Routes>
          <Route path="/" element={<Jobs />} />
          <Route path="/new" element={<NewJob />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/jobs/:id" element={<JobDetail />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

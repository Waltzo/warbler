import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import Datasets from "./pages/Datasets";
import Jobs from "./pages/Jobs";
import NewJob from "./pages/NewJob";
import JobDetail from "./pages/JobDetail";
import Prepare from "./pages/Prepare";
import CorpusReview from "./pages/CorpusReview";
import Test from "./pages/Test";
import "./styles.css";

function App() {
  return (
    <BrowserRouter>
      <nav className="nav">
        <span className="brand">🎙️ STT 튜너</span>
        <Link to="/">학습 작업</Link>
        <Link to="/new">새 학습</Link>
        <Link to="/prepare">데이터 준비</Link>
        <Link to="/test">테스트</Link>
        <Link to="/datasets">데이터셋</Link>
      </nav>
      <main className="container">
        <Routes>
          <Route path="/" element={<Jobs />} />
          <Route path="/new" element={<NewJob />} />
          <Route path="/prepare" element={<Prepare />} />
          <Route path="/prepare/:id" element={<CorpusReview />} />
          <Route path="/test" element={<Test />} />
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

"""FastAPI app entry. Run: uvicorn app.main:app --host 0.0.0.0 --port 8000"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import corpus, datasets, infer, jobs, system

app = FastAPI(title="STT Tuner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # single-user internal tool
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(jobs.router)
app.include_router(system.router)
app.include_router(corpus.router)
app.include_router(infer.router)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


if __name__ == "__main__":
    # `python -m app.main` — uses host/port from config.toml (env overrides).
    import uvicorn

    from . import config
    uvicorn.run("app.main:app", host=config.SERVER_HOST, port=config.SERVER_PORT)

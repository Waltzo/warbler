# STT Tuner

웹 기반 STT(음성인식) 모델 finetune 툴. Whisper / wav2vec2 지원, full finetune + LoRA, 실시간 진행 모니터링(loss/WER/CER, 로그, GPU). 단일사용자/내부용.

```
브라우저(React) ──HTTP/SSE──> FastAPI ──subprocess(CUDA_VISIBLE_DEVICES=N)──> training/train.py
                                  │                                              │
                                  └──── runs/<job_id>/ <── metrics.jsonl/log ────┘
```

## 설치

### Backend + 학습 코어
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
# A100 CUDA에 맞는 torch 먼저 설치 (예: cu121)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```

## 실행

터미널 1 — 백엔드:
```bash
cd backend
python -m app.main          # config.toml의 host/port 사용
# 또는: uvicorn app.main:app --host 0.0.0.0 --port 8000
```

터미널 2 — 프론트(개발):
```bash
cd frontend
npm run dev   # http://localhost:5173 , /jobs /datasets /system 은 :8000으로 프록시
```

배포 시: `npm run build` 후 `frontend/dist`를 정적 서빙(nginx 등)하거나 FastAPI `StaticFiles`로 마운트.

## GPU 지정

학습 잡은 항상 **별도 subprocess**로 뜨고, 백엔드가 `CUDA_VISIBLE_DEVICES=<gpu_index>`를 주입.
- UI **New Training** 폼의 **GPU** 드롭다운에서 선택 (nvidia-smi 목록).
- 기본값은 환경변수 `STT_DEFAULT_GPU` (기본 0).
- 동시 1개 잡만 허용 — 실행 중이면 새 잡 생성은 409.

4-GPU A100에서 GPU 2번만 쓰려면 폼에서 GPU 2 선택. 프로세스 안에서는 그 GPU가 `cuda:0`으로 보임.

## 데이터셋 형식

manifest = `.jsonl` 또는 `.csv`, 각 행:
```json
{"audio_path": "clips/0001.wav", "text": "안녕하세요"}
```
- `audio_path`: 절대경로 또는 `audio_root` 기준 상대경로 (별칭 `audio`/`path`).
- `text`: 정답 전사 (별칭 `transcript`/`sentence`).
- 오디오는 자동 16kHz 리샘플(wav/mp3/flac, librosa/soundfile).

**Datasets** 페이지에서 manifest 경로 등록 → 검증 + 미리보기.

## 산출물 (runs/<job_id>/)

| 파일 | 내용 |
|------|------|
| `config.json` | 학습 설정 |
| `status.json` | 잡 상태(pending/running/done/failed/stopped, pid, step) |
| `metrics.jsonl` | step별 loss/lr, eval별 loss/WER/CER (UI 차트 소스) |
| `train.log` | subprocess stdout/stderr |
| `checkpoints/` | 중간 체크포인트 |
| `model/` | 최종 모델(+processor). LoRA면 adapter |

## 설정 (config.toml)

루트의 `config.toml`에서 port / 기본 GPU 지정:
```toml
[server]
host = "0.0.0.0"
port = 8000

[gpu]
default_index = 0   # UI New Training 기본 GPU
```

우선순위: **환경변수 > config.toml > 기본값**. config 경로는 `STT_CONFIG`로 변경.

## 환경변수

| 변수 | 기본 | 설명 |
|------|------|------|
| `STT_CONFIG` | `<root>/config.toml` | 설정 파일 경로 |
| `STT_HOST` | config.toml / `0.0.0.0` | 서버 bind host |
| `STT_PORT` | config.toml / `8000` | 서버 port |
| `STT_DEFAULT_GPU` | config.toml / `0` | UI 기본 GPU |
| `STT_ROOT` | repo root | datasets/, runs/ 기준 경로 |
| `STT_DATASETS_DIR` | `<root>/datasets` | 데이터셋 레지스트리 위치 |
| `STT_RUNS_DIR` | `<root>/runs` | 잡 산출물 위치 |
| `STT_PYTHON` | 현재 인터프리터 | train subprocess 실행 파이썬 |

## 학습 코어 단독 실행 (디버그)

```bash
cd backend
CUDA_VISIBLE_DEVICES=0 python -m training.train --config runs/<job_id>/config.json
```

## 주의

- 인증 없음 — 신뢰된 내부 네트워크에서만 사용.
- 큰 모델 full finetune은 메모리 큼 → A100에서 `bf16` + 필요시 LoRA 권장.

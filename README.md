# STT Tuner

웹 기반 STT(음성인식) 모델 finetune 툴. Whisper / wav2vec2 지원, full finetune + LoRA, 실시간 진행 모니터링(loss/WER/CER, 로그, GPU). 오디오만 있을 때 자동 분할+초벌전사+교정으로 데이터셋 준비(faster-whisper). 단일사용자/내부용.

```
브라우저(React) ──HTTP/SSE──> FastAPI ──subprocess(CUDA_VISIBLE_DEVICES=N)──> training/train.py
                                  │                                              │
                                  └──── runs/<job_id>/ <── metrics.jsonl/log ────┘
```

## 설치

### Backend + 학습 코어
```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
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

## 데이터 준비 (오디오만 있을 때)

전사(text)가 없고 오디오만 있으면 **Prepare Data** 페이지서 manifest까지 만든다:

1. **Corpus 생성** — 오디오 폴더(서버 경로) 등록. 하위까지 스캔(wav/mp3/flac/m4a/ogg/opus).
2. **자동 분할 + 초벌전사** — faster-whisper(기본 large-v3)가 긴 파일을 VAD로 발화단위 분할 + 구간별 초벌 텍스트 생성. GPU 잡(학습과 동일하게 GPU 지정, 동시 1개 락). 결과: `clips/*.wav`(16k mono) + `segments.jsonl`.
3. **교정** — 각 segment 오디오 듣고 텍스트 수정 → "검토완료" 체크. "미검토만" 필터로 진행.
4. **Export** — reviewed segment만 `manifest.jsonl`로 내보내 데이터셋 등록 → New Training서 바로 선택.

저장: `datasets/<corpus_id>/{meta.json, clips/, segments.jsonl, manifest.jsonl}`.

전사 코어 단독 실행:
```bash
cd backend
CUDA_VISIBLE_DEVICES=2 python -m prep.transcribe \
  --corpus-dir datasets/<id> --run-dir runs/<job_id> --model large-v3 --language ko
```

## 파인튜닝 확인 (Test 페이지)

학습이 잘 됐는지 직접 확인:

1. **모델 선택** — 완료된 학습 잡 드롭다운(저장된 `runs/<id>/model/`). LoRA면 base+adapter 자동 합쳐 로드.
2. **A/B 비교** 체크 → base 모델과 동시 전사.
3. **오디오 업로드** → Transcribe → 파인튜닝 vs base 전사 결과 나란히 + 소요 시간.

모델은 백엔드에 캐시(최대 3개) → 반복/비교 빠름. GPU 지정 가능(`cuda:<index>` 직접). 첫 호출만 로드로 느림.

학습 중 정량 지표(WER/CER)는 JobDetail 차트서 확인 — eval셋 기준 수치가 내려가면 학습된 것.

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
port = 8000        # 백엔드 port

[gpu]
default_index = 0  # UI New Training 기본 GPU

[frontend]
port = 5173        # Vite dev 서버 port (server.port로 API 프록시)
```

우선순위: **환경변수 > config.toml > 기본값**. 백엔드 config 경로는 `STT_CONFIG`로 변경.
프론트(vite)는 빌드 시 `../config.toml`에서 `frontend.port`(자기 port)와 `server.port`(프록시 대상)를 읽음.

## 환경변수

| 변수 | 기본 | 설명 |
|------|------|------|
| `STT_CONFIG` | `<root>/config.toml` | 설정 파일 경로 |
| `STT_HOST` | config.toml / `0.0.0.0` | 서버 bind host |
| `STT_PORT` | config.toml / `8000` | 백엔드 port (vite 프록시 대상도 겸함) |
| `STT_FRONTEND_PORT` | config.toml / `5173` | vite dev 서버 port |
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

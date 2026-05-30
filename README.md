# IKG

IKG는 개인 북마크와 웹 문서를 수집하고, 임베딩 기반 벡터 검색과
BM25 기반 키워드 검색을 결합해 지식 그래프/검색 경험을 만드는
프로토타입 프로젝트입니다.

현재 저장소는 기존 개별 프로젝트를 하나의 monorepo로 통합하는
중간 단계입니다. 루트 기준 설치, 실행, 테스트가 가능한 최소 개발
환경을 만드는 것을 우선 목표로 합니다.

## 현재 상태

아직 개발 중인 초안 상태입니다.

- Python 패키지 구조는 `src` layout으로 정리되었습니다.
- Python workspace는 루트 `uv` workspace를 사용하는 방향으로 정했습니다.
- Core와 backend 의존성은 각 workspace의 `pyproject.toml`로 분리했습니다.
- `uv.lock`은 아직 새 workspace 기준으로 갱신되지 않았습니다.
- FE build/lint 환경과 루트 공통 개발 명령은 아직 정리 중입니다.

작업 목록과 우선순위는 [task.md](task.md)를 기준으로 관리합니다.

## Workspace 구조

```text
.
├── apps/
│   ├── graphview/          # 브라우저 확장 / 새 탭 시작 페이지
│   └── async_worker/       # FastAPI API 및 Celery 비동기 워커
├── packages/
│   └── ai_core/            # 임베딩, 인덱싱, 하이브리드 검색 코어
├── db/                     # 로컬 SQLite / FAISS index 파일 위치
├── pyproject.toml          # 루트 uv workspace 설정
├── task.md                 # monorepo 통합 작업 목록
└── uv.lock                 # 기존 lockfile, 갱신 필요
```

## Python Workspace

Python 버전은 `3.13`을 기준으로 맞춥니다.

```text
packages/ai_core
├── pyproject.toml
├── src/ai_core
│   ├── embedder.py
│   ├── indexer.py
│   └── hybrid_search/
├── scripts/
└── test/
```

`ai_core`는 패키지 import 이름입니다.

```python
from ai_core.embedder import BGEEmbedder
from ai_core.indexer import Indexer
```

`apps/async_worker`는 `ai_core`를 workspace dependency로 사용합니다.

## 의존성 설치

의도한 설치 방식은 루트에서 `uv`를 사용하는 것입니다.

```bash
uv sync
```

다만 현재 `uv.lock`은 새 workspace 구조 기준으로 갱신되지 않았습니다.
새 환경에서는 먼저 lockfile 갱신이 필요할 수 있습니다.

```bash
uv lock
uv sync
```

## Core 수동 실행

아직 수동 실행 스크립트와 자동화 테스트가 완전히 분리되지 않았습니다.
현재는 `PYTHONPATH`로 `src` 경로를 열고 기존 entry point를 실행할 수
있습니다.

```bash
PYTHONPATH=packages/ai_core/src python3 packages/ai_core/scripts/v3_test.py
PYTHONPATH=packages/ai_core/src python3 packages/ai_core/test/hybrid_search_test.py
PYTHONPATH=packages/ai_core/src python3 packages/ai_core/test/vector_search_test.py
PYTHONPATH=packages/ai_core/src python3 packages/ai_core/test/bm25_search_test.py
```

workspace 설치가 완료된 뒤에는 다음처럼 실행하는 것을 목표로 합니다.

```bash
uv run python packages/ai_core/scripts/v3_test.py
```

## Backend 실행

Backend 앱은 `apps/async_worker`에 있습니다.

FastAPI 실행 예시는 다음과 같습니다.

```bash
uv run uvicorn async_worker.main:app --reload
```

Celery worker 실행 예시는 다음과 같습니다.

```bash
uv run celery -A async_worker.tasks worker --loglevel=info
```

현재 backend는 Redis를 broker/backend로 사용합니다.

```bash
REDIS_URL=redis://localhost:6379/0
```

## Frontend 실행

Frontend 앱은 `apps/graphview`에 있습니다.

```bash
cd apps/graphview
npm install
npm run dev
```

현재 extension build script에는 OS 의존적인 명령이 남아 있어 정리가
필요합니다.

## 로컬 데이터 경로

현재 코드에서 주로 사용하는 기본 경로는 다음과 같습니다.

```text
db/ikg_metadata.db
db/ikg_vector.index
model/bge-m3-onnx-int8
```

DB, FAISS index, 모델 파일 경로는 아직 환경 변수 기반 설정으로 완전히
정리되지 않았습니다.

## 검증 명령

현재 확인한 최소 검증 명령은 다음과 같습니다.

```bash
python3 -m compileall -q packages/ai_core/src apps/async_worker/src packages/ai_core/test packages/ai_core/scripts
.venv/bin/ruff check packages/ai_core/src apps/async_worker/src packages/ai_core/test packages/ai_core/scripts
PYTHONPATH=packages/ai_core/src .venv/bin/python -c "import ai_core; print(ai_core.__all__)"
```

목표 검증 명령은 아직 정리 중입니다.

```bash
uv run pytest
uv run ruff check
npm run build
npm run lint
```

## 다음 작업

- `uv.lock`을 새 workspace 구조 기준으로 갱신한다.
- 루트에서 clean install이 가능한지 검증한다.
- `input()` 기반 수동 실행 파일과 pytest 테스트를 분리한다.
- Core public API 범위를 확정한다.
- FE build/lint 환경을 정리한다.
- 루트 공통 개발 명령을 추가한다.

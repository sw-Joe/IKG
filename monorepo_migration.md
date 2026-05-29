# 모노레포 마이그레이션 리뷰

## 범위

`IKG/`는 아래 3개 프로젝트를 하나의 모노레포로 통합하는 디렉터리이다.

- `packages/ikg-core`: `IKG-search-proto`에서 온 검색/인덱싱 코어
- `apps/bookmark-graphview`: 북마크 그래프 프론트엔드
- `apps/async-worker`: Python 비동기 백엔드 파이프라인

이전 리뷰에서 지적했던 `IKG-search-proto` 파일 누락 문제는 다시 확인한 결과 해결되었다. 이전에 누락으로 보였던 `hybrid_search/*`, `indexing.py`, `v3_test.py`, `log/v3_converged_log.txt`, `test/*`, `core/*` 파일은 현재 `packages/ikg-core` 아래에 존재하며 원본 파일과 내용도 일치한다.

## 남은 문제

### 1. Python 패키지 import 경로가 아직 불안정함

모노레포 구조로 옮기면서 코어 코드는 `packages/ikg-core` 아래로 이동했지만, Python import는 여전히 기존 단일 프로젝트/평면 디렉터리 구조를 가정하고 있다.

예시:

- `main.py`는 `from embedder import ...`, `from indexer import ...` 형태로 `BGEEmbedder`, `Indexer`를 import한다.
- `apps/async-worker/tasks.py`는 `from core.embedder import BGEEmbedder`를 사용한다.
- `packages/ikg-core` 안에는 `core/embedder.py`와 루트 레벨 `embedder.py`가 함께 있어, 실행 위치와 `PYTHONPATH`에 따라 서로 다른 파일이 import될 수 있다.

권장 방향:

- `ikg_core` 같은 하나의 공식 패키지명을 정한다.
- 소스 파일을 `packages/ikg-core/src/ikg_core/` 같은 실제 import 가능한 패키지 디렉터리로 정리한다.
- import를 `from ikg_core.embedder import BGEEmbedder`처럼 절대 import로 통일한다.
- 현재 작업 디렉터리에 따라 import가 달라지는 구조를 피한다.

### 2. Python workspace/package 설정이 불완전함

현재 루트에는 Python workspace를 선언하는 `pyproject.toml`이 없다. Python 설정은 아래처럼 분산되어 있다.

- `packages/ikg-core/pyproject.toml`
- `apps/async-worker/pyproject.toml`
- 루트 `uv.lock`
- `apps/async-worker/uv.lock`

이 상태에서는 패키지 간 관계가 명시적으로 정의되지 않는다. 특히 백엔드가 로컬 코어 패키지에 의존한다는 사실이 설정 파일에 드러나지 않는다.

권장 방향:

- `uv`를 계속 사용할 경우 루트에 workspace 설정을 추가한다.
- `packages/ikg-core`와 `apps/async-worker`를 workspace member로 등록한다.
- `apps/async-worker`가 workspace 내부의 코어 패키지에 의존하도록 설정한다.
- 의도적으로 앱별 lockfile을 유지할 이유가 없다면, 모노레포 기준 lockfile 정책을 하나로 정한다.

### 3. async-worker 의존성 선언이 실제 코드와 맞지 않음

`apps/async-worker/pyproject.toml`은 현재 아래처럼 되어 있다.

```toml
dependencies = []
```

하지만 실제 코드는 다음 패키지들을 import하고 사용한다.

- `fastapi`
- `pydantic`
- `celery`
- `faiss`
- `numpy`
- 로컬 코어 임베딩 코드

검증 중 확인된 실패:

```text
ModuleNotFoundError: No module named 'celery'
```

권장 방향:

- `apps/async-worker/pyproject.toml`에 실제 런타임 의존성을 추가한다.
- 로컬 코어 패키지를 의존성으로 추가한다.
- worker가 코어 패키지를 통해서만 추론 관련 기능을 사용한다면, 무거운 추론 의존성은 코어 패키지 쪽에 두는 방식도 고려한다.

### 4. 런타임 경로가 실행 위치에 묶여 있음

여러 파일 경로가 상대 경로로 작성되어 있어, 프로세스를 어디에서 실행하느냐에 따라 동작이 달라질 수 있다.

예시:

- `apps/async-worker/main.py`는 `db/ikg_metadata.db`를 연다.
- `apps/async-worker/tasks.py`는 `db/ikg_metadata.db`, `db/ikg_vector.index`, `./model/bge-m3-onnx-int8`를 사용한다.
- `main.py`는 `bookmarks-2026-01-18.json`, `./model/bge-m3-onnx-int8`를 사용한다.

모노레포에서는 서비스가 repo root, app 디렉터리, Docker working directory, CI runner 등 다양한 위치에서 실행될 수 있으므로 이 구조는 취약하다.

권장 방향:

- 런타임 설정을 환경 변수로 중앙화한다.
- 경로는 명확한 기준 디렉터리에서 해석되도록 만든다.
- 모델, 데이터베이스, FAISS index의 기대 위치를 문서화한다.
- 생성되는 런타임 데이터와 소스 패키지 디렉터리를 섞지 않는다.

### 5. 프론트엔드가 JS workspace 일부로 설정되어 있지 않음

`apps/bookmark-graphview`에는 자체 `package.json`이 있지만, 저장소 루트에는 npm/pnpm/yarn workspace 설정이 없다.

검증 중 확인된 실패:

```text
sh: 1: vite: not found
```

이 문제는 단순히 의존성이 설치되지 않았기 때문일 수 있다. 다만 현재 모노레포에는 루트에서 전체 앱을 설치하거나 빌드하는 표준 명령이 없다.

권장 방향:

- 프론트엔드를 루트에서 관리하려면 JS workspace를 추가한다.
- 루트에 `build:frontend`, `lint:frontend` 또는 package manager workspace 명령을 제공한다.
- 루트 README에 프론트엔드 의존성 설치와 빌드 방법을 정리한다.

### 6. 프론트엔드 확장 프로그램 빌드 스크립트가 Windows 전용임

`apps/bookmark-graphview/package.json`에는 아래 스크립트가 있다.

```json
"build:chrome": "vite build && copy public\\manifest.json dist\\manifest.json",
"build:firefox": "vite build && copy public\\manifest.firefox.json dist\\manifest.json"
```

`copy` 명령과 백슬래시 경로는 Windows 전용이다. Linux/macOS 또는 대부분의 CI runner에서는 실패한다.

권장 방향:

- cross-platform 방식으로 교체한다.
- 작은 Node script, `cpy-cli`, Vite plugin 설정 등을 사용할 수 있다.
- Linux만 타깃이라면 POSIX 호환 명령으로 정리하는 방법도 있다.

### 7. 검증 tooling이 현재 설정만으로는 준비되지 않음

이전 검증 시도는 로컬 도구 누락으로 막혔다.

- 루트 Python 환경에 `pytest`가 없다.
- 프론트엔드 빌드에 필요한 `vite`가 설치되어 있지 않다.
- async worker에 필요한 `celery`가 의존성으로 선언되어 있지 않다.

권장 방향:

- test/lint 도구를 dev dependency group에 추가한다.
- 루트 README에 공식 검증 명령을 정의한다.
- 최소한 아래 검증 명령은 문서화하는 것이 좋다.
  - Python import smoke test
  - core tests
  - async-worker import/start check
  - frontend build

## 권장 작업 순서

1. `packages/ikg-core`를 실제 Python 패키지로 정규화한다.
2. 모든 import를 공식 패키지명 기준으로 수정한다.
3. 루트 workspace 설정과 로컬 패키지 의존성을 추가한다.
4. `apps/async-worker` 의존성을 실제 코드에 맞게 고친다.
5. 런타임 경로와 환경 변수를 중앙화한다.
6. 루트 레벨 검증 명령을 추가한다.
7. 프론트엔드 확장 프로그램 빌드 스크립트를 cross-platform으로 바꾼다.

## 현재 상태 요약

- Search proto 파일 이전: 해결됨.
- 프론트엔드 소스 이전: 파일 비교 기준으로 완료된 것으로 보임.
- 모노레포 실행 모델: 아직 완료되지 않음.
- 가장 위험도가 높은 남은 영역: Python 패키지/import/의존성 구조.

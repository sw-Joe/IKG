# Monorepo 개발환경 통합 작업 목록

현재 프로젝트는 기존 3개의 개별 저장소를 하나의 monorepo로 통합하는
중간 단계에 있다.

- `apps/bookmark-graphview`: FE 브라우저 확장 / 새 탭 시작 페이지
- `apps/async-worker`: BE API 및 비동기 워커
- `packages/ikg-core`: AI inference, 인덱싱, 검색 코어

첫 번째 목표는 루트 디렉토리 기준으로 설치, 실행, 테스트가 가능한
최소 monorepo 개발환경을 만드는 것이다.

## 1. Monorepo 구조 확정

- [ ] 현재 최상위 구조를 유지할지 확정한다.
  - `apps/bookmark-graphview`
  - `apps/async-worker`
  - `packages/ikg-core`
- [ ] 루트 `README.md`에 각 workspace의 역할을 문서화한다.
- [ ] 루트 `main.py`의 역할을 결정한다.
  - 임시 실행 스크립트로 유지
  - `scripts/`로 이동
  - 패키지 CLI로 전환
- [x] `.gitignore`에 남아 있는 merge conflict marker를 제거한다.
- [ ] Python, Node, 모델 파일, DB 파일, 빌드 산출물 ignore 규칙을 통합한다.

## 2. Python 패키지 구조 정리

- [ ] `packages/ikg-core`를 import 가능한 Python 패키지로 만든다.
- [ ] 패키지 import 이름을 정한다.
  - 권장: `ikg_core`
- [ ] 필요하다면 소스 파일을 패키지 디렉토리 구조로 이동한다.
- [ ] 현재 깨질 수 있는 local import를 정리한다.
  - `from embedder import BGEEmbedder`
  - `from indexer import Indexer`
  - `from core.embedder import BGEEmbedder`
- [ ] package import 방식으로 통일한다.
  - 예: `from ikg_core.embedder import BGEEmbedder`
  - 예: `from ikg_core.indexer import Indexer`
- [ ] 수동 실행용 스크립트와 자동화 테스트를 분리한다.

## 3. Python 버전 및 의존성 통합

- [ ] monorepo에서 사용할 Python 버전을 하나로 결정한다.
- [ ] 루트 `.python-version`, 루트 `pyproject.toml`,
      `apps/async-worker/.python-version`을 맞춘다.
- [ ] 루트 `uv` workspace를 사용할지 결정한다.
- [ ] 의존성을 역할별로 분리한다.
  - Core: `faiss-cpu`, `onnxruntime`, `transformers`, `rank-bm25`,
    `trafilatura`, `playwright`
  - BE: `fastapi`, `uvicorn`, `celery`, `redis`, `pydantic`
  - Dev: `pytest`, `ruff`
- [ ] `apps/async-worker/pyproject.toml`에 누락된 의존성을 추가한다.
- [ ] 루트에서 clean install이 가능한지 검증한다.

## 4. Core와 Backend 경계 정의

- [ ] `ikg-core`가 외부에 제공할 public API를 정의한다.
  - `BGEEmbedder`
  - `Indexer`
  - `HybridSearcher`
  - 북마크 추출 유틸리티
- [ ] `apps/async-worker`는 core의 public API만 사용하도록 정리한다.
- [ ] 가능한 범위에서 하드코딩된 경로를 제거한다.
- [ ] 환경 변수 또는 설정값을 도입한다.
  - `IKG_DB_PATH`
  - `IKG_INDEX_PATH`
  - `IKG_MODEL_PATH`
  - `REDIS_URL`
- [ ] DB, FAISS index, 모델 파일의 기본 로컬 경로를 결정한다.

## 5. DB 및 FAISS 저장 규칙 안정화

- [ ] 표준 DB 경로를 정한다.
  - 예: `db/ikg_metadata.db`
- [ ] 표준 FAISS index 경로를 정한다.
  - 예: `db/ikg_vector.index`
- [ ] SQLite bookmark row와 FAISS vector id의 매핑 규칙을 문서화한다.
- [ ] DB row 수와 FAISS vector 수를 비교하는 무결성 검사를 추가한다.
- [ ] 중복 URL 처리 정책을 정의한다.
- [ ] Celery worker가 FAISS index를 갱신할 때 사용할 write lock 전략을 정한다.
- [ ] DB insert는 성공했지만 vector indexing이 실패한 경우의 복구 정책을 정한다.

## 6. Frontend 개발환경 정리

- [ ] 누락된 ESLint plugin을 의존성에 추가한다.
  - `eslint-plugin-react-hooks`
  - `eslint-plugin-react-refresh`
- [ ] FE 스타일링 방식을 결정한다.
  - Tailwind CSS 사용
  - 일반 CSS 사용
- [ ] Tailwind를 사용할 경우 설정 파일과 의존성을 추가한다.
- [ ] 일반 CSS를 사용할 경우 JSX의 Tailwind utility class를 제거하거나 대체한다.
- [ ] Chrome/Firefox extension build script를 OS 독립적으로 수정한다.
- [ ] `package.json`의 Windows 전용 `copy` 명령을 제거한다.
- [ ] `npm run build`가 통과하는지 검증한다.
- [ ] `npm run lint`가 통과하는지 검증한다.

## 7. FE와 BE 연동 계약 정의

- [ ] 어떤 기능이 브라우저 확장 내부에서만 동작할지 결정한다.
- [ ] 어떤 기능이 backend API를 호출할지 결정한다.
- [ ] API endpoint를 정의한다.
  - `POST /api/bookmarks`
  - `GET /api/search?q=...`
  - `GET /api/bookmarks`
- [ ] 로컬 개발용 CORS 설정을 추가한다.
- [ ] backend 접근에 필요한 browser extension permission을 정의한다.
- [ ] 검색창의 역할을 결정한다.
  - Google 검색
  - IKG 검색
  - 둘 다 지원
- [ ] 로컬 FE/BE 개발 URL을 문서화한다.

## 8. 루트 개발 명령 추가

- [ ] 공통 task runner를 선택한다.
  - `make`
  - `just`
  - package script
- [ ] 루트에서 실행할 공통 명령을 추가한다.
  - `install`
  - `dev-fe`
  - `dev-api`
  - `dev-worker`
  - `test`
  - `lint`
  - `format`
- [ ] 새 clone 후 의존성 설치만으로 공통 명령이 동작하는지 검증한다.

## 9. 테스트 및 Smoke Check 추가

- [ ] non-interactive core 테스트를 pytest 테스트로 정리한다.
- [ ] `input()`을 사용하는 수동 테스트는 `scripts/`로 이동하거나 manual test로 표시한다.
- [ ] `ikg_core` import smoke test를 추가한다.
- [ ] BE API smoke test를 추가한다.
- [ ] FAISS/SQLite 무결성에 대한 최소 테스트를 추가한다.
- [ ] FE build 검증을 추가한다.
- [ ] 최소 로컬 검증 명령을 정한다.
  - `uv run pytest`
  - `uv run ruff check`
  - `npm run build`
  - `npm run lint`

## 10. 문서 정리

- [ ] 루트 `README.md`를 monorepo 기준으로 다시 작성한다.
- [ ] 필요한 도구와 버전을 문서화한다.
- [ ] 설치 방법을 문서화한다.
- [ ] FE dev server 실행 방법을 문서화한다.
- [ ] BE API 실행 방법을 문서화한다.
- [ ] Celery worker 실행 방법을 문서화한다.
- [ ] 인덱싱 흐름을 문서화한다.
- [ ] 검색 흐름을 문서화한다.
- [ ] 브라우저 확장 빌드 및 로드 방법을 문서화한다.
- [ ] 각 app/package README는 해당 영역의 세부 실행 방법만 담도록 정리한다.

## 초기 실행 순서

1. `.gitignore` conflict marker 제거
2. Python package/import 구조 정리
3. Python 의존성 및 workspace 설정 통합
4. FE build/lint 환경 정리
5. 루트 개발 명령 추가
6. 루트 README 문서화

위 항목들이 완료되면 이 저장소는 최소한의 monorepo 상태가 된다.
즉, 루트 기준으로 설치 가능하고, 실행 가능하고, 테스트 가능하며,
프로젝트 구조를 문서만 보고 이해할 수 있어야 한다.

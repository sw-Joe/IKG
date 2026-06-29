# IKG (Intelligent Knowledge Graphing)

IKG는 개인 북마크와 웹 문서를 수집하고, 임베딩 기반 벡터 검색과 BM25 기반 키워드 검색을 결합해 고밀도 지식 그래프 및 시맨틱 검색 경험을 자원 통제형으로 구축하는 모노레포(Monorepo) 프로토타입 프로젝트입니다.

현재 자원이 제한된 환경(랩탑 등)에서의 안정성을 사수하기 위해 브라우저 싱글톤 컨텍스트 세마포어 가드와 SQLite 트랜잭션 뮤텍스가 백엔드 코어에 전격 정착되었으며, 인프라 컨테이너화 및 모노레포 툴체인 최적화가 진행 중입니다.


## Workspace 구조
```Plaintext
.
├── apps/
│   ├── FE/                 # React / Vite / TypeScript 기반 3D 지식 그래프 대시보드
│   └── be_api/             # FastAPI 기반 실시간 게이트웨이 및 비동기 직렬화 워커
├── packages/
│   └── ai_core/            # BGE-M3 ONNX, BM25, FAISS, 문맥 어텐션 융합 검색 코어
├── db/                     # 로컬 SQLite 매역 데이터 및 FAISS Index 공간 저장소
├── log/                    # ai_core 격리 런타임 로그 및 시스템 추적 아티팩트
├── pyproject.toml          # 루트 uv 모노레포 워크스페이스 정의 스펙
└── uv.lock                 # 최신 수렴 의존성 락파일
```


## 실행환경 구축 방법

프로젝트의 엔트리포인트와 종단간(E2E) 파이프라인을 로컬 워크스페이스 상에서 수동 기동하고 제어하기 위한 인프라 프로비저닝 스펙입니다.

### 1. 사전 필수 요구사항

- OS: Ubuntu Linux (24.04 LTS 권장)
- Runtime: Node.js (v24+ 권장), Python (3.13+)
- Package Manager: uv (Python 모노레포 관리), npm (Frontend 관리)

### 2. Python 모노레포 가상환경 및 의존성 정렬

프로젝트 루트 디렉토리에서 uv 툴체인을 이용하여 상호 참조 패키지를 동기화하고 Playwright 크롬 바이너리를 무결하게 가동합니다.

```Bash
# 1. uv 워크스페이스 전체 의존성 동기화 및 가상환경 생성
uv sync

# 2. 크롬 헤드리스 브라우저 커널 및 필수 OS 종속 라이브러리 일괄 빌드
uv run playwright install chromium --with-deps

# 3. 로컬 패키지 아티팩트 캐시 링크 동기화 강제 리프레시
uv sync --package ai-core --package be-api
```

### 3. 데이터베이스 인프라 구조 초기화 (최초 1회 필수)

벌크 임베딩 적재 및 격리 가드 테이블 마이그레이션을 영속화합니다.

```Bash
# SQLite 및 FAISS 벡터 저장 공간 디스크 뼈대 형성
uv run python migrate_db_infrastructure.py
```

### 4. 대량 북마크 일괄 마이그레이션 (선택 사항)

기존 브라우저에서 백업된 대용량 JSON 문서를 기반으로 1차 고밀도 텐서 공간을 형성합니다.

```Bash
# bookmarks.json 원본을 기하학적 차원 배열 및 SQLite 메인 테이블에 안착
uv run python packages/ai_core/src/ai_core/core/bulk_importer.py bookmarks.json
```


## 인프라 가동 프로토콜

### 1. Backend Gateway 가동 (be_api)

하드웨어 사양 맞춤형 세마포어와 입출력 스캔 미들웨어가 내장된 상용 수준의 게이트웨이를 포격 가동합니다.

```Bash
uv run --package be-api python -m be_api.app
```

#### 인프라 정합성 동기화 트리거
서버 기동 직후, 인메모리 스냅샷과 벡터 공간 축을 완전히 동기화하기 위해 새 터미널에서 아래 명령을 1회 호출해 주십시오.
```Bash
curl -X POST http://127.0.0.1:8000/api/system/sync
```

### 2. Frontend Dashboard 가동 (FE)

3D 지식 구조 토폴로지를 드로잉하는 Vite 개발 서버를 가동합니다.

```Bash
cd apps/FE
npm install
npm run dev
```

### 3. 관측성(Observability) 실시간 모니터링 분리

중앙 집중형 미들웨어 로깅 설계 사상에 따라, ai_core 패키지 내부의 무거운 웹 스크래핑 로그와 세마포어 상태 정보는 콘솔창을 어지럽히지 않고 파일로 격리 배출됩니다. 실시간 제어를 위해 별도 터미널에서 아래 스트리밍을 추적하십시오.

```Bash
tail -f log/ai_core_runtime.log
```


## Docker 패키징 (향후 구현 예정)

파이썬 백엔드 게이트웨이 및 ONNX 인퍼런스 인프라 환경 전체를 완벽히 격리된 컨테이너 사양으로 패키징하기 위한 문단입니다. 컨테이너 아키텍처 수립 시 가동 명세를 확정하여 반영할 예정입니다.

(현재 컨테이너라이징 작업 준비 중 - 빈 문서 상태 유지)


## 검증 및 테스트 명령

개발 컨텍스트의 무결성을 빠르게 검증하기 위한 최소 검증 스크립트 명세입니다.

```Bash
# 1. 모노레포 전체 소스코드 구문 무결성 정적 컴파일 체크
python3 -m compileall -q packages/ai_core/src apps/be_api/src

# 2. 코드 스타일 린팅 검수
.venv/bin/ruff check packages/ai_core/src apps/be_api/src

# 3. 하이브리드 검색 코어 레이어 격리 단위 테스트
uv run python packages/ai_core/test/hybrid_search_test.py
```
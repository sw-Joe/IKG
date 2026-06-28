import os
from pathlib import Path

from dotenv import load_dotenv

# 1. 최상단 루트 디렉토리 기점의 .env 파일을 찾아 시스템 환경변수 메모리에 적재
# 로컬 가동 시 .env가 주입되어 os.getenv가 작동할 수 있게 만듭니다.
load_dotenv()

# 2. config.py 파일의 물리적 위치를 기준으로 프로젝트 최상단 루트(IKG/) 절대 경로 동적 계산
# 구조 트리: packages/ai_core/src/ai_core/config.py -> 5단계 상위 부모(parents[4])가 루트
_CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = _CURRENT_FILE.parents[4]

# 3. 환경변수(.env) 설정을 최우선으로 바인딩하되, 누락 시 최상단 루트 기점의 절대 경로로 폴백
IKG_DB_PATH = os.getenv("IKG_DB_PATH", os.path.join(PROJECT_ROOT, "db", "ikg_metadata.db"))
IKG_INDEX_PATH = os.getenv("IKG_INDEX_PATH", os.path.join(PROJECT_ROOT, "db", "ikg_vector.index"))
IKG_MODEL_PATH = os.getenv("IKG_MODEL_PATH", os.path.join(PROJECT_ROOT, "model", "bge-m3-onnx-int8"))
IKG_MODEL_FILE = os.getenv("IKG_MODEL_FILE", "model_quantized.onnx")

# 4. 런타임 저장 폴더 누수 방지를 위한 디렉토리 선제 강제 생성 가드레일
os.makedirs(os.path.dirname(IKG_DB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(IKG_INDEX_PATH), exist_ok=True)

# 5. 콘솔 정합성 확인 피드백
print("[IKG CONFIG BOUND]")
print(f" - Project Root Path: {PROJECT_ROOT}")
print(f" - Active DB Path:    {IKG_DB_PATH}")
print(f" - Active Index Path: {IKG_INDEX_PATH}")
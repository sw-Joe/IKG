import logging
import os
import sys



def setup_logging():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    log_dir = os.path.join(project_root, "log")
    os.makedirs(log_dir, exist_ok=True)

    # 전역 공통 포맷 정의 (상용 수준의 정형화 스펙)
    log_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] (PID:%(process)d) -> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # =========================================================================
    # 1. 루트(Root) 로ger 및 기본 터미널 표준 출력 핸들러 설정
    # =========================================================================
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 기존 핸들러 중복 초기화 방지 가드
    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)

    # =========================================================================
    # 2. be_api 네임스페이스 전용 파일 핸들러 (선택 사항)
    # =========================================================================
    be_logger = logging.getLogger("be_api")
    be_file_path = os.path.join(log_dir, "be_gateway.log")
    be_file_handler = logging.FileHandler(be_file_path, encoding="utf-8")
    be_file_handler.setFormatter(log_formatter)
    be_logger.addHandler(be_file_handler)

    # =========================================================================
    # 3. [ISOLATION HARDENED]: ai_core 네임스페이스 완전 격리 핸들러
    # =========================================================================
    ai_logger = logging.getLogger("ai_core")
    ai_logger.setLevel(logging.INFO)
    
    # 전용 로그 파일 경로 바인딩 (/home/joe/PROJECT/IKG/log/ai_core_runtime.log)
    ai_file_path = os.path.join(log_dir, "ai_core_runtime.log")
    ai_file_handler = logging.FileHandler(ai_file_path, encoding="utf-8")
    ai_file_handler.setFormatter(log_formatter)
    ai_logger.addHandler(ai_file_handler)

    # [CRITICAL PROPERTY]: propagate 플래그를 False로 설정하여
    # ai_core 산하 로거들이 뱉는 로그가 메인 터미널(Root Logger)로 전파되어 화면을 어지럽히는 것을 차단합니다.
    ai_logger.propagate = False

    # 레거시 써드파티 노이즈 마스킹
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
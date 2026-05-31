import os
import logging
from logging.handlers import TimedRotatingFileHandler



def setup_logging():
    """애플리케이션 전역 및 하부 모듈 네임스페이스의 고해상도 로깅 시스템 초기화"""
    
    # 1. 최상위 루트 로거 및 자식 로거 타겟 인스턴스 웜업
    be_logger = logging.getLogger("be_api")
    ai_logger = logging.getLogger("ai_core")
    
    be_logger.setLevel(logging.DEBUG)
    ai_logger.setLevel(logging.DEBUG)

    # 2. 공통 출력 텍스트 서식(Formatter) 정의
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s (PID:%(process)d) -> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 3. Console Stream Handler (터미널 실시간 모니터링 버스 버퍼링)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)  # 콘솔 스트리밍은 주요 정보만 수렴

    # 4. File Persistent Handler (자정 기점 롤링 디스크 저장 - 최대 14일 보존)
    # 현재 파일 위치 기준 프로젝트 최상위 루트의 log 폴더 역산 추출
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    log_dir = os.path.join(base_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "ikg_runtime.log"),
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # 디스크에는 마이크로 디버그 매트릭스 전수 기록

    # 5. 기존 핸들러 중복 등록 방지 가드레일 후 체인 결합
    if not be_logger.handlers:
        be_logger.addHandler(stream_handler)
        be_logger.addHandler(file_handler)
        
    if not ai_logger.handlers:
        ai_logger.addHandler(stream_handler)
        ai_logger.addHandler(file_handler)

    be_logger.info("모듈형 중앙 집중 로깅 인프라 엔진이 활성화되었습니다.")
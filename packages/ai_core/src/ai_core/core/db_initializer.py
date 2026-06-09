import os
import sqlite3
import logging



logger = logging.getLogger("ai_core.core.db_initializer")


def initialize_database_schema(db_path: str) -> bool:
    """
    지정된 경로의 SQLite 데이터베이스 파일 및 필수 스키마 정의를 강제 구축합니다.
    이미 테이블이 존재할 경우 안전하게 바이패스(Idempotent)합니다.
    """
    try:
        # 1. 파일 상위 디렉토리 스토리지 존재성 동적 강제 보장
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        # 2. SQLite 커넥션 컨텍스트 바인딩 (WAL 가동 타임아웃 30초 인입)
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")

        # 3. 메타데이터 최종 진실 공급원 bookmarks 테이블 영속 선언
        # (is_deleted: 0=활성, 1=논리소거, 2=검증보류 격리자산)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bookmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT,
                content TEXT,
                created_at TEXT,
                is_deleted INTEGER DEFAULT 0
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bookmarks_isolated (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT,
                content TEXT,
                created_at TEXT,
                isolation_reason TEXT            -- 격리(보류) 처리 사유 명시
            );
        """)
        
        # 4. CQRS 조회 및 지연 소거 정합성 싱크 가속을 위한 복합 인덱스 선언
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_bookmarks_active 
            ON bookmarks (id, is_deleted);
        """)
        
        conn.commit()
        logger.info(f"[PROVISIONING SUCCESS] 데이터베이스 인프라가 무결하게 초기화 수렴되었습니다. -> 타겟: {db_path}")
        return True
        
    except Exception as e:
        logger.critical(f"[PROVISIONING CRITICAL CRASH] 데이터베이스 가드 스키마 초기화 실패: {e}", exc_info=True)
        raise e
    finally:
        if 'conn' in locals():
            conn.close()
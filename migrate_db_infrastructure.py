import logging
import os
import sqlite3

# 1. 스크립트 전용 로깅 컨텍스트 초기화
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ikg.infrastructure_migration")

IKG_DB_PATH = "/home/joe/PROJECT/IKG/db/ikg_metadata.db"

def run_one_time_migration():
    logger.info("==========================================================================")
    logger.info("[MIGRATION START] SQLite 단일 파일 내 테이블 분리 격리 마이그레이션 개시")
    logger.info("==========================================================================")
    
    if not os.path.exists(IKG_DB_PATH):
        logger.error(f"-> [CRITICAL] 마이그레이션 대상 DB 파일이 지정 경로에 부재합니다: {IKG_DB_PATH}")
        return

    conn = sqlite3.connect(IKG_DB_PATH, timeout=60.0)
    cursor = conn.cursor()
    
    try:
        # WAL 모드 활성화 상태 점검 및 커널 락 세션 확보
        cursor.execute("PRAGMA journal_mode=WAL;")
        
        # ---------------------------------------------------------------------
        # Step 1. 신규 구조인 보류 격리 자산 전용 샌드박스 테이블 선언 생성
        # ---------------------------------------------------------------------
        logger.info("Step 1. 신규 격리 샌드박스 테이블(bookmarks_isolated) 구조 프로비저닝...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bookmarks_isolated (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT,
                content TEXT,
                created_at TEXT,
                isolation_reason TEXT
            );
        """)
        
        # ---------------------------------------------------------------------
        # Step 2. 기존bookmarks 테이블에 index_written 동기화 체크포인트 열 주입
        # ---------------------------------------------------------------------
        logger.info("Step 2. 메인 테이블 내 물리 동기화 체크포인트 열(index_written) 확장 검사...")
        try:
            cursor.execute("ALTER TABLE bookmarks ADD COLUMN index_written INTEGER DEFAULT 0;")
            logger.info(" -> [SUCCESS] index_written 컬럼 신설 완료.")
        except sqlite3.OperationalError:
            # 이미 컬럼이 존재하는 경우 예외 가드 바이패스
            logger.info(" -> [SKIP] index_written 컬럼이 이미 존재합니다.")

        # ---------------------------------------------------------------------
        # Step 3. 원자적 데이터 이관 및 물리 격리 트랜잭션 집행 (BEGIN)
        # ---------------------------------------------------------------------
        logger.info("Step 3. 이원화 스키마 규칙에 의거한 기존 데이터 마이그레이션 트랜잭션 집행...")
        cursor.execute("BEGIN TRANSACTION;")
        
        # A. 기존에 수집 보류 격리 판정(is_deleted=2)을 받았던 행 전체 개수 계측
        cursor.execute("SELECT COUNT(*) FROM bookmarks WHERE is_deleted = 2")
        isolated_count = cursor.fetchone()[0]
        
        if isolated_count > 0:
            logger.info(f" -> 발견된 기존 보류 격리 대상 데이터: {isolated_count}건")
            
            # B. bookmarks_isolated 테이블로 기존 2번 플래그 자산들을 원자적 복사 (기존 멱등 데이터 보존)
            cursor.execute("""
                INSERT OR IGNORE INTO bookmarks_isolated (url, title, content, created_at, isolation_reason)
                SELECT url, title, content, created_at, 'BULK_IMPORT_VALIDATION_FAILED'
                FROM bookmarks
                WHERE is_deleted = 2
            """)
            
            # C. 메인 bookmarks 테이블 공간에서 이관 완료된 보류 행 물리 제거
            # 이 작업을 통해 메인 테이블에는 100% 무결한 활성 벡터 자산들만 정렬 상주하게 됩니다.
            cursor.execute("DELETE FROM bookmarks WHERE is_deleted = 2")
            logger.info(f" -> [SUCCESS] 보류 자산 {isolated_count}건 격리 테이블 이관 및 메인 공간 소거 완수.")
        else:
            logger.info(" -> 마이그레이션이 필요한 기존 보류 격리 데이터(is_deleted=2)가 없습니다.")

        # ---------------------------------------------------------------------
        # Step 4. 기존 정상 자산들에 대한 index_written = 1 초기화 마킹
        # ---------------------------------------------------------------------
        # 이미 5시간 동안 FAISS 인덱스(7.87MB) 빌드가 완료되었으므로, 
        # 메인 테이블에 남아있는 실존 활성 문서들은 완벽히 벡터 작성이 끝난 상태입니다. 전부 1로 동기화합니다.
        cursor.execute("UPDATE bookmarks SET index_written = 1 WHERE is_deleted = 0")
        cursor.execute("SELECT COUNT(*) FROM bookmarks WHERE index_written = 1")
        active_indexed_count = cursor.fetchone()[0]
        logger.info(f"Step 4. 기존 실존 활성 자산 {active_indexed_count}건에 대해 index_written=1 일괄 마킹 완수.")

        # 전과정 결함 없으므로 디스크 물리 반영 커밋 수렴
        conn.commit()
        logger.info("==========================================================================")
        logger.info("[MIGRATION SUCCESS] 인프라 스키마 전면 개편 및 원자적 이관 정합성 수렴 완료.")
        logger.info("==========================================================================")

    except Exception as e:
        conn.rollback()
        logger.error("==========================================================================")
        logger.error(f"[MIGRATION CRASH] 마이그레이션 연산 중 치명적 설계 결함 감지 후 롤백: {str(e)}")
        logger.error("==========================================================================")
    finally:
        conn.close()


if __name__ == "__main__":
    run_one_time_migration()
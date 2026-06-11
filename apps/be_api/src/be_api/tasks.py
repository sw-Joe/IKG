import os
import sqlite3
import logging
import faiss
import numpy as np

from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH, IKG_MODEL_PATH, IKG_MODEL_FILE
from ai_core.core.embedder import BGEEmbedder
from ai_core.core.indexer import VectorIndexer

logger = logging.getLogger("be_api.tasks")


class EmbeddedInferenceWorker:
    """내장형 동기식 단일 스레드 직렬화 컨텍스트 트랜잭션 전담 인프라 액터 클래스"""
    def __init__(self, db_path: str, index_path: str):
        self._db_path = db_path
        self._index_path = index_path

        logger.info("[WORKER INIT] EMBEDDED 모드 전용 BGE-M3 ONNX 고속 인퍼런스 엔진을 탑재합니다.")
        self.embedder = BGEEmbedder(model_path=IKG_MODEL_PATH, file_name=IKG_MODEL_FILE)
        self.indexer_engine = VectorIndexer(dimension=1024)


    def execute_sequential_inference_pipeline(self, bookmark_id: int) -> dict:
        """
        [WRITE ATOMIC]: 단건 증분 인덱싱 및 물리 파일 디스크 커밋 영구 기록 파이프라인
        - 하드 디스크 드라이브 저장 성공이 완벽히 보장(트리거)받은 최종 시점에만 index_written = 1 마킹을 집행합니다.
        """
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            # 메인 데이터 자산 테이블에서 타깃 식별
            cursor.execute("SELECT title, content FROM bookmarks WHERE id = ?", (bookmark_id,))
            row = cursor.fetchone()
            if not row:
                logger.warning(f"[WORKER SKIPPED] 임베딩 타깃 자산 식별자 부재 -> ID: #{bookmark_id}")
                return {"status": "SKIPPED"}

            combined_text = f"{row['title']} {row['content']}"

            # 1. 고밀도 다차원 행렬 BGE-M3 ONNX 단건 추론 실행
            query_vec = self.embedder.encode_single(combined_text)  # [1, 1024] 형상 반환
            vector_np = query_vec.astype("float32")

            # 2. FAISS 인덱스 가상 공간 로드 및 적재
            index = faiss.read_index(self._index_path)

            # 중복 파편 인입 방지를 위한 아이덴티티 안전 선제 소거
            purge_id_np = np.array([bookmark_id], dtype=np.int64)
            index.remove_ids(purge_id_np)

            # 3. 조밀 순차 발급된 무결한 고유 식별자(PK)로 차원 공간 영속 매핑 바인딩
            index.add_with_ids(vector_np, purge_id_np)

            # 4. 물리 하드 드라이브 스토리지 원자적 영구 기록 플러시
            faiss.write_index(index, self._index_path)

            # -----------------------------------------------------------------
            # [CRITICAL CHECKPOINT]: FAISS 물리 디스크 파일 저장이 완벽 완수된 직후 마킹
            # -----------------------------------------------------------------
            cursor.execute("UPDATE bookmarks SET index_written = 1 WHERE id = ?", (bookmark_id,))
            conn.commit()

            logger.info(f"[WORKER TRACKING SUCCESS] 메인 자산 ID #{bookmark_id} 차원 적재 및 index_written=1 체크포인트 수렴 완료.")
            return {"status": "SUCCESS", "id": bookmark_id}

        except Exception as e:
            conn.rollback()
            logger.error(f"[WORKER CRITICAL ERROR] 내장 큐 파이프라인 작동 실패: {str(e)}", exc_info=True)
            return {"status": "FAILED", "error": str(e)}
        finally:
            conn.close()


    def execute_sequential_removal_pipeline(self, bookmark_id: int) -> dict:
        """[DELETE ATOMIC]: 단건 물리 완전 삭제 발생 시 FAISS 인덱스 공간 하이브리드 동기 소거 프로토콜"""
        try:
            index = faiss.read_index(self._index_path)
            purge_id_np = np.array([bookmark_id], dtype=np.int64)

            # FAISS 내부 벡터 공간에서 해당 고유 식별자 매핑 좌표 물리 소거
            index.remove_ids(purge_id_np)

            # 디스크 파일 업데이트 플러시 기록
            faiss.write_index(index, self._index_path)
            logger.info(f"[WORKER RE-INDEXER] FAISS 인덱스 공간 내 자산 ID #{bookmark_id} 기하학 벡터 영구 소거 완수.")
            return {"status": "SUCCESS", "purged_id": bookmark_id}
        except Exception as e:
            logger.error(f"[WORKER REMOVAL CRASH] FAISS 공간 소거 연산 중 장애 가드 발생: {e}")
            return {"status": "FAILED", "error": str(e)}
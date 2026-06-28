import logging
import os
import sqlite3

import faiss
import numpy as np

from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH

logger = logging.getLogger("ai_core.core.indexer")


class VectorIndexer:
    def __init__(self, db_path=None, index_path=None, dimension=1024):
        self.db_path = db_path or IKG_DB_PATH
        self.index_path = index_path or IKG_INDEX_PATH
        self.dimension = dimension
        self._initialize_id_map_index()


    def _initialize_id_map_index(self):
        """인덱스 파일 시스템 부재 시 최초 뼈대 구축 가드"""
        if not os.path.exists(self.index_path):
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            sub_index = faiss.IndexFlatIP(self.dimension)
            id_map_index = faiss.IndexIDMap2(sub_index)
            faiss.write_index(id_map_index, self.index_path)


    def sync_index_with_database(self, embedder) -> dict:
        """
        RDBMS 데이터베이스 스냅샷을 원천 오리지널로 취급하여
        FAISS IndexIDMap과의 물리 불일치(누락/잔존) 자산을 완벽하게 자동 수렴 보정.
        """
        logger.info("[INDEXER COMMAND] 스토리지 전역 정합성 크로스 체크 매칭을 시작합니다.")
        
        # 1. 디스크 최신 FAISS 인덱스 로드 및 내부 영속 식별자 리스트화
        index = faiss.read_index(self.index_path)

        # 가드 로직: 로드된 인덱스가 IDMap 구조가 아닐 경우 예외를 뿜지 않고 자동 안전 재빌드 유도
        if not isinstance(index, faiss.IndexIDMap) and not hasattr(index, 'id_map'):
            logger.warning("[INDEXER] 구형 IndexFlatIP 규격이 디스크에서 감지되었습니다. 정합성 확보를 위해 인덱스를 제로셋 재빌드합니다.")
            sub_index = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIDMap(sub_index)
            faiss_ids = set()
        else:
            faiss_ids = set(faiss.vector_to_array(index.id_map))
        
        # 2. SQLite 데이터베이스 상태 콘텍스트 스캔 (is_deleted = 0 활성 자산 한정)
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT id, title, content FROM bookmarks WHERE is_deleted = 0")
            db_rows = cursor.fetchall()
            
            db_bookmarks = {row['id']: f"{row['title']} {row['content']}" for row in db_rows}
            db_ids = set(db_bookmarks.keys())
            
            # 3. 고속 수학적 차집합 분기점 도출
            ids_to_index = db_ids - faiss_ids   # Case A: DB에만 존재 (색인 유실분 추가 대상)
            ids_to_remove = faiss_ids - db_ids  # Case B: FAISS에만 잔존 (Soft-Deleted 가비지 물리 제거 대상)
            
            summary = {
                "initial_faiss_count": index.ntotal,
                "scraped_db_count": len(db_ids),
                "added_count": len(ids_to_index),
                "purged_count": len(ids_to_remove),
                "status": "NO_CHANGE"
            }
            
            if not ids_to_index and not ids_to_remove:
                logger.info("[INDEXER SYSTEM] DB와 FAISS 벡터 인덱스 정합성이 이미 100% 일치합니다. 작업을 스킵합니다.")
                return summary

            # 4. 소거 오퍼레이션 실행 (Orphaned / Soft-Deleted Vectors)
            if ids_to_remove:
                logger.info(f"[INDEXER SYSTEM] 고립 벡터 파편 {len(ids_to_remove)}건에 대한 타겟 물리 청소를 시작합니다.")
                purge_ids_np = np.array(list(ids_to_remove), dtype=np.int64)
                index.remove_ids(purge_ids_np)
                
            # 5. 증분 색인 오퍼레이션 실행 (Catch-up Bulk Ingestion)
            if ids_to_index:
                logger.info(f"[INDEXER SYSTEM] 유실 및 누락 자산 {len(ids_to_index)}건에 대한 실시간 증분 색인을 시작합니다.")
                batch_vectors = []
                batch_ids = []
                
                for idx, bookmark_id in enumerate(ids_to_index, 1):
                    text_content = db_bookmarks[bookmark_id]
                    
                    # 싱글톤 인퍼런스 엔진을 통한 임베딩 벡터 생성
                    query_vec = embedder.encode(text_content)
                    vector_np = query_vec[0].astype("float32")
                    
                    batch_vectors.append(vector_np)
                    batch_ids.append(bookmark_id)
                    
                    if idx % 50 == 0 or idx == len(ids_to_index):
                        logger.info(f" -> 벌크 고속 임베딩 추론 진행 중... ({idx}/{len(ids_to_index)})")
                
                # NumPy 행렬 가속 구조 재정렬 후 IndexIDMap 결합
                final_vectors_np = np.array(batch_vectors).astype("float32")
                final_ids_np = np.array(batch_ids, dtype=np.int64)
                index.add_with_ids(final_vectors_np, final_ids_np)

            # 6. 변경 사항 디스크 스냅샷 영구 동기화 커밋
            faiss.write_index(index, self.index_path)
            
            # 7. 만약 물리 소거된 항목이 있다면 DB 청소 트레이드오프 마감 (Hard Delete 동기화)
            if ids_to_remove:
                placeholders = ",".join(["?"] * len(ids_to_remove))
                cursor.execute(f"DELETE FROM bookmarks WHERE id IN ({placeholders})", list(ids_to_remove))
                conn.commit()

            summary["status"] = "SYNCHRONIZED"
            summary["final_faiss_count"] = index.ntotal
            logger.info(f"[INDEXER SUCCESS] 데이터베이스 동기화 완료 수렴 상태 진입 -> 총 벡터 수: {index.ntotal}")
            return summary

        except Exception as e:
            logger.error(f"[INDEXER CRITICAL] 정합성 보정 오퍼레이션 중 인프라 크래시: {e}", exc_info=True)
            raise
        finally:
            conn.close()


    def check_and_purge_garbage(self, current_index, purge_threshold: int = 20):
        """Soft-Delete 처리된 가비지 파편을 일괄 수집하여 지연된 물리 소거(Deferred Purge) 단독 집행"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        cursor = conn.cursor()
        
        try:
            # 논리 삭제 마킹된 로우 쿼리
            cursor.execute("SELECT id FROM bookmarks WHERE is_deleted = 1")
            deleted_ids = [row[0] for row in cursor.fetchall()]
            
            # 지정된 가비지 임계 임계치를 초과 도달 시에만 물리 소거 연산 집행하여 디스크 부하 제어
            if len(deleted_ids) >= purge_threshold:
                print(f"[PURGE COMMAND] 누적 가비지 임계값 도달 ({len(deleted_ids)}개). 물리 정리를 시작")
                
                # IndexIDMap 고유 기능인 특정 ID 배열 단독 타겟 소거 실행
                purge_ids_np = np.array(deleted_ids, dtype=np.int64)
                current_index.remove_ids(purge_ids_np)
                
                # 물리 청소 완결 바이너리 파일 영구 플러시
                faiss.write_index(current_index, self.index_path)
                
                # 데이터 정합성 종결을 위한 RDBMS 내의 실데이터 최종 하드 삭제(Hard Delete) 변환
                placeholders = ",".join(["?"] * len(deleted_ids))
                cursor.execute(f"DELETE FROM bookmarks WHERE id IN ({placeholders})", deleted_ids)
                conn.commit()
                print("[PURGE COMMAND SUCCESS] 디스크 벡터 최적화 및 메타데이터 영구 청소 완료")
        except Exception as e:
            print(f"[PURGE ERROR] 지연된 물리 소거 처리 중 예외 발생: {e}")
        finally:
            conn.close()
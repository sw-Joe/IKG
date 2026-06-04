import os
import sqlite3

import faiss
import numpy as np

from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH


class VectorIndexer:
    """FAISS IndexIDMap 물리 바인딩 기반 실시간 증분 색인 및 지연 소거 제어 컴포넌트"""
    def __init__(self, db_path=None, index_path=None, dimension=1024):
        self.db_path = db_path or IKG_DB_PATH
        self.index_path = index_path or IKG_INDEX_PATH
        self.dimension = dimension
        self._initialize_id_map_index()


    def _initialize_id_map_index(self):
        """디스크 상에 인덱스가 전무할 시 고유 고 식별자 매핑형 인덱스 원자적 빌드"""
        if not os.path.exists(self.index_path):
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            # 하부 내부 Flat 내적(Cosine 유사도 대응) 코어 선언
            sub_index = faiss.IndexFlatIP(self.dimension)
            # 순서 왜곡 방지를 위한 고유 ID 강제 결합 레이어 래핑
            id_map_index = faiss.IndexIDMap(sub_index)
            faiss.write_index(id_map_index, self.index_path)


    def add_document_vector(self, bookmark_id: int, text_content: str, embedder):
        """단건 가공 원문을 고밀도 Dense 벡터로 변환 후 지정된 PK ID와 물리 결합 및 디스크 저장"""
        # 1. BGE-M3 ONNX 싱글톤 인스턴스를 통한 고속 임베딩 추론
        query_vec = embedder.encode(text_content)
        vector_np = query_vec[0].astype("float32")
        
        # 2. 독점 파일 I/O 오픈
        index = faiss.read_index(self.index_path)
        
        # 3. 데이터 정합성을 위한 numpy 2차원 매핑 변환
        ids_np = np.array([bookmark_id], dtype=np.int64)
        vectors_np = np.expand_dims(vector_np, axis=0)
        
        # 물리 증분 색인 실행 (순서 밀림 완전 방어)
        index.add_with_ids(vectors_np, ids_np)
        faiss.write_index(index, self.index_path)
        print(f"[INDEXER COMMAND] 북마크 #{bookmark_id} 증분 완료 (전체 벡터 수: {index.ntotal})")
        
        # 4. 연쇄 후처리: 누적 가비지 추적 스케줄러 트리거
        self.check_and_purge_garbage(index, purge_threshold=20)


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
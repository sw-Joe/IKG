import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

# 수선된 분할 로깅 설정 모듈 및 코어 임포트
from be_api.logger_config import setup_logging
from ai_core import HybridSearcher



# 1. 인스턴스 초기화 전 로깅 서식 파이프라인 수립 가동
setup_logging()
logger = logging.getLogger("be_api.app")

# 2. FastAPI ASGI 엔진 수립
app = FastAPI(title="IKG Hybrid Search Gateway", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 하이브리드 검색 코어 웜업
searcher_engine = HybridSearcher()


@app.get("/api/graph")
def get_similarity_graph_topology(threshold: float = 0.85):
    logger.info(f"네트워크 시각화 벡터 엣지 토폴로지 추출 요청 감지 (FE 수신 threshold: {threshold})")
    
    if threshold < 0.85:
        logger.warning(f"인입 임계치({threshold}) 수위 미달로 안전 마진 한계선 0.85로 강제 오버라이드")
        threshold = 0.85

    try:
        searcher_engine.reload_indices()
        documents = searcher_engine.documents
        faiss_index = searcher_engine.index
        total_count = len(documents)
        
        logger.info(f"토폴로지 분석 매트릭스 웜업 - 수집 노드: {total_count}개 | 필터 임계값: {threshold}")
        
        if total_count == 0:
            return {"nodes": [], "edges": []}
            
        nodes = [{"id": str(doc["id"]), "title": doc["title"], "url": doc["url"]} for doc in documents]
        edges = []
        
        vectors = np.array([faiss_index.reconstruct(i) for i in range(total_count)]).astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        normalized_vectors = vectors / norms
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
        
        for i in range(total_count):
            for j in range(i + 1, total_count):
                sim_score = float(similarity_matrix[i][j])
                if sim_score >= threshold:
                    edges.append({
                        "source": str(documents[i]["id"]),
                        "target": str(documents[j]["id"]),
                        "value": round(sim_score, 4)
                    })
                    
        logger.info(f"벡터 공간 그래프 수렴 완결 -> 노드: {len(nodes)}개 | 연결선: {len(edges)}개")
        return {"nodes": nodes, "edges": edges}
        
    except Exception as e:
        logger.error(f"그래프 컴파일 연산 중 치명적 행렬 크래시: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="그래프 토폴로지 분석 연산 실패")


@app.get("/api/search")
def search_bookmarks_endpoint(
    q: Optional[str] = None, 
    query: Optional[str] = None, 
    limit: int = 5
):
    effective_query = q or query
    logger.info(f"실시간 하이브리드 검색 요청 수신 -> 명세 분석: q={q} | query={query} | 확정질의어='{effective_query}'")
    
    if not effective_query or not effective_query.strip():
        logger.warning("공백 질의어 인입 유입 차단 빈 배열 즉시 반환 조치")
        return []
        
    try:
        searcher_engine.reload_indices()
        results = searcher_engine.search(query=effective_query, top_n=limit)
        logger.info(f"하이브리드 검색 추론 파이프라인 반환 완결 (출력 볼륨: {len(results)}건)")
        return results
    except Exception as e:
        logger.error(f"하이브리드 검색 추론 연산 중 게이트웨이 예외: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="하이브리드 검색 랭킹 연산 장애")

if __name__ == "__main__":
    uvicorn.run("be_api.app:app", host="0.0.0.0", port=8000, reload=True)
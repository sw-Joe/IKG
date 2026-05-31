export interface BookmarkPayload {
  url: string;
  title: string;
  content: string;
}

export interface GraphNode {
  id: string;
  title: string;
  url: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  value: number;
}

export interface GraphTopologyResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export const bookmarkService = {
  /**
   * 1. 신규 북마크 접수 및 비동기 파이프라인 격리 위임 트리거
   * @returns 202 Accepted 영수증 데이터 수신
   */
  async createBookmark(payload: BookmarkPayload): Promise<any> {
    const response = await fetch("http://localhost:8000/api/bookmarks", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    
    if (response.status !== 202) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Gateway Pydantic Validation 에러 발생.");
    }
    return response.json();
  },

  /**
   * 2. v3 최종안 코어 기반 실시간 하이브리드 검색 랭킹 리스트 수신
   * @param query 유저 검색 질의어
   */
  async searchBookmarks(query: string, limit: number = 5): Promise<any[]> {
    const response = await fetch(
      `http://localhost:8000/api/search?q=${encodeURIComponent(query)}&limit=${limit}`
    );
    if (!response.ok) {
      throw new Error("하이브리드 검색 파이프라인 연산 요청에 실패했습니다.");
    }
    return response.json();
  },

  /**
   * 3. FAISS 벡터 유사도 매트릭스 연산 기반 실시간 시맨틱 토폴로지 데이터 수신
   * @param threshold 코사인 유사도 하한 가드라인 (기본값: 0.65)
   */
  async getGraphTopology(threshold: number = 0.85): Promise<GraphTopologyResponse> {
    const response = await fetch(`http://localhost:8000/api/graph?threshold=${threshold}`);
    if (!response.ok) {
      throw new Error("벡터 공간 상의 시맨틱 토폴로지 그래프 맵 동기화에 실패했습니다.");
    }
    return response.json();
  }
};
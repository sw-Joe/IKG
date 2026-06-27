export interface BookmarkPayload {
  url: string;
  title: string;
  content: string;
}

export interface BookmarkIngestPayload {
  url: string;
  title?: string;
  content?: string;
}

export interface GraphNode {
  id: string;
  title: string;
  url: string;
  group?: string; // 도메인 앵커 분기용 확장
}

export interface GraphEdge {
  source: string;
  target: string;
  value: number;
}

export interface GraphTopologyResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  links?: GraphEdge[];
}

const BE_BASE_URL = "http://127.0.0.1:8000";
// const BE_BASE_URL = "http://192.168.0.4:8000";


export const bookmarkService = {
  /**
   * 1. [CREATE] 신규 북마크 접수 및 비동기 파이프라인 격리 위임 트리거
   */
  async createBookmark(payload: BookmarkIngestPayload): Promise<any> {
    const response = await fetch(`${BE_BASE_URL}/api/bookmarks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "자산 선적재 예외 발생.");
    }
    return response.json();
  },

  /**
   * 2. [READ] 코어 기반 실시간 하이브리드 검색 랭킹 리스트 수신 (q 파라미터 규격 통일)
   */
  async searchBookmarks(query: string, limit: number = 5): Promise<any[]> {
    const response = await fetch(
      `${BE_BASE_URL}/api/search?q=${encodeURIComponent(query)}&limit=${limit}`
    );
    if (!response.ok) {
      throw new Error("하이브리드 검색 파이프라인 연산 요청 실패.");
    }
    return response.json();
  },

  /**
   * 3. [READ] 기하학적 공간 시각화 토폴로지 데이터셋 수신
   */
  async getGraphTopology(threshold: number = 0.85): Promise<GraphTopologyResponse> {
    const response = await fetch(`${BE_BASE_URL}/api/graph?threshold=${threshold}`);
    if (!response.ok) {
      throw new Error("공간 토폴로지 데이터 바인딩 실패.");
    }
    const data = await response.json();
    return {
      nodes: data.nodes || [],
      edges: data.edges || data.links || [],
      links: data.links || data.edges || [],
    };
  },

  /**
   * 4. [UPDATE - 신규]: 기존 북마크의 콘텍스트 수정 및 재인덱싱 위임
   */
  async updateBookmark(id: number, payload: BookmarkPayload): Promise<any> {
    const response = await fetch(`${BE_BASE_URL}/api/bookmarks/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "자산 정정 오퍼레이션 실패.");
    }
    return response.json();
  },

  /**
   * 5. [DELETE - 신규]: 단일 북마크 실시간 Soft-Delete 집행
   */
  async deleteBookmark(id: number): Promise<any> {
    const response = await fetch(`${BE_BASE_URL}/api/bookmarks/${id}`, {
      method: "DELETE",
    });
    if (!response.ok) {
      throw new Error("자산 소거 오퍼레이션 트랜잭션 붕괴.");
    }
    return response.json();
  }
};

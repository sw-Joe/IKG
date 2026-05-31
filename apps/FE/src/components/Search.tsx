import React, { useState } from "react";
import { bookmarkService } from "../utils/bookmarkService";
import "./Search.css";

// 하이브리드 검색 결과 데이터 인터페이스 정의
interface SearchResult {
  id: number;
  url: string;
  title: string;
  content: string;
  score: number;
}

interface SearchComponentProps {
  /** 검색된 문서 ID 리스트를 상위 그래프 컴포넌트로 토스하여 노드를 하이라이트하기 위한 콜백 */
  onSearchComplete: (matchingIds: string[]) => void;
  /** 검색어 클리어 시 그래프 상태를 원복하기 위한 콜백 */
  onSearchClear: () => void;
}

export const Search: React.FC<SearchComponentProps> = ({ onSearchComplete, onSearchClear }) => {
  const [query, setQuery] = useState<string>("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.strip || !query.trim()) {
      handleClear();
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // 1. 백엔드 v3 최종안 레이어 분리형 하이브리드 랭킹 엔진 호출
      const searchData = await bookmarkService.searchBookmarks(query.trim(), 5);
      setResults(searchData);

      // 2. 검색 결과로 나온 문서들의 ID 배열 추출 후 문자열 캐스팅
      const matchingIds = searchData.map((item: SearchResult) => String(item.id));
      
      // 3. 상위 그래프 컴포넌트로 ID를 공유하여 D3/Force-Graph 노드 동적 하이라이트 유도
      onSearchComplete(matchingIds);
    } catch (err: any) {
      console.error("[SEARCH COMPONENT ERROR]", err);
      setError(err.message || "하이브리드 랭킹 연산 중 장애가 발생했습니다.");
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setQuery("");
    setResults([]);
    setError(null);
    onSearchClear();
  };

  return (
    <div className="hybrid-search-container">
      {/* 하이브리드 검색 입력 섹션 */}
      <form onSubmit={handleSearch} className="search-form">
        <input
          type="text"
          className="search-input"
          placeholder="인드레이싱된 지식 자산 하이브리드 검색 (v3 Core)..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={loading}
        />
        {query && (
          <button type="button" className="clear-button" onClick={handleClear}>
            ✕
          </button>
        )}
        <button type="submit" className="search-submit-btn" disabled={loading}>
          {loading ? "연산 중..." : "검색"}
        </button>
      </form>

      {/* 에러 피드백 레이어 */}
      {error && <div className="search-error-message">{error}</div>}

      {/* 하이브리드 결과 랭킹 리스트 렌더링 섹션 */}
      {results.length > 0 && (
        <div className="search-results-panel">
          <h3 className="panel-title">하이브리드 랭킹 매칭 결과 (Top 5)</h3>
          <ul className="results-list">
            {results.map((item, index) => (
              <li key={item.id} className="result-item">
                <div className="result-rank-badge">{index + 1}</div>
                <div className="result-info">
                  <a href={item.url} target="_blank" rel="noopener noreferrer" className="result-title">
                    {item.title}
                  </a>
                  <p className="result-snippet">
                    {item.content.length > 120 ? `${item.content.substring(0, 120)}...` : item.content}
                  </p>
                  <div className="result-meta">
                    <span className="meta-score">융합 스코어: {item.score.toFixed(4)}</span>
                    <span className="meta-url">{item.url}</span>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Zero-Hits 검문소 필터링 피드백 */}
      {query.trim() && !loading && results.length === 0 && !error && (
        <div className="search-no-results">
          최외각 랭크 검문소 통과 실패: 질의와 매칭되는 유효 지식 자산이 존재하지 않습니다 (Zero-Hits).
        </div>
      )}
    </div>
  );
};
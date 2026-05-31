// apps/FE/src/components/Search.tsx
import React, { useState } from "react";
import "./Search.css";

interface SearchResult {
  id: number;
  url: string;
  title: string;
  content: string;
  score: number;
  score_lex_raw: number;
  score_sem_raw: number;
}

interface SearchComponentProps {
  onSearchComplete: (matchingIds: string[]) => void;
  onSearchClear: () => void;
}

export const Search: React.FC<SearchComponentProps> = ({ onSearchComplete, onSearchClear }) => {
  const [query, setQuery] = useState<string>("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleServerSearchExecute = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query || !query.trim()) {
      handleClear();
      return;
    }

    setLoading(true);
    setError(null);
    console.log(`[FE INFERENCE NETWORK] 하이브리드 추론 질의 가동 -> '${query.trim()}'`);

    try {
      const response = await fetch(
        `http://127.0.0.1:8000/api/search?query=${encodeURIComponent(query.trim())}&limit=5`,
        {
          method: "GET",
          headers: {
            "Accept": "application/json",
            "Content-Type": "application/json"
          }
        }
      );

      if (!response.ok) {
        throw new Error(`백엔드 서버 장애 수신: 상태 코드 ${response.status}`);
      }

      const searchData = await response.json();
      setResults(searchData);

      if (searchData.length > 0) {
        onSearchComplete(searchData.map((item: SearchResult) => String(item.id)));
      } else {
        onSearchComplete([]);
      }
    } catch (err: any) {
      console.error("❌ [E2E CONNECTION ERROR] API 엔드포인트 수렴 단절:", err);
      setError("AI 하이브리드 검색 인퍼런스 파이프라인 장애");
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
    <div className="hybrid-search-core-wrapper">
      <form onSubmit={handleServerSearchExecute} className="search-form-layout">
        <div className="search-input-relative-container">
          <input
            type="text"
            className="search-input-field"
            placeholder="지식베이스 하이브리드 검색어 입력..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
          {query && (
            <button type="button" className="inner-clear-button" onClick={handleClear}>
              ✕
            </button>
          )}
        </div>
        <button type="submit" className="search-trigger-button" disabled={loading}>
          {loading ? "연산중" : "검색"}
        </button>
      </form>

      {error && <div className="search-error-layer">{error}</div>}

      {/* 사이드바 내부 고밀도 적재형 카드 렌더링 레이어 */}
      {results.length > 0 && (
        <div className="search-results-viewport-panel">
          <h3 className="panel-headline-title">하이브리드 랭킹 결과 (Top 5)</h3>
          <ul className="results-list-stack">
            {results.map((item, index) => (
              <li key={item.id} className="result-item-card">
                <div className="card-top-identity-row">
                  <div className="result-rank-badge-node">{index + 1}</div>
                  <a href={item.url} target="_blank" rel="noopener noreferrer" className="result-title-link">
                    {item.title}
                  </a>
                </div>
                <p className="result-content-snippet">
                  {item.content.length > 105 ? `${item.content.substring(0, 105)}...` : item.content}
                </p>
                <div className="card-bottom-metrics-grid">
                  <span className="metric-badge-final">융합 스코어: {item.score.toFixed(4)}</span>
                  <div className="sub-metrics-row">
                    <span>ID: #{item.id}</span>
                    <span>|</span>
                    <span>시맨틱: {item.score_sem_raw.toFixed(3)}</span>
                    <span>|</span>
                    <span>렉시컬: {item.score_lex_raw.toFixed(1)}</span>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {query.trim() && !loading && results.length === 0 && !error && (
        <div className="zero-hits-placeholder-layer">
          ❌ 가중치 임계치를 충족하는 임베딩 노드가 공간 상에 존재하지 않습니다.
        </div>
      )}
    </div>
  );
};
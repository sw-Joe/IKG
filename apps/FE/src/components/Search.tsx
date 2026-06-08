import React, { useState } from "react";
import { bookmarkService } from "../utils/bookmarkService";
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

  // 인라인 정정 수정을 위한 로컬 폼 상태 머신
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [editContent, setEditContent] = useState("");

  const handleServerSearchExecute = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query || !query.trim()) {
      handleClear();
      return;
    }
    setLoading(true);
    setError(null);

    try {
      // 보정된 통신 인터페이스 통합 관측 호출
      const data = await bookmarkService.searchBookmarks(query.trim());
      setResults(data);
      const matchingIds = data.map((item) => String(item.id));
      onSearchComplete(matchingIds);
    } catch (err: any) {
      setError(err.message || "하이브리드 엔진 랭킹 연산 오류");
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

  // [DELETE ACTION]: 데이터베이스 논리 소거 트리거 및 클라이언트 즉시 반영
  const handleDeleteClick = async (id: number) => {
    if (!window.confirm("이 지식 자산을 공간에서 영구 삭제하시겠습니까?")) return;
    try {
      await bookmarkService.deleteBookmark(id);
      // UI 스레드에서 즉시 카드를 제거하여 Immediate Consistency 확보
      setResults(prev => prev.filter(item => item.id !== id));
    } catch (err: any) {
      alert(`소거 실패: ${err.message}`);
    }
  };

  // [UPDATE ACTION]: 수정 폼 활성화 워크플로우
  const handleEditInit = (item: SearchResult) => {
    setEditingId(item.id);
    setEditTitle(item.title);
    setEditContent(item.content);
  };

  const handleUpdateSubmit = async (id: number, originalUrl: string) => {
    try {
      await bookmarkService.updateBookmark(id, {
        url: originalUrl,
        title: editTitle,
        content: editContent
      });
      alert("정정 요청이 비동기 큐에 접수되었습니다.");
      setResults(prev => prev.map(item => item.id === id ? { ...item, title: editTitle, content: editContent } : item));
      setEditingId(null);
    } catch (err: any) {
      alert(`정정 실패: ${err.message}`);
    }
  };

  return (
    <div className="search-wrapper-context">
      <form onSubmit={handleServerSearchExecute} className="search-form-row">
        <input
          type="text"
          className="search-input-field"
          placeholder="기술 자산 차원 질의어 입력..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit" className="search-submit-btn">검색</button>
        {query && <button type="button" onClick={handleClear} className="search-clear-btn">X</button>}
      </form>

      {loading && <div className="status-info-label">ONNX 임베딩 질의 분석 매트릭스 계산 중...</div>}
      {error && <div className="status-error-label">⚠ {error}</div>}

      {!loading && results.length > 0 && (
        <div className="results-viewport-area">
          <h3 className="panel-headline-title">하이브리드 랭킹 결과 (Top 5)</h3>
          <ul className="results-list-stack">
            {results.map((item, index) => (
              <li key={item.id} className="result-item-card">
                {editingId === item.id ? (
                  <div className="edit-form-container text-slate-200 flex flex-col gap-2 p-2">
                    <input type="text" className="bg-slate-800 p-1 rounded border border-slate-700 text-xs" value={editTitle} onChange={e => setEditTitle(e.target.value)} />
                    <textarea className="bg-slate-800 p-1 rounded border border-slate-700 text-xs h-20" value={editContent} onChange={e => setEditContent(e.target.value)} />
                    <div className="flex gap-2 justify-end text-xs">
                      <button onClick={() => handleUpdateSubmit(item.id, item.url)} className="bg-emerald-600 px-2 py-1 rounded">저장</button>
                      <button onClick={() => setEditingId(null)} className="bg-slate-700 px-2 py-1 rounded">취소</button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="card-top-identity-row flex justify-between items-start">
                      <div className="flex items-center gap-2">
                        <div className="result-rank-badge-node">{index + 1}</div>
                        <a href={item.url} target="_blank" rel="noopener noreferrer" className="result-title-link">
                          {item.title}
                        </a>
                      </div>
                      {/* CRUD 조작 제어부 컴포넌트 결합 */}
                      <div className="flex gap-1.5 text-xs text-slate-400">
                        <button onClick={() => handleEditInit(item)} className="hover:text-blue-400">수정</button>
                        <span>|</span>
                        <button onClick={() => handleDeleteClick(item.id)} className="hover:text-rose-400">삭제</button>
                      </div>
                    </div>
                    <p className="result-content-snippet">
                      {item.content.length > 105 ? `${item.content.substring(0, 105)}...` : item.content}
                    </p>
                    <div className="card-bottom-metrics-grid">
                      <span className="metric-badge-final">융합 스코어: {item.score.toFixed(4)}</span>
                    </div>
                  </>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
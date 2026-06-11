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
      const data = await bookmarkService.searchBookmarks(query.trim());
      setResults(data);
      const matchingIds = data.map((item) => String(item.id));
      onSearchComplete(matchingIds);
    } catch (err: any) {
      console.error("[SEARCH RUNTIME ERROR]", err);
      setError(err.message || "검색 연산 중 예외가 발생했습니다.");
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

  // 인라인 수정 진입 제어 파이프라인
  const handleEditInit = (item: SearchResult) => {
    setEditingId(item.id);
    setEditTitle(item.title);
    setEditContent(item.content);
  };

  const handleEditCancel = () => {
    setEditingId(null);
    setEditTitle("");
    setEditContent("");
  };

  const handleEditSave = async (id: number) => {
    try {
      const targetItem = results.find((r) => r.id === id);
      if (!targetItem) return;

      await bookmarkService.updateBookmark(id, {
        url: targetItem.url,
        title: editTitle,
        content: editContent,
      });

      // 로컬 메모리 상태 즉시 갱신
      setResults((prev) =>
        prev.map((item) =>
          item.id === id ? { ...item, title: editTitle, content: editContent } : item
        )
      );
      setEditingId(null);
    } catch (err: any) {
      alert(`정정 반영 실패: ${err.message}`);
    }
  };

  const handleDeleteClick = async (id: number) => {
    if (!window.confirm("해당 지식 자산을 원격 매트릭스에서 영구 커팅하시겠습니까?")) return;
    try {
      await bookmarkService.deleteBookmark(id);
      setResults((prev) => prev.filter((item) => item.id !== id));
    } catch (err: any) {
      alert(`삭제 실패: ${err.message}`);
    }
  };

  return (
    <div className="w-full flex flex-col">
      {/* 검색 바 입력 폼 시스템 제어부 */}
      <form onSubmit={handleServerSearchExecute} className="search-form-wrapper">
        <input
          type="text"
          className="search-input-field"
          placeholder="지식 매트릭스 검색어 입력..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </form>

      {loading && <div className="text-xs text-blue-400 py-2 animate-pulse">엔진 하이브리드 추론 중...</div>}
      {error && <div className="text-xs text-rose-400 py-2">{error}</div>}

      {!loading && results.length > 0 && (
        <div className="results-container-block">
          <div className="flex justify-between items-center mb-1">
            <span className="text-[11px] text-slate-500">인덱싱 검색 결과 {results.length}건</span>
            <button onClick={handleClear} className="text-[11px] text-slate-400 hover:text-white underline">
              초기화
            </button>
          </div>

          <ul className="results-list-wrapper">
            {results.map((item, index) => (
              <li key={item.id} className="result-item-card">
                {editingId === item.id ? (
                  /* 새롭게 디자인이 바인딩된 인라인 수정 폼 */
                  <div className="edit-form-zone">
                    <input
                      type="text"
                      className="edit-input-field"
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                    />
                    <textarea
                      className="edit-textarea-field"
                      value={editContent}
                      onChange={(e) => setEditContent(e.target.value)}
                    />
                    <div className="edit-actions-row">
                      <button onClick={() => handleEditSave(item.id)} className="btn-action-save">
                        저장
                      </button>
                      <button onClick={handleEditCancel} className="btn-action-cancel">
                        취소
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="card-top-identity-row flex justify-between items-start">
                      <div className="flex items-center gap-2 overflow-hidden">
                        <div className="result-rank-badge-node flex-shrink-0">{index + 1}</div>
                        <a href={item.url} target="_blank" rel="noopener noreferrer" className="result-title-link">
                          {item.title}
                        </a>
                      </div>
                      <div className="flex gap-1.5 text-[11px] text-slate-500 flex-shrink-0 ml-2">
                        <button onClick={() => handleEditInit(item)} className="hover:text-blue-400 transition-colors">수정</button>
                        <span>|</span>
                        <button onClick={() => handleDeleteClick(item.id)} className="hover:text-rose-400 transition-colors">삭제</button>
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
// apps/FE/src/App.jsx
import { useState } from 'react';
import { BookmarkGraph } from './components/BookmarkGraph';
import { SideBar } from './components/SideBar';

function App() {
  const [highlightIds, setHighlightIds] = useState([]);

  const handleSearchComplete = (matchingIds) => {
    console.log("[APP MATRIX LOG] AI 하이브리드 검색 매칭 노드 인입:", matchingIds);
    setHighlightIds(matchingIds);
  };

  const handleSearchClear = () => {
    console.log("[APP MATRIX LOG] 컨텍스트 하이라이트 인덱스 세션 초기화");
    setHighlightIds([]);
  };

  // apps/FE/src/App.jsx - 리턴 컴포넌트 내 레이아웃 클래스 정정

  return (
    <div className="app-viewport-container bg-slate-900"> {/* [변경] bg-black -> bg-slate-900 */}
      {/* 백그라운드 스팟 조명도 블루-슬레이트 톤에 맞게 미세 조정 */}
      <div className="absolute top-[-10%] left-[50%] -translate-x-1/2 w-[1000px] h-[350px] bg-blue-500/5 blur-[130px] rounded-full pointer-events-none z-0" />

      {/* 1. 최상단 고정 헤더 (Header) */}
      {/* [변경]: bg-zinc-950/80 -> bg-slate-950/80, border-zinc-800 -> border-slate-800 */}
      <header className="w-full h-16 border-b border-slate-800 bg-slate-950/80 backdrop-blur-md flex items-center justify-between px-6 z-20 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.6)]" />
          <h1 className="text-sm font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-400 select-none">
            Intelligent Knowledge Graphing
          </h1>
        </div>
        {/* ... 결과 스탬프 배지 영역 동일 ... */}
      </header>

      {/* 2. 하단 레이아웃 메인 수평 격리 구역 (main) */}
      <main className="main-content-split-zone">
        <SideBar 
          onSearchComplete={handleSearchComplete} 
          onSearchClear={handleSearchClear} 
        />

        {/* 우측 독립형 토폴로지 지식 그래프 캔버스 보드 */}
        {/* [변경]: border-zinc-900 -> border-slate-900 */}
        <section className="graph-canvas-section border-l border-slate-900">
          <BookmarkGraph highlightNodes={highlightIds} />
        </section>
      </main>
    </div>
  );
}
export default App;
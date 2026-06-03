import { useDeferredValue, useState } from 'react';
import { BookmarkGraph } from './components/BookmarkGraph';
import { SideBar } from './components/SideBar';

function App() {
  const [highlightIds, setHighlightIds] = useState([]);
  
  // [최적화 2단계]: 렌더링 우선순위 조정 (상태 업데이트 지연)
  // 검색 결과에 따른 그래프 포커싱 연산을 백그라운드로 지연시켜 사용자 타이핑 응답성을 확보합니다.
  const deferredHighlightIds = useDeferredValue(highlightIds);

  const handleSearchComplete = (matchingIds) => {
    console.log("[APP MATRIX LOG] AI 하이브리드 검색 매칭 노드 인입:", matchingIds);
    setHighlightIds(matchingIds);
  };

  const handleSearchClear = () => {
    console.log("[APP MATRIX LOG] 컨텍스트 하이라이트 인덱스 세션 초기화");
    setHighlightIds([]);
  };

  return (
    <div className="app-viewport-container bg-slate-900">
      <div className="absolute top-[-10%] left-[50%] -translate-x-1/2 w-[1000px] h-[350px] bg-blue-500/5 blur-[130px] rounded-full pointer-events-none z-0" />

      <header className="w-full h-16 border-b border-slate-800 bg-slate-950/80 backdrop-blur-md flex items-center justify-between px-6 z-20 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.6)]" />
          <h1 className="text-sm font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-400 select-none">
            Intelligent Knowledge Graphing
          </h1>
        </div>
      </header>

      <main className="main-content-split-zone">
        <SideBar 
          onSearchComplete={handleSearchComplete} 
          onSearchClear={handleSearchClear} 
        />

        <section className="graph-canvas-section border-l border-slate-900">
          {/* 지연된 상태값을 하위 캔버스 컴포넌트에 주입 */}
          <BookmarkGraph highlightNodes={deferredHighlightIds} />
        </section>
      </main>
    </div>
  );
}
export default App;
// apps/FE/src/components/SideBar.tsx
import React from "react";
import { Search } from "./Search";
import "./Search.css";

interface SideBarProps {
  onSearchComplete: (matchingIds: string[]) => void;
  onSearchClear: () => void;
}

export const SideBar: React.FC<SideBarProps> = ({ onSearchComplete, onSearchClear }) => {
  return (
    <aside className="ikg-sidebar-container">
      <div className="sidebar-header-zone">
        <h2 className="sidebar-title">지식 매트릭스 탐색기</h2>
        <p className="sidebar-subtitle">v3 AI Hybrid Vector Core</p>
      </div>
      <div className="sidebar-content-scroll-zone">
        {/* 하이브리드 추론용 단일 입력창 컴포넌트 내장 바인딩 */}
        <Search onSearchComplete={onSearchComplete} onSearchClear={onSearchClear} />
      </div>
    </aside>
  );
};
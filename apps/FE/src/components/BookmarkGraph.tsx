import ForceGraph from 'force-graph';
import React, { memo, useEffect, useRef, useState } from 'react';
import { bookmarkService, GraphEdge, GraphNode } from '../utils/bookmarkService';

interface BookmarkGraphProps {
  highlightNodes: string[];
}

// [최적화 2단계]: 컴포넌트 메모이제이션
// props(highlightNodes)가 얕은 비교(Shallow Compare) 기준으로 변경되지 않는 한 리렌더링을 방지합니다.
export const BookmarkGraph: React.FC<BookmarkGraphProps> = memo(({ highlightNodes }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphInstanceRef = useRef<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [rawNodes, setRawNodes] = useState<GraphNode[]>([]);
  const [rawEdges, setRawEdges] = useState<GraphEdge[]>([]);

  useEffect(() => {
    bookmarkService.getGraphTopology()
      .then((data) => {
        setRawNodes(data.nodes);
        setRawEdges(data.edges);
        setLoading(false);
      })
      .catch((err) => {
        console.error("토폴로지 인입 실패:", err);
        setLoading(false);
      });
  }, []);

  // D3 런타임 공간 수렴 빌드 파이프라인
  useEffect(() => {
    if (!containerRef.current || loading || rawNodes.length === 0) return;

    if (!graphInstanceRef.current) {
      const graph = ForceGraph()(containerRef.current)
        .graphData({ nodes: rawNodes, links: rawEdges })
        .nodeId('id')
        .nodeVal((node: any) => node.group === 'folder' ? 7 : 3.5)
        .nodeColor((node: any) => {
          if (node.group === 'folder') return '#60a5fa'; 
          return '#94a3b8'; 
        })
        .nodeLabel('title')
        .linkColor(() => 'rgba(148, 163, 184, 0.12)')
        .linkWidth(1)
        .backgroundColor('#1e293b');
        // 1단계 최적화 코드가 있다면 여기에 병합 유지 (e.g., .cooldownTicks(150))

      graphInstanceRef.current = graph;
    }
  }, [loading, rawNodes, rawEdges]);

  // 상위 단일 검색창과의 포커싱 인터록 데이터 정렬
  useEffect(() => {
    if (!graphInstanceRef.current || !highlightNodes || highlightNodes.length === 0) return;

    const target = rawNodes.find((n: any) => String(n.id) === highlightNodes[0]);
    if (target) {
      graphInstanceRef.current.centerAt(target.x, target.y, 1000);
      graphInstanceRef.current.zoom(3.8, 1000);
    }
  }, [highlightNodes, rawNodes]);

  if (loading) {
    return (
      <div className="w-full h-full flex items-center justify-center text-zinc-500 font-mono text-xs bg-[#040406]">
        [SYSTEM] FAISS 밀집 벡터 공간 구조 토폴로지 동적 분석 중...
      </div>
    );
  }

  return (
    <div 
      ref={containerRef} 
      className="w-full h-full relative overflow-hidden"
    />
  );
});

// 디버깅 용이성을 위한 DisplayName 명시
BookmarkGraph.displayName = 'BookmarkGraph';
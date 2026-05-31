// apps/FE/src/components/BookmarkGraph.tsx
import ForceGraph from 'force-graph';
import React, { useEffect, useRef, useState } from 'react';
import { bookmarkService, GraphEdge, GraphNode } from '../utils/bookmarkService';

interface BookmarkGraphProps {
  highlightNodes: string[];
}

export const BookmarkGraph: React.FC<BookmarkGraphProps> = ({ highlightNodes }) => {
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
        .nodeVal(4)
        .nodeLabel('title')
        .linkColor(() => 'rgba(255, 255, 255, 0.08)')
        .linkWidth(1)
        .backgroundColor('#303841'); // 요구사항 반영: 칠흑색 기조 정형화

      graphInstanceRef.current = graph;
    } else {
      graphInstanceRef.current.graphData({ nodes: rawNodes, links: rawEdges });
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

  // [교정 완결]: 기존 인라인 100vw, 100vh 절대 수치를 박멸하고 부모 플렉스 구역을 100% 완벽히 추종하도록 격리
  return (
    <div 
      ref={containerRef} 
      className="w-full h-full relative overflow-hidden"
    />
  );
};
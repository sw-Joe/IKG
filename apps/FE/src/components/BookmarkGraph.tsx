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

    // apps/FE/src/components/BookmarkGraph.tsx 수정안 스니펫

    if (!graphInstanceRef.current) {
    const graph = ForceGraph()(containerRef.current)
        .graphData({ nodes: rawNodes, links: rawEdges })
        .nodeId('id')
        // 1. 그룹별 노드 크기 차등화 (폴더 가중치 부여)
        .nodeVal((node: any) => node.group === 'folder' ? 7 : 3.5)
        // 2. 그룹별/상태별 노드 색상 분기
        .nodeColor((node: any) => {
        if (node.group === 'folder') return '#60a5fa'; // Slate Blue
        return '#94a3b8'; // Muted Slate
        })
        .nodeLabel('title')
        // 3. 엣지 가시성 최적화 (연한 그리드 톤)
        .linkColor(() => 'rgba(148, 163, 184, 0.12)')
        .linkWidth(1)
        // 4. 전역 테마 변수 대역에 맞춘 배경색 단일화
        .backgroundColor('#1e293b'); 

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

  // [교정 완결]: 기존 인라인 100vw, 100vh 절대 수치를 제거, 부모 플렉스 구역을 100% 완벽히 추종하도록 격리
  return (
    <div 
      ref={containerRef} 
      className="w-full h-full relative overflow-hidden"
    />
  );
};
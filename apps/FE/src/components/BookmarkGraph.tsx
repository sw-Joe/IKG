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
        .nodeVal((node: any) => node.group === 'folder' ? 7 : 3.5)
        .nodeColor((node: any) => {
          if (node.group === 'folder') return '#60a5fa';
          return '#94a3b8';
        })
        .nodeLabel('title')
        .linkColor(() => 'rgba(148, 163, 184, 0.12)')
        .linkWidth(1)
        .backgroundColor('#1e293b')
        // [최적화 1단계: 물리 엔진 연산 제한]
        .cooldownTicks(150) // 150 틱(초기 레이아웃 수렴에 충분한 수준) 이후 연산 강제 종료
        .onEngineStop(() => {
          console.info("[SYSTEM] 토폴로지 레이아웃 수렴 완료. Force 엔진 정지 및 CPU 리소스 해제.");
          // 초기 렌더링 시 전체 그래프가 화면에 꽉 차도록 뷰포트 자동 정렬이 필요한 경우 아래 주석 해제
          // graph.zoomToFit(400, 20); 
        });

      // (선택적 Edge Case 대응): 엔진 정지 후 사용자가 노드를 드래그할 때 
      // 주변 노드들이 다시 탄성적으로 반응하게 하려면 시뮬레이션을 일시적으로 재가열(Reheat)해야 함
      /*
      graph.onNodeDrag(() => {
        // 드래그 중 물리 엔진 임시 활성화
        graph.d3ReheatSimulation(); 
      });
      */

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
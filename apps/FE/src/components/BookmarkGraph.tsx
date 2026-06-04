// apps/FE/src/components/BookmarkGraph.tsx
import ForceGraph from 'force-graph';
import React, { memo, useEffect, useRef, useState } from 'react';
import { bookmarkService, GraphEdge, GraphNode } from '../utils/bookmarkService';
import { drawNodeElement } from '../utils/graphCanvasRenderer';
import { synthesizeTopology } from '../utils/graphDataProcessor';

interface BookmarkGraphProps {
  highlightNodes: string[];
}


export const BookmarkGraph: React.FC<BookmarkGraphProps> = memo(({ highlightNodes }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphInstanceRef = useRef<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [rawNodes, setRawNodes] = useState<GraphNode[]>([]);
  const [rawEdges, setRawEdges] = useState<GraphEdge[]>([]);

  // 1. 도메인 클러스터 필터 융합형 데이터 수집 세션
  useEffect(() => {
    bookmarkService.getGraphTopology()
      .then((data) => {
        // 모듈 분할된 데이터 전처리 프로세서 가동
        const { nodes, edges } = synthesizeTopology(data.nodes, data.edges);
        setRawNodes(nodes);
        setRawEdges(edges);
        setLoading(false);
      })
      .catch((err) => {
        console.error("토폴로지 인입 실패:", err);
        setLoading(false);
      });
  }, []);

  // 2. D3-Force 그래픽스 인스턴스 오케스트레이션 파이프라인
  useEffect(() => {
    if (!containerRef.current || loading || rawNodes.length === 0) return;

    if (!graphInstanceRef.current) {
      const graph = ForceGraph()(containerRef.current)
        .graphData({ nodes: rawNodes, links: rawEdges })
        .nodeId('id')
        .nodeVal((node: any) => {
          if (node.group === 'domain_anchor') return 10;
          return node.group === 'folder' ? 7 : 3.5;
        })
        .nodeLabel('title') // Hover 시 네이티브 툴팁 브라우저 캐싱 출력용
        .linkColor(() => 'rgba(148, 163, 184, 0.12)')
        .linkWidth(1)
        .backgroundColor('#1e293b')
        
        // 외부 모듈로 갱신된 드로잉 래퍼 매핑
        .nodeCanvasObject((node, ctx, globalScale) => {
          drawNodeElement(node, ctx, globalScale);
        })
        
        // 앵커 클릭 이벤트 처리 부근
        .onNodeClick((node: any) => {
          if (node.group === 'domain_anchor' || node.group === 'folder') return;
          if (node.url) {
            window.open(node.url, '_blank', 'noopener,noreferrer');
          }
        })
        
        // 호버 포인터 상태 전이 처리 부근
        .onNodeHover((node: any) => {
          if (containerRef.current) {
            const isInteractive = node && node.group !== 'domain_anchor';
            containerRef.current.style.cursor = isInteractive ? 'pointer' : 'default';
          }
        });

      // 가상 도메인 구조적 탄성 결합력이 시맨틱 분산 거리를 방해하지 않도록 감쇄율 조율
      graph.d3Force('link')?.strength((link: any) => {
        return link.id?.startsWith('structural_edge:') ? 0.1 : 0.7;
      });

      graphInstanceRef.current = graph;
    }
  }, [loading, rawNodes, rawEdges]);

  // 3. 글로벌 검색 피드백 동기화 파이프라인
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

BookmarkGraph.displayName = 'BookmarkGraph';
// apps/FE/src/components/BookmarkGraph.tsx
import ForceGraph from 'force-graph';
import React, { useEffect, useRef, useState } from 'react';
import { bookmarkService, GraphEdge, GraphNode } from '../utils/bookmarkService';

interface BookmarkGraphProps {
  highlightNodes: string[];
}

// memoization
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

  // D3 런타임 공간 수렴 빌프 파이프라인
  useEffect(() => {
    if (!containerRef.current || loading || rawNodes.length === 0) return;

    if (!graphInstanceRef.current) {
      // 1. 전역 또는 컴포넌트 스코프 상수로 제약 조건 선언
      const MAX_LABEL_LENGTH = 14; 

      const graph = ForceGraph()(containerRef.current)
        .graphData({ nodes: rawNodes, links: rawEdges })
        .nodeId('id')
        .nodeVal((node: any) => node.group === 'folder' ? 7 : 3.5)
        .nodeLabel('title') // Hover 시 툴팁에는 전체 원본 이름이 나오도록 유지
        .linkColor(() => 'rgba(148, 163, 184, 0.12)')
        .linkWidth(1)
        .backgroundColor('#1e293b')

        // [기능 1]: 텍스트 길이 제한 제어 (3단계 LOD 로직 확장)
        .nodeCanvasObject((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
          const radius = node.group === 'folder' ? 7 : 3.5;
          
          // 노드 중심원 드로잉
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
          ctx.fillStyle = node.group === 'folder' ? '#60a5fa' : '#94a3b8';
          ctx.fill();

          const LOD_ZOOM_THRESHOLD = 2.0;
          if (globalScale >= LOD_ZOOM_THRESHOLD) {
            const rawLabel = node.title || '';
            
            // 글자 수 제한 및 말줄임표(...) 처리 파이프라인
            const truncatedLabel = rawLabel.length > MAX_LABEL_LENGTH
              ? rawLabel.substring(0, MAX_LABEL_LENGTH) + '...'
              : rawLabel;

            const fontSize = 11 / globalScale;
            ctx.font = `500 ${fontSize}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillStyle = 'rgba(241, 245, 249, 0.85)';
            ctx.fillText(truncatedLabel, node.x, node.y + radius + (1.5 / globalScale));
          }
        })

        // [기능 2]: 클릭 인터록 라우팅 (Anchor 역할 수행)
        .onNodeClick((node: any) => {
          if (node.group === 'folder') {
            console.info(`[ROUTING] 폴더 노드 클릭: ${node.title} (내부 컨텍스트 확장 가능 공간)`);
            // 내부 상태 인터록 혹은 폴더 깊이 탐색 트리거 배치 구역
            return;
          }

          // 북마크 노드일 경우 실제 URL 필드 검증 후 탭 전환 개방
          if (node.url) {
            window.open(node.url, '_blank', 'noopener,noreferrer');
          } else {
            console.warn(`[DATA WARNING] 해당 노드에 맵핑된 유효한 URL 스키마가 존재하지 않습니다. ID: ${node.id}`);
          }
        })

        // [기능 3]: 인터랙션 시각화 제어 (Hover 시 커서 포인터 전환)
        .onNodeHover((node: any) => {
          if (containerRef.current) {
            // 노드 위에 마우스가 올라갔을 때만 클릭 가능한 Anchor 형태의 포인터로 스타일 변경
            containerRef.current.style.cursor = node ? 'pointer' : 'default';
          }
        });

      graphInstanceRef.current = graph;
    }
  }, [loading, rawNodes, rawEdges]);

  // 상위 단일 검색창과의 포커싱 인터록 데이터 정렬
  useEffect(() => {
    if (!graphInstanceRef.current || !highlightNodes || highlightNodes.length === 0) return;

    const target = rawNodes.find((n: any) => String(n.id) === highlightNodes[0]);
    if (target) {
      graphInstanceRef.current.centerAt(target.x, target.y, 1000);
      graphInstanceRef.current.zoom(3.8, 1000); // 줌 배율이 3.8로 설정되므로 검색 포커싱 시에는 LOD 라벨이 무조건 노출됨
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
};
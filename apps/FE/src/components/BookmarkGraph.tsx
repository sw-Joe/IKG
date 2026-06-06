import ForceGraph from 'force-graph';
import React, { memo, useEffect, useMemo, useRef, useState } from 'react';
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
  const [backendRawNodes, setBackendRawNodes] = useState<GraphNode[]>([]);
  const [backendRawEdges, setBackendRawEdges] = useState<GraphEdge[]>([]);
  const [expandedDomains, setExpandedDomains] = useState<Set<string>>(new Set());

  // 1. 오리지널 토폴로지 데이터 비동기 인입 파이프라인
  useEffect(() => {
    bookmarkService.getGraphTopology()
      .then((data) => {
        setBackendRawNodes(data.nodes);
        setBackendRawEdges(data.edges);
        setLoading(false);
      })
      .catch((err) => {
        console.error("토폴로지 인입 실패:", err);
        setLoading(false);
      });
  }, []);

  // 2. 외부 모듈 가동 구역 (useMemo 내부에서 외부 함수 결합 및 런타임 픽셀 좌표 바인딩)
  const processedData = useMemo(() => {
    if (backendRawNodes.length === 0) return { nodes: [], links: [] };

    const coordMap = new Map<string, { x: number; y: number; vx: number; vy: number }>();
    if (graphInstanceRef.current) {
      const currentNodes = graphInstanceRef.current.graphData().nodes;
      currentNodes.forEach((n: any) => {
        if (n.x !== undefined) {
          coordMap.set(String(n.id), { x: n.x, y: n.y, vx: n.vx, vy: n.vy });
        }
      });
    }

    // 모듈 분할 함수 호출로 일체화 및 간소화 완료
    return synthesizeTopology(backendRawNodes, backendRawEdges, expandedDomains, coordMap);
  }, [backendRawNodes, backendRawEdges, expandedDomains]);

  // 3. 글로벌 검색 시 닫힌 도메인 계층 구조 강제 개방 가드
  useEffect(() => {
    if (!highlightNodes || highlightNodes.length === 0 || backendRawNodes.length === 0) return;

    const targetNodeId = highlightNodes[0];
    const targetNode = backendRawNodes.find((n: any) => String(n.id) === targetNodeId);

    if (targetNode && targetNode.url) {
      try {
        const urlObj = new URL(targetNode.url);
        const domain = urlObj.hostname.replace('www.', '');
        const parentDomainId = `domain_anchor:${domain}`;

        if (!expandedDomains.has(parentDomainId)) {
          setExpandedDomains((prev) => {
            const next = new Set(prev);
            next.add(parentDomainId);
            return next;
          });
        }
      } catch (e) {
        // 예외 스킵
      }
    }
  }, [highlightNodes, backendRawNodes]);

  // 4. [인스턴스 격리 아키텍처]: 오직 최초 1회만 그래픽스 본체를 DOM에 인입 및 바인딩
  useEffect(() => {
    if (!containerRef.current || loading || backendRawNodes.length === 0) return;

    if (!graphInstanceRef.current) {
      const graph = ForceGraph()(containerRef.current)
        .nodeId('id')
        .nodeLabel('title')
        .linkColor(() => 'rgba(148, 163, 184, 0.12)')
        .linkWidth(1)
        .backgroundColor('#1e293b')
        
        // 외부 렌더러 모듈 적용 연동 완료
        .nodeCanvasObject((node, ctx, globalScale) => {
          drawNodeElement(node, ctx, globalScale);
        })
        
        .onNodeClick((node: any) => {
          if (node.group === 'domain_anchor') {
            setExpandedDomains((prev) => {
              const next = new Set(prev);
              if (next.has(node.id)) {
                next.delete(node.id);
              } else {
                next.add(node.id);
              }
              return next;
            });
            return;
          }
          if (node.group === 'folder') return;
          if (node.url) {
            window.open(node.url, '_blank', 'noopener,noreferrer');
          }
        })
        .onNodeHover((node: any) => {
          if (containerRef.current) {
            containerRef.current.style.cursor = node ? 'pointer' : 'default';
          }
        });

      // 저스펙 디바이스 방어형 수렴 파라미터 고정 초기화
      graph.d3Force('link')?.strength((link: any) => {
        return link.id?.startsWith('structural_edge:') ? 0.02 : 0.07;
      });
      graph.d3Force('charge')?.strength(-160).distanceMax(250);
      graph.d3Force('center')?.strength(0.05);
      graph.d3AlphaMin(0.005);

      graphInstanceRef.current = graph;
    }

    // [버그 해결 핵심]: 데이터의 동적 업데이트 시 DOM을 파괴하는 클린업을 제거하고 오직 언마운트 시에만 클린업을 집행하도록 구조적 분리
    return () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = '';
        graphInstanceRef.current = null;
      }
    };
  }, [loading]); // backendRawNodes의 최초 로딩 수렴 시점에 한해서만 초기 구동되도록 가둠

  // 5. [데이터 주입 파이프라인 분리]: 토폴로지 데이터 유동 갱신 세션 전담 구역
  useEffect(() => {
    if (graphInstanceRef.current && processedData.nodes.length > 0) {
      // 캔버스 엘리먼트를 파괴하지 않고 내부 정점과 링크 정보만 안전하게 동적 스와핑 진행
      graphInstanceRef.current.graphData(processedData);
      graphInstanceRef.current.d3ReheatSimulation();
    }
  }, [processedData]);

  // 6. 검색 포커싱 피드백 인터록
  useEffect(() => {
    if (!graphInstanceRef.current || !highlightNodes || highlightNodes.length === 0 || loading) return;

    const currentNodes = graphInstanceRef.current.graphData().nodes;
    const target = currentNodes.find((n: any) => String(n.id) === highlightNodes[0]);
    
    if (target) {
      setTimeout(() => {
        graphInstanceRef.current.centerAt(target.x, target.y, 800);
        graphInstanceRef.current.zoom(3.8, 800);
      }, 50);
    }
  }, [highlightNodes, loading]);

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
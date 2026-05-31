import ForceGraph from 'force-graph';
import React, { useEffect, useRef, useState } from 'react';
import { bookmarkService, GraphEdge, GraphNode } from '../utils/bookmarkService';
import { Search } from './Search';

export const BookmarkGraph: React.FC = () => {
    const containerRef = useRef<HTMLDivElement>(null);
    const graphInstanceRef = useRef<any>(null);
    const hoveredNodeRef = useRef<any>(null);

    // 인프라 런타임 상태 관리
    const [loading, setLoading] = useState<boolean>(true);
    const [rawNodes, setRawNodes] = useState<GraphNode[]>([]);
    const [rawEdges, setRawEdges] = useState<GraphEdge[]>([]);

    // 하이브리드 검색 필터링 동기화 상태 스냅샷
    const [highlightedNodeIds, setHighlightedNodeIds] = useState<string[]>([]);
    const [isSearching, setIsSearching] = useState<boolean>(false);

    // 1. 초기 웜업: 백엔드 FAISS 임베딩 유사도 토폴로지 엔진 쿼리
    useEffect(() => {
        bookmarkService.getGraphTopology()
            .then((data) => {
                setRawNodes(data.nodes);
                setRawEdges(data.edges);
                setLoading(false);
            })
            .catch((err) => {
                console.error("[GRAPH INITIALIZE ERROR] 토폴로지 인입 실패:", err);
                setLoading(false);
            });
    }, []);

    // 2. 데이터 유입 및 검색 하이라이트 상태 변경 시 D3 렌더 루프 동적 바인딩
    useEffect(() => {
        if (!containerRef.current || rawNodes.length === 0) return;

        // 백엔드 평면 구조 데이터를 ForceGraph 규격 포맷으로 컨버팅
        const graphData = {
            nodes: rawNodes.map(node => ({
                id: node.id,
                title: node.title,
                url: node.url,
                // 검색 중이고 매칭 그룹에 속하면 강조 크기 부여, 아니면 기본 크기
                val: isSearching && highlightedNodeIds.includes(node.id) ? 100 : 15
            })),
            links: rawEdges.map(edge => ({
                source: edge.source,
                target: edge.target,
                value: edge.value
            }))
        };

        // 싱글톤 그래프 인스턴스 초기화 가드
        if (!graphInstanceRef.current) {
            const Graph = ForceGraph()(containerRef.current)
                .backgroundColor('#000000')
                .nodeId('id')
                .nodeLabel('title')
                .linkColor(() => 'rgba(59, 130, 246, 0.24)') // 시맨틱 결합선 파란색 마진 처리
                .linkWidth((link: any) => (link.value || 0.5) * 2)
                .d3AlphaDecay(0.04)
                .d3VelocityDecay(0.3)
                .onNodeClick((node: any) => {
                    if (node.url) {
                        window.open(node.url, '_blank'); // 온디바이스 서핑 보호를 위한 새 탭 처리
                    } else {
                        Graph.centerAt(node.x, node.y, 1000);
                        Graph.zoom(4, 2000);
                    }
                })
                .onNodeHover((node: any) => {
                    containerRef.current!.style.cursor = node ? 'pointer' : null;
                    hoveredNodeRef.current = node;
                });

            graphInstanceRef.current = Graph;
        }

        // 기존 고해상도 커스텀 캔버스 렌더링 루프 자산 보존 및 투명도(Opacity) 가드레일 이식
        graphInstanceRef.current
            .graphData(graphData)
            .nodeCanvasObject((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
                const label = node.title;
                const fontSize = Math.max(10 / globalScale, 4.5);
                ctx.font = `${fontSize}px sans-serif`;

                // [핵심 변경] 검색 상태에 따른 시각적 격리 마스킹 처리 (오컴의 면도날 기법)
                let opacity = 1.0;
                if (isSearching) {
                    opacity = highlightedNodeIds.includes(node.id) ? 1.0 : 0.15;
                }

                // 호버링 링 하이라이트 렌더링
                if (node === hoveredNodeRef.current || (isSearching && highlightedNodeIds.includes(node.id))) {
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, 8, 0, 2 * Math.PI, false);
                    ctx.fillStyle = `rgba(59, 130, 246, ${opacity * 0.4})`;
                    ctx.fill();
                }

                // 코어 노드 포인트 플로팅
                ctx.beginPath();
                ctx.arc(node.x, node.y, 4.5, 0, 2 * Math.PI, false);
                ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
                ctx.fill();

                // 텍스트 라벨 가독성 가드레일
                if (globalScale >= 2.5 || (isSearching && highlightedNodeIds.includes(node.id))) {
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'top';
                    ctx.fillStyle = `rgba(148, 163, 184, ${opacity})`; // slate-400 컬러 명세
                    ctx.fillText(
                        label.length > 15 ? `${label.substring(0, 15)}...` : label, 
                        node.x, 
                        node.y + 7
                    );
                }
            });

    }, [rawNodes, rawEdges, highlightedNodeIds, isSearching]);

    // 하이브리드 검색창 상호 작용 인터페이스 콜백 수식 바인딩
    const handleSearchComplete = (matchingIds: string[]) => {
        setHighlightedNodeIds(matchingIds);
        setIsSearching(true);
        if (graphInstanceRef.current && matchingIds.length > 0) {
            // 검색 매칭 등극 시 인덱스 공간의 첫 번째 최상위 결과 노드로 d3 카메라 부드럽게 무빙
            const targetNode = graphInstanceRef.current.graphData().nodes.find((n: any) => n.id === matchingIds[0]);
            if (targetNode) {
                graphInstanceRef.current.centerAt(targetNode.x, targetNode.y, 1000);
                graphInstanceRef.current.zoom(3.5, 1000);
            }
        }
    };

    const handleSearchClear = () => {
        setHighlightedNodeIds([]);
        setIsSearching(false);
        if (graphInstanceRef.current) {
            graphInstanceRef.current.zoomToFit(1000);
        }
    };

    if (loading) {
        return (
            <div style={{ backgroundColor: '#000000', color: '#94a3b8', width: '100vw', height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '14px', fontFamily: 'sans-serif' }}>
                FAISS 밀집 벡터 공간 구조 토폴로지 동적 분석 중...
            </div>
        );
    }

    return (
        <div className="graph-viewport-wrapper" style={{ position: 'relative', width: '100vw', height: '100vh', overflow: 'hidden' }}>
            {/* 하이브리드 검색창 패널을 시각화 뷰 좌측 상단에 플로팅 레이어로 안전하게 오버레이 */}
            <div className="search-overlay-panel" style={{ position: 'absolute', top: '24px', left: '24px', zIndex: 10 }}>
                <Search 
                    onSearchComplete={handleSearchComplete} 
                    onSearchClear={handleSearchClear} 
                />
            </div>

            {/* D3 Force-Graph Core Mount Element */}
            <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
        </div>
    );
};

export default BookmarkGraph;
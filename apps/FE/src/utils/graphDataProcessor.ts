// apps/FE/src/utils/graphDataProcessor.ts
import { GraphEdge, GraphNode } from './bookmarkService';

export const MIN_CLUSTER_SIZE = 2;

interface DomainGroup {
  domainStr: string;
  children: string[];
}

/**
 * 백엔드 원본 토폴로지 데이터에 도메인 계층 구조를 동적으로 합성합니다.
 * 하위 북마크 노드가 최소 MIN_CLUSTER_SIZE 개 이상인 도메인만 가상 앵커를 생성합니다.
 */
export function synthesizeTopology(nodes: GraphNode[], edges: GraphEdge[]) {
  const synthesizedNodes = [...nodes];
  const synthesizedEdges = [...edges];
  
  // 1st Pass: 도메인별 빈도 측정 및 자식 노드 ID 매핑 집계
  const domainMap = new Map<string, DomainGroup>();

  nodes.forEach((node: any) => {
    if (node.group === 'folder' || !node.url) return;

    try {
      const urlObj = new URL(node.url);
      const domain = urlObj.hostname.replace('www.', '');
      const virtualDomainId = `domain_anchor:${domain}`;

      if (!domainMap.has(virtualDomainId)) {
        domainMap.set(virtualDomainId, { domainStr: domain, children: [] });
      }
      domainMap.get(virtualDomainId)!.children.push(node.id);
    } catch (e) {
      // 정형화되지 않은 가비지 URL 예외 처리 가드
    }
  });

  // 2nd Pass: 임계값(Threshold) 검증 후 조건부 앵커 노드 및 구조적 링크 바인딩
  domainMap.forEach((groupInfo, virtualDomainId) => {
    if (groupInfo.children.length >= MIN_CLUSTER_SIZE) {
      // 가상 부모 도메인 노드 인입
      synthesizedNodes.push({
        id: virtualDomainId,
        title: groupInfo.domainStr,
        group: 'domain_anchor',
        url: ''
      });

      // 자식 리프 노드들과의 구조적 링크 연쇄 주입
      groupInfo.children.forEach((childId) => {
        synthesizedEdges.push({
          source: virtualDomainId,
          target: childId,
          id: `structural_edge:${virtualDomainId}->${childId}`
        });
      });
    }
  });

  return { nodes: synthesizedNodes, edges: synthesizedEdges };
}
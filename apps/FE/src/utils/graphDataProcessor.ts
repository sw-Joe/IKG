// apps/FE/src/utils/graphDataProcessor.ts

export interface ProcessedTopology {
  nodes: any[];
  links: any[]; // [교정]: edges에서 links로 명세 변경
}

const MIN_CLUSTER_SIZE = 2;

/**
 * expandedDomains(활성화된 도메인 ID Set) 및 coordMap(기존 런타임 좌표)을 기반으로 토폴로지를 동적으로 캡슐화합니다.
 */
export function synthesizeTopology(
  rawNodes: any[], 
  rawEdges: any[], 
  expandedDomains: Set<string>,
  coordMap?: Map<string, { x: number; y: number; vx: number; vy: number }>
): ProcessedTopology {
  const finalNodes: any[] = [];
  const finalEdges: any[] = []; // 내부 연산 배열 이름은 유지하되 리턴 구조에서 맵핑
  
  // 1. 폴더 노드 및 도메인 분기용 해시 맵 바인딩 (1st Pass)
  const domainGroupMap = new Map<string, { title: string; children: any[] }>();
  const isolatedNodes: any[] = [];

  rawNodes.forEach((node: any) => {
    if (node.group === 'folder') {
      const preserved = coordMap?.get(String(node.id));
      finalNodes.push({ ...node, ...preserved });
      return;
    }
    
    if (node.group === 'domain_anchor') return;

    if (!node.url) {
      const preserved = coordMap?.get(String(node.id));
      isolatedNodes.push({ ...node, ...preserved });
      return;
    }

    try {
      const urlObj = new URL(node.url);
      const domain = urlObj.hostname.replace('www.', '');
      const virtualDomainId = `domain_anchor:${domain}`;

      if (!domainGroupMap.has(virtualDomainId)) {
        domainGroupMap.set(virtualDomainId, { title: domain, children: [] });
      }
      domainGroupMap.get(virtualDomainId)!.children.push(node);
    } catch (e) {
      const preserved = coordMap?.get(String(node.id));
      isolatedNodes.push({ ...node, ...preserved });
    }
  });

  finalNodes.push(...isolatedNodes);

  // 2. 2nd Pass: 접힘/펼침 세션 스캔 연산 및 부모 시드 기반 좌표 사상
  domainGroupMap.forEach((group, virtualDomainId) => {
    const isExpanded = expandedDomains.has(virtualDomainId);
    const childCount = group.children.length;

    if (childCount >= MIN_CLUSTER_SIZE) {
      const preservedParent = coordMap?.get(virtualDomainId);
      finalNodes.push({
        id: virtualDomainId,
        title: group.title,
        group: 'domain_anchor',
        hiddenCount: isExpanded ? 0 : childCount,
        url: '',
        ...preservedParent
      });

      const parentX = preservedParent?.x ?? 0;
      const parentY = preservedParent?.y ?? 0;

      if (isExpanded) {
        group.children.forEach((childNode) => {
          const preservedChild = coordMap?.get(String(childNode.id));
          finalNodes.push({
            ...childNode,
            x: preservedChild?.x ?? (parentX + (Math.random() - 0.5) * 12),
            y: preservedChild?.y ?? (parentY + (Math.random() - 0.5) * 12),
            vx: preservedChild?.vx ?? 0,
            vy: preservedChild?.vy ?? 0
          });

          finalEdges.push({
            source: virtualDomainId,
            target: childNode.id,
            id: `structural_edge:${virtualDomainId}->${childNode.id}`
          });
        });
      }
    } else {
      group.children.forEach((childNode) => {
        const preservedChild = coordMap?.get(String(childNode.id));
        finalNodes.push({ ...childNode, ...preservedChild });
      });
    }
  });

  // 3. 오리지널 시맨틱 엣지 필터링 (고스트 정점 커팅 규칙)
  const activeNodeIds = new Set(finalNodes.map(n => String(n.id)));
  rawEdges.forEach((edgeInd) => {
    const sourceId = typeof edgeInd.source === 'object' ? String((edgeInd.source as any).id) : String(edgeInd.source);
    const targetId = typeof edgeInd.target === 'object' ? String((edgeInd.target as any).id) : String(edgeInd.target);

    if (activeNodeIds.has(sourceId) && activeNodeIds.has(targetId)) {
      finalEdges.push({
        source: sourceId,
        target: targetId,
        id: edgeInd.id || `semantic_edge:${sourceId}->${targetId}`
      });
    }
  });

  // [교정]: D3 사상을 위해 반드시 'links' 규격으로 변환하여 반환해야 함
  return { nodes: finalNodes, links: finalEdges };
}
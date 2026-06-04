// apps/FE/src/utils/graphCanvasRenderer.ts

const MAX_LABEL_LENGTH = 14;

/**
 * 줌 배율(globalScale) 및 노드 스펙에 따라 Canvas 2D 그래픽을 동적으로 드로잉합니다.
 */
export function drawNodeElement(node: any, ctx: CanvasRenderingContext2D, globalScale: number) {
  let radius = 3.5;
  let fillColor = '#94a3b8';

  // 1. 노드 그룹별 가시적 형상 위상 정의
  if (node.group === 'domain_anchor') {
    radius = 9;
    fillColor = '#10b981'; // 옵시디언 에메랄드 그린 톤
  } else if (node.group === 'folder') {
    radius = 7;
    fillColor = '#60a5fa';
  }

  // 2. 물리 구체 원형 드로잉
  ctx.beginPath();
  ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
  ctx.fillStyle = fillColor;
  ctx.fill();

  // 3. Level of Detail (LOD) 가시성 임계값 연산
  const LOD_ZOOM_THRESHOLD = node.group === 'domain_anchor' ? 0.8 : 2.0;

  if (globalScale >= LOD_ZOOM_THRESHOLD) {
    const rawLabel = node.title || '';
    
    // 일반 북마크 노드에만 선택적으로 글자 수 트렁케이션(말줄임표) 파이프라인 수행
    const truncatedLabel = (node.group !== 'domain_anchor' && rawLabel.length > MAX_LABEL_LENGTH)
      ? rawLabel.substring(0, MAX_LABEL_LENGTH) + '...'
      : rawLabel;

    const fontSize = node.group === 'domain_anchor' ? 13 / globalScale : 11 / globalScale;
    ctx.font = node.group === 'domain_anchor' ? `700 ${fontSize}px sans-serif` : `500 ${fontSize}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = node.group === 'domain_anchor' ? '#10b981' : 'rgba(241, 245, 249, 0.85)';
    
    ctx.fillText(truncatedLabel, node.x, node.y + radius + 2);
  }
}
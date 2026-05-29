import { forceCollide } from 'd3-force';
import ForceGraph from 'force-graph';
import React, { useEffect, useRef } from 'react';
import { GraphData } from '../types';
import { bookmarkService } from '../utils/bookmarkService';

export const BookmarkGraph: React.FC = () => {
    const containerRef = useRef<HTMLDivElement>(null);
    const graphInstanceRef = useRef<any>(null);
    const hoveredNodeRef = useRef<any>(null);
    const imgCache = useRef<{ [key: string]: HTMLImageElement }>({});

    useEffect(() => {
        if (!containerRef.current) return;

        // Initialize Graph
        // @ts-ignore
        const Graph = ForceGraph()(containerRef.current)
            .backgroundColor('#000000')
            .nodeId('id')
            .nodeLabel('title')
            .nodeVal('val')
            .linkColor(() => '#333333')
            .nodeColor((node: any) => node.group === 'folder' ? '#ffffff' : '#888888')
            .d3AlphaDecay(0.04) // Stabilize faster
            .d3VelocityDecay(0.3) // Higher friction
            .onNodeClick((node: any) => {
                if (node.url) {
                    window.location.href = node.url;
                } else {
                    Graph.centerAt(node.x, node.y, 1000);
                    Graph.zoom(4, 2000);
                }
            });

        graphInstanceRef.current = Graph;

        // Custom Node Rendering
        Graph.nodeCanvasObject((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
            const label = node.title;
            // Scale font size
            const fontSize = Math.max(3, 12 / globalScale); 
            ctx.font = `${fontSize}px Sans-Serif`;
            
            // Interaction State check
            const isHovered = node === hoveredNodeRef.current;
            const isRoot = node.isRoot;
            
            // Node Radius
            const r = Math.sqrt(Math.max(0, node.val || 1)) * 2;

            // Draw Glow (Hover or Root)
            if (isHovered || isRoot) {
                ctx.beginPath();
                ctx.arc(node.x, node.y, r + (isRoot ? 10 / globalScale : 2 / globalScale), 0, 2 * Math.PI, false); 
                ctx.fillStyle = isRoot ? 'rgba(255, 215, 0, 0.3)' : 'rgba(255, 255, 255, 0.4)'; // Gold for root
                ctx.fill();
            }

            // Draw Node Circle
            ctx.beginPath();
            ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false);
            
            if (isRoot) {
                 ctx.fillStyle = '#FFD700'; // Gold
            } else {
                 ctx.fillStyle = node.group === 'folder' ? '#ffffff' : '#444444';
            }
            ctx.fill();

            // Draw Favicon for bookmarks
            if (node.group === 'bookmark' && node.url) {
                let img = imgCache.current[node.url];
                if (!img) {
                    img = new Image();
                    img.src = `https://www.google.com/s2/favicons?domain=${new URL(node.url).hostname}&sz=32`;
                    imgCache.current[node.url] = img;
                }

                if (img.complete && img.naturalWidth > 0) {
                    ctx.save();
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, r - 0.5, 0, 2 * Math.PI, false);
                    ctx.clip();
                    try {
                        ctx.drawImage(img, node.x - r, node.y - r, r * 2, r * 2);
                    } catch (e) {
                         // Fallback
                    }
                    ctx.restore();
                }
            }

            // Draw Label
            const showLabel = node.group === 'folder' || isHovered || globalScale > 2 || isRoot;

            if (showLabel) {
                const textWidth = ctx.measureText(label).width;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = isHovered ? '#ffffff' : (isRoot ? '#FFD700' : 'rgba(255, 255, 255, 0.8)');
                ctx.fillText(label, node.x, node.y + r + fontSize, textWidth);
            }
        });

        // Add custom hover logic
        Graph.onNodeHover((node: any) => {
            hoveredNodeRef.current = node;
            if (containerRef.current) {
                containerRef.current.style.cursor = node ? 'pointer' : 'default';
            }
        });

        // Load Data
        const loadData = async () => {
            const tree = await bookmarkService.getTree();
            const { nodes, links } = transformData(tree);
            Graph.graphData({ nodes, links });

            // Apply specific physics forces
            // Prevent overlap
            Graph.d3Force('collide', forceCollide((node: any) => {
                const r = Math.sqrt(Math.max(0, node.val || 1)) * 2;
                return r + 2; // Radius + Padding
            }));

            // Strong repulsion for spacing
            Graph.d3Force('charge').strength(-300);
            
            // Adjust links
            Graph.d3Force('link')
                .distance((link: any) => {
                    // Logic: Keep folders closer to root, bookmarks closer to folders (?)
                    // Or keep general spacing structure
                    return 50;
                })
                .strength(0.5); // Slightly tighter structure
            
            // Warmup
            Graph.d3Force('charge').strength(-500); // Stronger initial push
            setTimeout(() => {
                 Graph.d3Force('charge').strength(-300); // Settle down
            }, 1000);
        };

        loadData();

        // Resize Handler
        const handleResize = () => {
            if (containerRef.current) {
                Graph.width(containerRef.current.clientWidth);
                Graph.height(containerRef.current.clientHeight);
            }
        };
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
             graphInstanceRef.current = null;
        };

    }, []);

    // Helper to transform bookmark tree to graph nodes/links
    const transformData = (tree: any[]): GraphData => {
        const nodes: any[] = [];
        const links: any[] = [];

        const traverse = (items: any[], parentId: string | null = null, depth: number = 0) => {
            items.forEach(item => {
                const isFolder = !item.url;
                const isRoot = parentId === null || depth === 0;

                // Calculate size based on depth
                // Massive difference for visibility at global scale
                let val = 5; 
                if (isRoot) {
                    val = 400; // Radius ~40
                } else if (isFolder) {
                    // Start at 150 (Radius ~24), decay structure
                    val = Math.max(20, 150 - ((depth - 1) * 30)); 
                } else {
                    val = 5; // Radius ~4.5
                }

                nodes.push({
                    id: item.id,
                    title: item.title || (item.url ? new URL(item.url).hostname : 'Folder'),
                    group: isFolder ? 'folder' : 'bookmark',
                    url: item.url,
                    val: val,
                    isRoot: isRoot
                });

                if (parentId) {
                    links.push({ source: parentId, target: item.id });
                }

                if (item.children) {
                    traverse(item.children, item.id, depth + 1);
                }
            });
        };

        traverse(tree);
        return { nodes, links };
    };

    return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />;
};

export default BookmarkGraph;

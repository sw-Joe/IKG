// Chrome Bookmark Interfaces
export interface BookmarkNode {
    id: string;
    parentId?: string;
    index?: number;
    url?: string;
    title: string;
    dateAdded?: number;
    dateGroupModified?: number;
    children?: BookmarkNode[];
    blockInteraction?: boolean;
}

// Graph Interfaces (force-graph)
export interface GraphNode {
    id: string;
    title: string;
    group: 'folder' | 'bookmark';
    url?: string;
    val: number; // Size
    x?: number;
    y?: number;
}

export interface GraphLink {
    source: string | GraphNode;
    target: string | GraphNode;
}

export interface GraphData {
    nodes: GraphNode[];
    links: GraphLink[];
}

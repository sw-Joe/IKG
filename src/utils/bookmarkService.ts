import { BookmarkNode } from '../types';

const isExtension = typeof chrome !== 'undefined' && !!chrome.bookmarks;

// Helper to flatten bookmark tree
const flattenBookmarks = (nodes: BookmarkNode[]): BookmarkNode[] => {
    let result: BookmarkNode[] = [];
    for (const node of nodes) {
        if (node.url) {
            result.push(node);
        } else if (node.children) {
            result = result.concat(flattenBookmarks(node.children));
        }
    }
    return result;
};

export const bookmarkService = {
    isExtension,

    getTree: async (): Promise<BookmarkNode[]> => {
        if (isExtension) {
            return new Promise((resolve) => {
                chrome.bookmarks.getTree((tree) => {
                    resolve(tree as BookmarkNode[]);
                });
            });
        } else {
            // Local Mock Data
            const saved = localStorage.getItem('bookmarks');
            const simpleList: BookmarkNode[] = saved ? JSON.parse(saved) : [];
            
            // Mocking a tree structure
            return Promise.resolve([
                {
                    id: '0',
                    title: 'Root',
                    children: [
                        {
                            id: '1',
                            title: 'Bookmarks Bar',
                            children: simpleList
                        }
                    ]
                } as BookmarkNode
            ]);
        }
    },

    getAsFlatList: async (): Promise<BookmarkNode[]> => {
        const tree = await bookmarkService.getTree();
        return flattenBookmarks(tree);
    },

    create: async (bookmark: { title: string; url: string }): Promise<void> => {
        if (isExtension) {
            return new Promise((resolve) => {
                chrome.bookmarks.create({
                    title: bookmark.title,
                    url: bookmark.url,
                    parentId: '1'
                }, () => resolve());
            });
        } else {
            const saved = localStorage.getItem('bookmarks');
            const list: BookmarkNode[] = saved ? JSON.parse(saved) : [];
            const newBookmark = { 
                ...bookmark, 
                id: Date.now().toString(),
                dateAdded: Date.now()
            };
            list.push(newBookmark as BookmarkNode);
            localStorage.setItem('bookmarks', JSON.stringify(list));
            return Promise.resolve();
        }
    },

    remove: async (id: string): Promise<void> => {
        if (isExtension) {
            return new Promise((resolve) => {
                chrome.bookmarks.remove(id, () => resolve());
            });
        } else {
            const saved = localStorage.getItem('bookmarks');
            let list: BookmarkNode[] = saved ? JSON.parse(saved) : [];
            list = list.filter(b => b.id !== id);
            localStorage.setItem('bookmarks', JSON.stringify(list));
            return Promise.resolve();
        }
    }
};

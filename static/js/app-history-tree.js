(function initAppHistoryTreeModule(global) {
    // History tree cards + thumbnail hydration for the sidebar/grid.
    global.AppHistoryTreeMethods = {
        renderHistoryTreeGrid() {
            const grid = document.getElementById('history-tree-grid');
            if (!grid) return;
            const items = this.state.conversations || [];
            if (!items.length) {
                grid.innerHTML = '<div class="text-sm text-zinc-500">No history yet. Upload a file to get started.</div>';
                return;
            }
            grid.innerHTML = items
                .map((conv) => {
                    const id = conv[0];
                    const summary = this.escapeHtml(this.state.historyMap[id]?.summary || conv[3] || 'Untitled Tree');
                    const meta = this.escapeHtml([conv[1], conv[2]].filter(Boolean).join(' · ') || 'No metadata');
                    const thumb = `
                        <div class="history-tree-thumb" data-thumb-conv-id="${id}">
                            <div class="text-[10px] text-zinc-500 absolute inset-0 flex items-center justify-center">Loading tree...</div>
                        </div>
                    `;
                    return `
                        <div class="history-tree-card" onclick="app.openHistoryTree('${id}')">
                            ${thumb}
                            <p class="text-sm font-semibold text-zinc-100 truncate">${summary}</p>
                            <p class="text-[11px] text-zinc-500 mt-1 truncate">${meta}</p>
                        </div>
                    `;
                })
                .join('');
            this.hydrateHistoryTreeThumbs();
        },

        async fetchHistoryTreeStructure(conversationId) {
            try {
                const res = await fetch(`/api/history/${encodeURIComponent(conversationId)}/tree-structure`);
                const data = await res.json();
                if (data.success) return data;
                return null;
            } catch (e) {
                return null;
            }
        },

        renderHistoryTreeThumb(thumbEl, structure) {
            if (!thumbEl) return;
            const nodes = Array.isArray(structure?.nodes) ? structure.nodes : [];
            const edges = Array.isArray(structure?.edges) ? structure.edges : [];
            if (!nodes.length) {
                thumbEl.innerHTML = '<div class="text-[10px] text-zinc-500 absolute inset-0 flex items-center justify-center">No tree snapshot</div>';
                return;
            }

            const byDepth = {};
            nodes.forEach((n) => {
                const d = Number(n.depth || 0);
                if (!byDepth[d]) byDepth[d] = [];
                byDepth[d].push(n);
            });
            const depths = Object.keys(byDepth)
                .map(Number)
                .sort((a, b) => a - b);
            const positions = {};
            const w = thumbEl.clientWidth || 220;
            const h = thumbEl.clientHeight || 110;
            depths.forEach((d, di) => {
                const row = byDepth[d];
                row.forEach((n, ni) => {
                    const x = ((ni + 1) / (row.length + 1)) * w;
                    const y = ((di + 1) / (depths.length + 1)) * h;
                    positions[n.id] = { x, y };
                });
            });

            let html = '';
            edges.forEach((e) => {
                const from = positions[e.from];
                const to = positions[e.to];
                if (!from || !to) return;
                const dx = to.x - from.x;
                const dy = to.y - from.y;
                const len = Math.sqrt(dx * dx + dy * dy);
                const deg = Math.atan2(dy, dx) * (180 / Math.PI);
                html += `<div class="history-tree-link" style="left:${from.x}px;top:${from.y}px;width:${len}px;transform:rotate(${deg}deg)"></div>`;
            });
            nodes.forEach((n) => {
                const p = positions[n.id];
                if (!p) return;
                html += `<div class="history-tree-dot" title="${this.escapeHtml(n.name || '')}" style="left:${p.x - 4}px;top:${p.y - 4}px"></div>`;
            });
            thumbEl.innerHTML = html;
        },

        async hydrateHistoryTreeThumbs() {
            const thumbs = document.querySelectorAll('[data-thumb-conv-id]');
            if (!thumbs.length) return;
            for (const thumb of thumbs) {
                const convId = thumb.getAttribute('data-thumb-conv-id');
                if (!convId) continue;
                const structure = await this.fetchHistoryTreeStructure(convId);
                this.renderHistoryTreeThumb(thumb, structure);
            }
        },

        openHistoryTree(id) {
            if (!id) return;
            if (this.state.activeId && this.state.activeId !== id) {
                this.persistCurrentTreeChat(true);
            }
            this.state.activeId = id;
            this.state.nestedFocusPath = [];
            this.state.nestedPathStack = [];
            this.state.nestedNodeOffsets = {};
            this.state.nestedViewportZoom = 1;
            this.state.nestedViewportScroll = null;
            this.state.nestedFeatureStackOffset = { x: 0, y: 0 };
            this.state.nestedRawData = null;
            this.state.nestedRawDataConversationId = null;
            this.state.nestedExpandState = { index: false, body: false };
            this.state.nestedSemanticFocus = null;
            this.state.nestedDockOpen = false;
            this.setNestedDockVisible(false);
            this.updateActiveConversationTitle(id);
            this.pushModelOutput(`Loaded history tree model: ${id}`);
            this.setView('spanning-tree');
        }
    };
})(window);

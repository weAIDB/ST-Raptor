(function initAppViewRouterModule(global) {
    global.AppViewRouterMethods = {
        updateCenterUploadVisibility() {
            const wrapper = document.getElementById('center-upload-wrapper');
            if (wrapper) {
                wrapper.classList.toggle('hidden', !this.state.isSidebarCollapsed);
            }
        },

        async setView(viewName) {
            if (this.state.view === 'spanning-tree' && viewName !== 'spanning-tree') {
                this.persistCurrentTreeChat(true);
            }
            this.state.view = viewName;
            const isTreeView = viewName === 'spanning-tree';
            document.body.classList.toggle('tree-editor-mode', isTreeView);
            document.getElementById('main-scroll-area')?.classList.toggle('tree-view', isTreeView);
            if (!isTreeView) {
                this.setNestedDockVisible(false);
            }

            ['chat', 'spanning-tree', 'chain-of-thought'].forEach(v => {
                const el = document.getElementById(`${v}-view`);
                if (el) el.classList.add('hidden');
            });
            document.getElementById(`${viewName}-view`).classList.remove('hidden');

            document.querySelectorAll('.nav-item').forEach(item => {
                if (item.getAttribute('data-view') === viewName) {
                    item.classList.add('bg-white/10', 'text-white');
                    item.classList.remove('text-muted-foreground');
                } else {
                    item.classList.remove('bg-white/10', 'text-white');
                    item.classList.add('text-muted-foreground');
                }
            });

            if (viewName === 'chat') {
                document.getElementById('view-header').classList.toggle('hidden', !this.state.activeId);
            } else {
                document.getElementById('view-header').classList.toggle('hidden', viewName === 'spanning-tree');
                if (viewName === 'spanning-tree') {
                    console.info('[tree-grid-state]', { reason: 'setView:spanning-tree-enter' });
                    await this.restoreTreeChatForConversation(this.state.activeId);
                    this.state.tablePreviewExpanded = false;
                    this.toggleTablePreview(false);
                    this.toggleTreeChat(true);
                    this.state.isTracePlaying = false;
                    this.updateTracePlaybackButton();
                    this.updateStrictTraceWarning(true);
                    this.updateTraceDebugOverlay(null, null);
                    this.toggleTreeTopbar(false);
                    this.loadTree();
                }
            }
            if (viewName === 'chain-of-thought') this.refreshChain();
            lucide.createIcons();
        }
    };
})(window);

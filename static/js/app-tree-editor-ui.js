(function initAppTreeEditorUiModule(global) {
    // UI-only tree editor controls.
    // This module intentionally does not contain trace payload parsing logic.
    // It cooperates with /static/js/nested-feature-viewer.js and /static/js/trace-stage-viewer.js via app methods.
    global.AppTreeEditorUiMethods = {
        toggleTablePreview(forceExpanded = null) {
            const section = document.getElementById('uploaded-files-section');
            const btn = document.getElementById('toggle-table-preview-btn');
            if (!section || !btn) return;
            const nextExpanded = forceExpanded === null ? !this.state.tablePreviewExpanded : !!forceExpanded;
            this.state.tablePreviewExpanded = nextExpanded;
            section.classList.toggle('tree-table-collapsed', !nextExpanded);
            btn.textContent = nextExpanded ? 'Hide Table Preview' : 'Show Table Preview';
            if (nextExpanded) this.renderTablePreview();
        },

        toggleTreeChat(forceOpen = null) {
            const wrap = document.querySelector('.tree-side-chat-wrap');
            const panel = document.getElementById('tree-side-chat');
            const btn = document.getElementById('toggle-tree-chat-btn');
            const edgeIcon = document.querySelector('#tree-chat-edge-toggle i');
            const saveWrap = document.querySelector('.tree-save-floating');
            if (!wrap || !panel || !btn) return;
            const nextOpen = forceOpen === null ? !this.state.treeChatOpen : !!forceOpen;
            this.state.treeChatOpen = nextOpen;
            wrap.classList.toggle('collapsed', !nextOpen);
            panel.classList.toggle('collapsed', !nextOpen);
            btn.textContent = nextOpen ? 'Hide Q&A' : 'Show Q&A';
            if (saveWrap) saveWrap.style.display = nextOpen ? 'none' : '';
            if (edgeIcon) {
                edgeIcon.setAttribute('data-lucide', nextOpen ? 'chevron-right' : 'chevron-left');
                lucide.createIcons();
            }
        },

        toggleTreeTopbar(forceCollapse = null) {
            const topbar = document.querySelector('.tree-editor-topbar');
            const btn = document.getElementById('tree-topbar-collapse-btn');
            if (!topbar || !btn) return;
            const nextCollapsed = forceCollapse === null ? !this.state.treeTopbarCollapsed : !!forceCollapse;
            this.state.treeTopbarCollapsed = nextCollapsed;
            topbar.classList.toggle('is-collapsed', nextCollapsed);
            btn.textContent = nextCollapsed ? '▾ Expand' : '▴ Collapse';
            btn.title = nextCollapsed ? 'Expand top bar' : 'Collapse top bar';
            btn.setAttribute('aria-label', nextCollapsed ? 'Expand top bar' : 'Collapse top bar');
            this.setupTreeModeScroller();
        },

        updateTreeMainGridLayoutState(reason = 'unknown') {
            const grid = document.getElementById('tree-main-grid');
            if (!grid) return;
            const keepNestedColumn = !!this.state.nestedDockOpen || !!this.state.nestedDockAnimatingClose;
            grid.classList.toggle('is-nested-hidden', !keepNestedColumn);
            grid.classList.toggle('is-nested-active', !!this.state.nestedDockOpen);
            grid.classList.toggle('is-trace-collapsed', !this.state.traceStageDrawerOpen);
            const snapshot = `${grid.className}|nested:${!!this.state.nestedDockOpen}|trace:${!!this.state.traceStageDrawerOpen}|closing:${!!this.state.nestedDockAnimatingClose}`;
            if (this.state._treeGridDebugLast !== snapshot) {
                this.state._treeGridDebugLast = snapshot;
                console.info('[tree-grid-state]', {
                    reason,
                    classes: grid.className,
                    nestedDockOpen: !!this.state.nestedDockOpen,
                    nestedDockAnimatingClose: !!this.state.nestedDockAnimatingClose,
                    traceStageDrawerOpen: !!this.state.traceStageDrawerOpen
                });
            }
        },

        clearNestedDockContent() {
            const content = document.getElementById('nested-side-content');
            if (!content) return;
            content.innerHTML = '<div class="trace-empty-state">Click a \"Subtree\" action in trace to locate and display the corresponding nested level here.</div>';
        },

        clearNestedDockState() {
            this.state.nestedSemanticFocus = null;
            this.state.nestedFocusPath = [];
            this.state.nestedPathStack = [];
            this.state.nestedExpandState = { index: false, body: false };
            this.setNestedDockVisible(false);
        },

        setNestedDockVisible(forceOpen = null) {
            const pane = document.getElementById('nested-side-pane');
            if (!pane) return;
            const isOpen = !!this.state.nestedDockOpen;
            const nextOpen = forceOpen === null ? !this.state.nestedDockOpen : !!forceOpen;
            if (!nextOpen && !isOpen && !this.state.nestedDockAnimatingClose) {
                // Already closed: do not enter "animating close", otherwise grid briefly expands.
                pane.classList.add('hidden');
                pane.classList.add('is-collapsed');
                this.updateTreeMainGridLayoutState('setNestedDockVisible:noop-close');
                return;
            }
            if (nextOpen && isOpen && !this.state.nestedDockAnimatingClose) {
                // Already open: keep visible state stable, no-op.
                pane.classList.remove('hidden');
                pane.classList.remove('is-collapsed');
                this.updateTreeMainGridLayoutState('setNestedDockVisible:noop-open');
                return;
            }
            if (this.state.nestedDockHideTimer) {
                clearTimeout(this.state.nestedDockHideTimer);
                this.state.nestedDockHideTimer = null;
            }
            if (nextOpen) {
                this.state.nestedDockAnimatingClose = false;
                this.state.nestedDockOpen = true;
                pane.classList.add('is-collapsed');
                pane.classList.remove('hidden');
                setTimeout(() => pane.classList.remove('is-collapsed'), 24);
            } else {
                this.state.nestedDockAnimatingClose = true;
                this.state.nestedDockOpen = false;
                pane.classList.add('is-collapsed');
                this.state.nestedDockHideTimer = setTimeout(() => {
                    if (!this.state.nestedDockOpen) {
                        pane.classList.add('hidden');
                    }
                    this.state.nestedDockAnimatingClose = false;
                    this.updateTreeMainGridLayoutState('setNestedDockVisible:close-finished');
                    this.state.nestedDockHideTimer = null;
                }, 360);
                this.state.nestedSemanticFocus = null;
                this.clearNestedDockContent();
            }
            this.updateTreeMainGridLayoutState(`setNestedDockVisible:${nextOpen ? 'open' : 'close'}`);
            if (Array.isArray(this.state.nestedDockReflowTimers)) {
                this.state.nestedDockReflowTimers.forEach((timer) => clearTimeout(timer));
                this.state.nestedDockReflowTimers = [];
            }
            if (nextOpen && typeof this.renderNestedFeatureView === 'function') {
                // During grid width transition, first render may capture a tiny viewport.
                // Do a single late reflow to avoid repeated flash while still fixing small first width.
                this.state.nestedDockReflowTimers = [320].map((delay) => setTimeout(() => {
                    if (!this.state.nestedDockOpen) return;
                    this.renderNestedFeatureView({ preserveViewport: true });
                }, delay));
            }
        },

        toggleTraceStageDrawer(forceOpen = null) {
            const drawer = document.getElementById('trace-stage-drawer');
            if (!drawer) return;
            const nextOpen = forceOpen === null ? !this.state.traceStageDrawerOpen : !!forceOpen;
            if (this.state.traceDrawerHideTimer) {
                clearTimeout(this.state.traceDrawerHideTimer);
                this.state.traceDrawerHideTimer = null;
            }
            this.state.traceStageDrawerOpen = nextOpen;
            if (nextOpen) {
                drawer.classList.add('collapsed');
                drawer.classList.remove('hidden');
                setTimeout(() => drawer.classList.remove('collapsed'), 24);
            } else {
                drawer.classList.add('collapsed');
                this.state.traceDrawerHideTimer = setTimeout(() => {
                    if (!this.state.traceStageDrawerOpen) {
                        drawer.classList.add('hidden');
                    }
                    this.state.traceDrawerHideTimer = null;
                }, 360);
            }
            if (!nextOpen) {
                this.stopTraceStageAppend();
                this.closeTraceSubtreePreview();
            }
            this.updateTreeMainGridLayoutState(`toggleTraceStageDrawer:${nextOpen ? 'open' : 'close'}`);
        },

        setupTreeModeScroller() {
            const viewport = document.getElementById('tree-mode-scroll');
            const prevBtn = document.getElementById('tree-mode-scroll-prev');
            const nextBtn = document.getElementById('tree-mode-scroll-next');
            if (!viewport || !prevBtn || !nextBtn) return;
            const updateNavState = () => {
                const maxLeft = Math.max(0, Math.ceil(viewport.scrollWidth - viewport.clientWidth));
                const cur = Math.max(0, Math.ceil(viewport.scrollLeft || 0));
                const canPrev = cur > 2;
                const canNext = cur < (maxLeft - 2);
                prevBtn.disabled = !canPrev;
                nextBtn.disabled = !canNext;
            };
            const scrollStep = () => Math.max(220, Math.floor(viewport.clientWidth * 0.82));
            const scrollByStep = (delta) => {
                const maxLeft = Math.max(0, viewport.scrollWidth - viewport.clientWidth);
                if (maxLeft <= 0) {
                    updateNavState();
                    return;
                }
                const current = Math.max(0, viewport.scrollLeft || 0);
                const target = Math.max(0, Math.min(maxLeft, current + delta));
                if (typeof viewport.scrollTo === 'function') {
                    viewport.scrollTo({ left: target, behavior: 'smooth' });
                } else {
                    viewport.scrollLeft = target;
                }
                requestAnimationFrame(updateNavState);
                setTimeout(updateNavState, 220);
            };
            const scheduleNavRefresh = () => {
                requestAnimationFrame(() => {
                    updateNavState();
                    requestAnimationFrame(updateNavState);
                });
                setTimeout(updateNavState, 80);
                setTimeout(updateNavState, 240);
                setTimeout(updateNavState, 420);
            };

            if (viewport.dataset.bound !== '1') {
                viewport.dataset.bound = '1';
                prevBtn.addEventListener('click', (ev) => {
                    ev.preventDefault();
                    ev.stopPropagation();
                    scrollByStep(-scrollStep());
                });
                nextBtn.addEventListener('click', (ev) => {
                    ev.preventDefault();
                    ev.stopPropagation();
                    scrollByStep(scrollStep());
                });
                viewport.addEventListener('scroll', updateNavState, { passive: true });
                window.addEventListener('resize', updateNavState);
                if (window.ResizeObserver) {
                    const ro = new ResizeObserver(() => updateNavState());
                    ro.observe(viewport);
                    const inner = viewport.querySelector('.tree-mode-scroll-inner');
                    if (inner) ro.observe(inner);
                    this._treeModeScrollResizeObserver = ro;
                }
                if (window.MutationObserver) {
                    const mo = new MutationObserver(() => updateNavState());
                    const inner = viewport.querySelector('.tree-mode-scroll-inner');
                    if (inner) {
                        mo.observe(inner, { childList: true, subtree: false, attributes: true });
                    }
                    this._treeModeScrollMutationObserver = mo;
                }
            }
            scheduleNavRefresh();
        }
    };
})(window);

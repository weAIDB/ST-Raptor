(function installNestedFeatureViewerModule() {
    const clampZoom = (value) => Math.max(0.55, Math.min(2.2, Number(value || 1)));

    function getNestedDom(container) {
        if (!container) return {};
        return {
            viewportEl: container.querySelector('[data-nested-viewport]'),
            canvasEl: container.querySelector('[data-nested-canvas]'),
            featureNodeEl: container.querySelector('[data-nested-feature-node]'),
        };
    }

    window.installNestedFeatureViewer = function installNestedFeatureViewer(app) {
        Object.assign(app, {
            _nestedBuildRootFeature(rawData = undefined) {
                const source = rawData === undefined ? this.state.nestedRawData : rawData;
                if (!source || typeof source !== 'object' || Array.isArray(source)) return null;
                return { root: source };
            },

            _nestedBreadcrumbLabelForTarget(target = null) {
                if (!target || typeof target !== 'object') return 'sub feature tree';
                const nestedData = target.nestedData;
                if (nestedData && typeof nestedData === 'object' && !Array.isArray(nestedData)) {
                    const keys = Object.keys(nestedData).map((k) => String(k || '').trim()).filter(Boolean);
                    if (keys.length) return keys[0];
                }
                if (Array.isArray(nestedData) && nestedData.length) {
                    return '[0]';
                }
                const fallback = String(target.name || '').trim();
                return fallback || 'sub feature tree';
            },

            _isNestedPanelActive() {
                return (this.state.treeViewMode || 'nested') === 'nested' || !!this.state.nestedDockOpen;
            },

            _getNestedRenderContainer() {
                const mode = this.state.treeViewMode || 'nested';
                if (mode === 'nested') {
                    return document.getElementById('tree-content');
                }
                return document.getElementById('nested-side-content') || document.getElementById('tree-content');
            },

            _getNestedFeatureAtPath() {
                const stack = Array.isArray(this.state.nestedPathStack) ? this.state.nestedPathStack : [];
                if (stack.length) {
                    const cur = stack[stack.length - 1]?.data;
                    if (cur && typeof cur === 'object' && !Array.isArray(cur)) return cur;
                }
                const root = this.state.nestedRawData;
                const rootFeature = this._nestedBuildRootFeature(root);
                if (!rootFeature) return null;
                const path = Array.isArray(this.state.nestedFocusPath) ? this.state.nestedFocusPath : [];
                if (!path.length) return rootFeature;
                let cur = root;
                for (const key of path) {
                    if (!cur || typeof cur !== 'object' || Array.isArray(cur)) return rootFeature;
                    if (!(key in cur)) return rootFeature;
                    cur = cur[key];
                }
                if (cur && typeof cur === 'object' && !Array.isArray(cur)) return cur;
                return rootFeature;
            },

            _nestedParseFeatureLevel(featureObj) {
                const indexNodes = [];
                const bodyNodes = [];
                if (!featureObj || typeof featureObj !== 'object' || Array.isArray(featureObj)) {
                    return { indexNodes, bodyNodes };
                }

                let bodyIdx = 0;
                const pushBodyNode = (indexIdx, name, opts = {}) => {
                    bodyNodes.push({
                        indexIdx,
                        name: String(name ?? ''),
                        isNested: !!opts.isNested,
                        nestedData: opts.nestedData || null,
                        meta: String(opts.meta || ''),
                    });
                    bodyIdx += 1;
                };
                const splitBodyParts = (rawValue) => {
                    const text = String(rawValue ?? '').trim();
                    if (!text) return [];
                    return text
                        .split(',')
                        .map((s) => String(s || '').trim())
                        .filter(Boolean);
                };
                const pushNestedPlaceholder = (indexIdx, nestedValue) => {
                    const nestedCount = nestedValue && typeof nestedValue === 'object' && !Array.isArray(nestedValue)
                        ? Object.keys(nestedValue).length
                        : 0;
                    pushBodyNode(indexIdx, 'sub feature tree', {
                        isNested: true,
                        nestedData: nestedValue,
                        meta: nestedCount > 0 ? `(${nestedCount})` : '',
                    });
                };

                Object.keys(featureObj).forEach((indexKey, indexIdx) => {
                    const value = featureObj[indexKey];
                    const startBodyIdx = bodyIdx;

                    if (value && typeof value === 'object' && !Array.isArray(value)) {
                        pushNestedPlaceholder(indexIdx, value);
                    } else if (Array.isArray(value)) {
                        value.forEach((item) => {
                            if (item && typeof item === 'object' && !Array.isArray(item)) {
                                pushNestedPlaceholder(indexIdx, item);
                            } else if (Array.isArray(item)) {
                                item.forEach((inner) => {
                                    if (inner && typeof inner === 'object' && !Array.isArray(inner)) {
                                        pushNestedPlaceholder(indexIdx, inner);
                                        return;
                                    }
                                    splitBodyParts(inner).forEach((part) => pushBodyNode(indexIdx, part));
                                });
                            } else {
                                splitBodyParts(item).forEach((part) => pushBodyNode(indexIdx, part));
                            }
                        });
                    } else {
                        splitBodyParts(value).forEach((part) => pushBodyNode(indexIdx, part));
                    }

                    if (bodyIdx === startBodyIdx) {
                        pushBodyNode(indexIdx, '-');
                    }

                    const bodyIndices = [];
                    for (let i = startBodyIdx; i < bodyIdx; i += 1) {
                        bodyIndices.push(i);
                    }
                    indexNodes.push({
                        key: String(indexKey),
                        bodyIndices,
                    });
                });

                return { indexNodes, bodyNodes };
            },

            _nestedGetOffset(id) {
                const all = this.state.nestedNodeOffsets || {};
                const val = all[id];
                if (!val || typeof val !== 'object') return { x: 0, y: 0 };
                return {
                    x: Number(val.x) || 0,
                    y: Number(val.y) || 0,
                };
            },

            _syncNestedViewportState(viewportEl) {
                if (!viewportEl) return;
                this.state.nestedViewportScroll = {
                    left: Math.max(0, viewportEl.scrollLeft || 0),
                    top: Math.max(0, viewportEl.scrollTop || 0),
                };
            },

            _getNestedFeatureNodeOffset() {
                const offset = this.state.nestedFeatureStackOffset || {};
                return {
                    x: Number(offset.x) || 0,
                    y: Number(offset.y) || 0,
                };
            },

            _getNestedFeatureNodeTransform() {
                const offset = this._getNestedFeatureNodeOffset();
                return `translate(calc(-50% + ${offset.x}px), calc(-50% + ${offset.y}px))`;
            },

            _applyNestedFeatureNodeTransform(container) {
                const { featureNodeEl } = getNestedDom(container);
                if (!featureNodeEl) return;
                featureNodeEl.style.transform = this._getNestedFeatureNodeTransform();
            },

            _captureNestedViewportState(container) {
                const { viewportEl } = getNestedDom(container);
                if (!viewportEl) return this.state.nestedViewportScroll || null;
                const snapshot = {
                    left: Math.max(0, viewportEl.scrollLeft || 0),
                    top: Math.max(0, viewportEl.scrollTop || 0),
                };
                this.state.nestedViewportScroll = snapshot;
                return snapshot;
            },

            _centerNestedViewportInCanvas(container) {
                const { viewportEl, canvasEl } = getNestedDom(container);
                if (!viewportEl || !canvasEl) return;
                const canvasWidth = Math.max(1, Number(viewportEl.dataset.fixedCanvasW || 0) || canvasEl.offsetWidth || viewportEl.scrollWidth || 1);
                const canvasHeight = Math.max(1, Number(viewportEl.dataset.fixedCanvasH || 0) || canvasEl.offsetHeight || viewportEl.scrollHeight || 1);
                viewportEl.scrollLeft = Math.max(0, (canvasWidth - viewportEl.clientWidth) / 2);
                viewportEl.scrollTop = Math.max(0, (canvasHeight - viewportEl.clientHeight) / 2);
                this._syncNestedViewportState(viewportEl);
            },

            _centerNestedFeatureNodeInViewport(container) {
                const { viewportEl, canvasEl } = getNestedDom(container);
                if (!viewportEl || !canvasEl) return;
                const zoom = clampZoom(this.state.nestedViewportZoom || viewportEl.dataset.zoom || 1);
                const canvasWidth = Math.max(1, Number(viewportEl.dataset.fixedCanvasW || 0) || canvasEl.offsetWidth || viewportEl.scrollWidth || 1);
                const canvasHeight = Math.max(1, Number(viewportEl.dataset.fixedCanvasH || 0) || canvasEl.offsetHeight || viewportEl.scrollHeight || 1);
                const featureOffset = this._getNestedFeatureNodeOffset();
                // feature node 默认锚在画布中心，因此它的中心点 = 画布中心 + 用户平移偏移。
                const featureCenterX = (canvasWidth / 2) + featureOffset.x;
                const featureCenterY = (canvasHeight / 2) + featureOffset.y;
                viewportEl.scrollLeft = Math.max(0, featureCenterX * zoom - (viewportEl.clientWidth / 2));
                viewportEl.scrollTop = Math.max(0, featureCenterY * zoom - (viewportEl.clientHeight / 2));
                this._syncNestedViewportState(viewportEl);
            },

            _restoreNestedViewportState(container, snapshot = null) {
                const { viewportEl } = getNestedDom(container);
                if (!viewportEl) return;
                const saved = snapshot || this.state.nestedViewportScroll;
                if (!saved) {
                    this._centerNestedFeatureNodeInViewport(container);
                    return;
                }
                viewportEl.scrollLeft = Math.max(0, Number(saved.left) || 0);
                viewportEl.scrollTop = Math.max(0, Number(saved.top) || 0);
                this._syncNestedViewportState(viewportEl);
            },

            _scheduleNestedViewportRestore(container, snapshot = null) {
                if (!container) return;
                const restore = () => {
                    if (!this._isNestedPanelActive()) return;
                    this._restoreNestedViewportState(container, snapshot);
                };
                restore();
                requestAnimationFrame(restore);
                requestAnimationFrame(() => requestAnimationFrame(restore));
            },

            _getNestedViewportMetrics(container = null) {
                const host = container || this._getNestedRenderContainer();
                const contentRect = host?.getBoundingClientRect();
                const viewportWidth = Math.max(420, Math.floor(contentRect?.width || window.innerWidth));
                const viewportHeight = Math.max(420, Math.floor(contentRect?.height || (window.innerHeight - 180)));
                return { viewportWidth, viewportHeight };
            },

            _renderNestedConnectorLines(container) {
                if (!container) return;
                const stageEl = container.querySelector('.nested-stage');
                const svgEl = stageEl?.querySelector('[data-nested-stage-svg]');
                if (!stageEl || !svgEl) return;
                const stageRect = stageEl.getBoundingClientRect();
                if (!stageRect.width || !stageRect.height) return;

                const indexEls = Array.from(stageEl.querySelectorAll('[data-index-node]'));
                const bodyAnchorEls = Array.from(stageEl.querySelectorAll('[data-nested-body-group-anchor]'));
                const n = Math.min(indexEls.length, bodyAnchorEls.length);
                const paths = [];

                for (let i = 0; i < n; i += 1) {
                    const srcRect = indexEls[i].getBoundingClientRect();
                    const dstRect = bodyAnchorEls[i].getBoundingClientRect();
                    const x1 = srcRect.left - stageRect.left + srcRect.width / 2;
                    const y1 = srcRect.bottom - stageRect.top;
                    const x2 = dstRect.left - stageRect.left + dstRect.width / 2;
                    const y2 = dstRect.top - stageRect.top + dstRect.height / 2;
                    if (!Number.isFinite(x1) || !Number.isFinite(y1) || !Number.isFinite(x2) || !Number.isFinite(y2)) continue;

                    const bend = Math.max(16, (y2 - y1) * 0.44);
                    const c1x = x1;
                    const c1y = y1 + bend;
                    const c2x = x2;
                    const c2y = y2 - bend;
                    paths.push(
                        `<path class="nested-line" d="M ${x1} ${y1} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${x2} ${y2}" marker-end="url(#nested-arrowhead)" />`
                    );
                }

                svgEl.setAttribute('viewBox', `0 0 ${Math.max(stageRect.width, 1)} ${Math.max(stageRect.height, 1)}`);
                svgEl.innerHTML = `
                    <defs>
                        <marker id="nested-arrowhead" markerWidth="7" markerHeight="6" refX="6.2" refY="3" orient="auto">
                            <polygon points="0 0, 7 3, 0 6" fill="#ef4444"></polygon>
                        </marker>
                    </defs>
                    ${paths.join('')}
                `;
            },

            _applyNestedViewportZoom(container, anchorClient = null) {
                const { viewportEl, canvasEl } = getNestedDom(container);
                if (!viewportEl || !canvasEl) return;

                const zoom = clampZoom(this.state.nestedViewportZoom || 1);
                const prevZoom = clampZoom(viewportEl.dataset.zoom || 1);
                const viewportWidth = Math.max(1, Number(viewportEl.dataset.fixedViewportW || 0) || viewportEl.clientWidth || 1);
                const viewportHeight = Math.max(1, Number(viewportEl.dataset.fixedViewportH || 0) || viewportEl.clientHeight || 1);
                const canvasWidth = Math.max(1, Number(viewportEl.dataset.fixedCanvasW || 0) || canvasEl.scrollWidth || viewportEl.clientWidth || 1);
                const canvasHeight = Math.max(1, Number(viewportEl.dataset.fixedCanvasH || 0) || canvasEl.scrollHeight || viewportEl.clientHeight || 1);

                const rect = viewportEl.getBoundingClientRect();
                let canvasAnchorX = 0;
                let canvasAnchorY = 0;
                if (anchorClient && Number.isFinite(anchorClient.x) && Number.isFinite(anchorClient.y)) {
                    canvasAnchorX = (viewportEl.scrollLeft + (anchorClient.x - rect.left)) / prevZoom;
                    canvasAnchorY = (viewportEl.scrollTop + (anchorClient.y - rect.top)) / prevZoom;
                }

                viewportEl.style.width = `${viewportWidth}px`;
                viewportEl.style.height = `${viewportHeight}px`;
                canvasEl.style.width = `${canvasWidth}px`;
                canvasEl.style.height = `${canvasHeight}px`;
                canvasEl.style.transform = `scale(${zoom})`;
                viewportEl.dataset.zoom = String(zoom);

                if (anchorClient && Number.isFinite(anchorClient.x) && Number.isFinite(anchorClient.y)) {
                    viewportEl.scrollLeft = Math.max(0, canvasAnchorX * zoom - (anchorClient.x - rect.left));
                    viewportEl.scrollTop = Math.max(0, canvasAnchorY * zoom - (anchorClient.y - rect.top));
                }

                this._syncNestedViewportState(viewportEl);
                this._renderNestedConnectorLines(container);
            },

            _bindNestedViewportZoom(container) {
                if (!container) return;
                if (container.dataset.nestedZoomBound === '1') return;
                container.dataset.nestedZoomBound = '1';
                container.addEventListener('wheel', (e) => {
                    if (!this._isNestedPanelActive()) return;
                    const hit = e.target instanceof Element ? e.target.closest('[data-nested-viewport]') : null;
                    if (!hit) return;
                    e.preventDefault();
                    if (!(e.ctrlKey || e.metaKey)) return;
                    const cur = clampZoom(this.state.nestedViewportZoom || 1);
                    const scale = e.deltaY < 0 ? 1.08 : 0.92;
                    const next = clampZoom(cur * scale);
                    if (Math.abs(next - cur) < 1e-4) return;
                    this.state.nestedViewportZoom = next;
                    this._applyNestedViewportZoom(container, {
                        x: e.clientX,
                        y: e.clientY,
                    });
                }, { passive: false });
            },

            _bindNestedViewportPan(container) {
                const { viewportEl } = getNestedDom(container);
                if (!viewportEl || viewportEl.dataset.panBound === '1') return;
                viewportEl.dataset.panBound = '1';
                viewportEl.addEventListener('pointerdown', (ev) => {
                    if (ev.button !== 0) return;
                    const target = ev.target instanceof Element ? ev.target : null;
                    if (target?.closest('[data-nested-drag-id], [data-nested-bc-idx], [data-nested-toggle], button, input, textarea, select, a, label')) {
                        return;
                    }

                    const startX = ev.clientX;
                    const startY = ev.clientY;
                    const startLeft = viewportEl.scrollLeft;
                    const startTop = viewportEl.scrollTop;
                    viewportEl.classList.add('dragging-viewport');

                    const onMove = (moveEv) => {
                        viewportEl.scrollLeft = startLeft - (moveEv.clientX - startX);
                        viewportEl.scrollTop = startTop - (moveEv.clientY - startY);
                        this._syncNestedViewportState(viewportEl);
                    };

                    const onUp = () => {
                        viewportEl.classList.remove('dragging-viewport');
                        this._syncNestedViewportState(viewportEl);
                        window.removeEventListener('pointermove', onMove);
                        window.removeEventListener('pointerup', onUp);
                        window.removeEventListener('pointercancel', onUp);
                    };

                    window.addEventListener('pointermove', onMove);
                    window.addEventListener('pointerup', onUp, { once: true });
                    window.addEventListener('pointercancel', onUp, { once: true });
                    ev.preventDefault();
                });
            },

            _bindNestedViewportScroll(container) {
                const { viewportEl } = getNestedDom(container);
                if (!viewportEl || viewportEl.dataset.scrollBound === '1') return;
                viewportEl.dataset.scrollBound = '1';
                viewportEl.addEventListener('scroll', () => {
                    this._syncNestedViewportState(viewportEl);
                }, { passive: true });
            },

            _bindNestedDrag(container) {
                if (!container) return;
                container.querySelectorAll('[data-nested-drag-id]').forEach((el) => {
                    if (el.dataset.dragBound === '1') return;
                    el.dataset.dragBound = '1';
                    el.addEventListener('pointerdown', (ev) => {
                        if (ev.button !== 0) return;
                        const dragId = String(el.getAttribute('data-nested-drag-id') || '');
                        if (!dragId) return;
                        const start = this._nestedGetOffset(dragId);
                        const startX = ev.clientX;
                        const startY = ev.clientY;
                        let moved = false;
                        el.classList.add('dragging');

                        const onMove = (moveEv) => {
                            const dx = moveEv.clientX - startX;
                            const dy = moveEv.clientY - startY;
                            if (Math.abs(dx) > 2 || Math.abs(dy) > 2) moved = true;
                            const x = start.x + dx;
                            const y = start.y + dy;
                            this.state.nestedNodeOffsets = {
                                ...(this.state.nestedNodeOffsets || {}),
                                [dragId]: { x, y },
                            };
                            el.style.transform = `translate(${x}px, ${y}px)`;
                            this._renderNestedConnectorLines(container);
                        };

                        const onUp = (upEv) => {
                            const dx = upEv.clientX - startX;
                            const dy = upEv.clientY - startY;
                            const x = start.x + dx;
                            const y = start.y + dy;
                            this.state.nestedNodeOffsets = {
                                ...(this.state.nestedNodeOffsets || {}),
                                [dragId]: { x, y },
                            };
                            if (moved) {
                                el.dataset.dragMoved = '1';
                                setTimeout(() => {
                                    if (el) el.dataset.dragMoved = '';
                                }, 220);
                            }
                            el.classList.remove('dragging');
                            this._renderNestedConnectorLines(container);
                            window.removeEventListener('pointermove', onMove);
                            window.removeEventListener('pointerup', onUp);
                        };

                        window.addEventListener('pointermove', onMove);
                        window.addEventListener('pointerup', onUp, { once: true });
                    });
                });
            },

            _setNestedColumnGlow(container, colIndex = null) {
                if (!container) return;
                container.querySelectorAll('.nested-col-glow').forEach((el) => el.classList.remove('nested-col-glow'));
                if (!Number.isFinite(colIndex) || colIndex < 0) return;
                const selector = `[data-nested-col="${colIndex}"], [data-nested-body-col="${colIndex}"]`;
                container.querySelectorAll(selector).forEach((el) => el.classList.add('nested-col-glow'));
            },

            _bindNestedColumnGlow(container) {
                if (!container) return;
                container.querySelectorAll('[data-nested-col]').forEach((el) => {
                    const colIndex = Number(el.getAttribute('data-nested-col'));
                    if (!Number.isFinite(colIndex) || colIndex < 0) return;
                    el.addEventListener('mouseenter', () => this._setNestedColumnGlow(container, colIndex));
                    el.addEventListener('mouseleave', () => this._setNestedColumnGlow(container, null));
                });
            },

            _bindNestedStageMotion(container) {
                if (!container) return;
                const stageEl = container.querySelector('.nested-stage');
                const { featureNodeEl } = getNestedDom(container);
                if (!stageEl || !featureNodeEl || stageEl.dataset.motionBound === '1') return;
                stageEl.dataset.motionBound = '1';

                const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
                const applyMotion = (mx = 0, my = 0, active = false) => {
                    featureNodeEl.style.setProperty('--nested-mx', String(mx));
                    featureNodeEl.style.setProperty('--nested-my', String(my));
                    featureNodeEl.classList.toggle('is-interacting', !!active);
                };

                stageEl.addEventListener('pointermove', (ev) => {
                    if (prefersReducedMotion()) return;
                    const rect = stageEl.getBoundingClientRect();
                    if (!rect.width || !rect.height) return;
                    const mx = ((ev.clientX - rect.left) / rect.width) - 0.5;
                    const my = ((ev.clientY - rect.top) / rect.height) - 0.5;
                    applyMotion(
                        Math.max(-0.5, Math.min(0.5, mx)).toFixed(4),
                        Math.max(-0.5, Math.min(0.5, my)).toFixed(4),
                        true
                    );
                });

                stageEl.addEventListener('pointerleave', () => applyMotion(0, 0, false));
                applyMotion(0, 0, false);
            },

            _preserveNestedFeatureNodePosition(container, beforeRect) {
                if (!container || !beforeRect) return;
                const { featureNodeEl } = getNestedDom(container);
                if (!featureNodeEl) return;
                requestAnimationFrame(() => {
                    const afterRect = featureNodeEl.getBoundingClientRect();
                    const dx = beforeRect.left - afterRect.left;
                    const dy = beforeRect.top - afterRect.top;
                    if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) {
                        this._scheduleNestedConnectorRefresh(container);
                        return;
                    }
                    const prev = this._getNestedFeatureNodeOffset();
                    this.state.nestedFeatureStackOffset = {
                        x: prev.x + dx,
                        y: prev.y + dy,
                    };
                    this._applyNestedFeatureNodeTransform(container);
                    this._scheduleNestedConnectorRefresh(container);
                });
            },

            _scheduleNestedConnectorRefresh(container) {
                if (!container) return;
                const svgEl = container.querySelector('[data-nested-stage-svg]');
                const shouldShow = !!this.state.nestedExpandState?.index && !!this.state.nestedExpandState?.body;
                if (svgEl) {
                    svgEl.classList.toggle('is-hidden', !shouldShow);
                    if (!shouldShow) {
                        svgEl.innerHTML = '';
                    }
                }
                if (!shouldShow) return;
                if (Array.isArray(this._nestedConnectorRefreshTimers)) {
                    this._nestedConnectorRefreshTimers.forEach((timer) => window.clearTimeout(timer));
                }
                this._nestedConnectorRefreshTimers = [0, 120, 240, 380].map((delay) => window.setTimeout(() => {
                    if (!this._isNestedPanelActive()) return;
                    this._renderNestedConnectorLines(container);
                }, delay));
            },

            _bindNestedViewportResize() {
                if (this._nestedViewportResizeBound) return;
                this._nestedViewportResizeBound = true;
                window.addEventListener('resize', () => {
                    if (!this._isNestedPanelActive()) return;
                    window.clearTimeout(this._nestedViewportResizeTimer);
                    this._nestedViewportResizeTimer = window.setTimeout(() => {
                        this.renderNestedFeatureView();
                    }, 80);
                });
            },

            renderNestedFeatureView(options = {}) {
                const renderHost = this._getNestedRenderContainer();
                if (!renderHost) return;

                const preserveViewport = options.preserveViewport !== false;
                const prevViewport = preserveViewport ? this._captureNestedViewportState(renderHost) : null;
                const shouldCenterViewport = !preserveViewport || !prevViewport;
                if (!preserveViewport) {
                    this.state.nestedViewportScroll = null;
                    this.state.nestedFeatureStackOffset = { x: 0, y: 0 };
                }

                const { viewportWidth, viewportHeight } = this._getNestedViewportMetrics(renderHost);
                const canvasWidth = Math.max(3200, Math.ceil(viewportWidth * 2.05));
                const canvasHeight = Math.max(1700, Math.ceil(viewportHeight * 1.45));
                const featureNodeTransform = this._getNestedFeatureNodeTransform();
                const featureObj = this._getNestedFeatureAtPath();
                const path = Array.isArray(this.state.nestedFocusPath) ? this.state.nestedFocusPath : [];
                const semanticFocus = this.state.nestedSemanticFocus && typeof this.state.nestedSemanticFocus === 'object'
                    ? this.state.nestedSemanticFocus
                    : null;
                const focusIndexPath = Array.isArray(semanticFocus?.indexPath)
                    ? semanticFocus.indexPath.map((part) => String(part || '').trim()).filter(Boolean)
                    : [];
                const focusParentPath = focusIndexPath.slice(0, -1);
                const focusIndexKey = focusIndexPath.length
                    ? String(focusIndexPath[focusIndexPath.length - 1] || '').trim()
                    : '';
                const highlightCurrentLevel = !!focusIndexKey
                    && focusParentPath.length === path.length
                    && focusParentPath.every((part, idx) => String(part) === String(path[idx]));

                if (!featureObj || typeof featureObj !== 'object' || Array.isArray(featureObj)) {
                    renderHost.innerHTML = `<div class="nested-feature-wrap"><div class="nested-feature-card">Current level is not a feature node. Click back to the previous level.</div></div>`;
                    return;
                }

                const { indexNodes, bodyNodes } = this._nestedParseFeatureLevel(featureObj);
                const indexCount = indexNodes.length;
                const bodyCount = bodyNodes.length;
                const expandIndex = !!this.state.nestedExpandState?.index;
                const expandBody = !!this.state.nestedExpandState?.body;
                const pathKey = ['ROOT', ...path].join(' > ');

                const crumbHtml = ['ROOT', ...path].map((name, i) => {
                    const idx = i - 1;
                    return `<button class="nested-bc-btn" data-nested-bc-idx="${idx}">${this.escapeHtml(String(name))}</button>`;
                }).join('<span class="nested-bc-sep">></span>');

                const indexNodesHtml = indexNodes.map((node, idx) => {
                    const label = node.key.length > 18 ? `${node.key.slice(0, 18)}...` : node.key;
                    const dragId = `idx::${pathKey}::${node.key}`;
                    const off = this._nestedGetOffset(dragId);
                    const isSemanticFocus = highlightCurrentLevel && String(node.key) === focusIndexKey;
                    return `<div class="nested-box index-chip ${isSemanticFocus ? 'is-semantic-focus' : ''}" data-index-node="${idx}" data-nested-col="${idx}" data-nested-drag-id="${this.escapeHtml(dragId)}" style="--nested-stagger:${idx};transform:translate(${off.x}px, ${off.y}px);" title="${this.escapeHtml(node.key)}">${this.escapeHtml(label)}</div>`;
                }).join('');

                const bodyGroupsHtml = indexNodes.map((indexNode, groupIdx) => {
                    const isSemanticFocus = highlightCurrentLevel && String(indexNode.key) === focusIndexKey;
                    const chips = indexNode.bodyIndices.map((bodyIdx) => {
                        const body = bodyNodes[bodyIdx];
                        const label = body.name.length > 22 ? `${body.name.slice(0, 22)}...` : body.name;
                        const suffix = body.meta ? ` ${body.meta}` : '';
                        const icon = body.isNested ? ' ↘' : '';
                        const dragId = `body::${pathKey}::${indexNode.key}::${body.name}::${bodyIdx}`;
                        const off = this._nestedGetOffset(dragId);
                        return `<button class="nested-box ${body.isNested ? 'clickable' : ''}" data-nested-body-idx="${bodyIdx}" data-nested-col="${groupIdx}" data-nested-drag-id="${this.escapeHtml(dragId)}" style="--nested-stagger:${bodyIdx};transform:translate(${off.x}px, ${off.y}px);" title="${this.escapeHtml(body.name)}">${this.escapeHtml(label)}${this.escapeHtml(suffix)}${icon}</button>`;
                    }).join('');
                    return `<div class="nested-body-col ${isSemanticFocus ? 'is-semantic-focus' : ''}" data-nested-body-col="${groupIdx}" style="--nested-col-stagger:${groupIdx};"><div class="nested-body-group-anchor" data-nested-body-group-anchor="${groupIdx}"></div>${chips || '<div class="nested-muted-note">-</div>'}</div>`;
                }).join('');

                renderHost.innerHTML = `
                    <div class="nested-viewer-fixed-title">Feature Viewer</div>
                    <div class="nested-world-holder" data-nested-viewport data-fixed-viewport-w="${viewportWidth}" data-fixed-viewport-h="${viewportHeight}" data-fixed-canvas-w="${canvasWidth}" data-fixed-canvas-h="${canvasHeight}" style="width:${viewportWidth}px;height:${viewportHeight}px;">
                        <div class="nested-world" data-nested-canvas style="width:${canvasWidth}px;height:${canvasHeight}px;">
                            <div class="nested-feature-wrap">
                                <div class="nested-feature-center">
                                    <div class="nested-feature-stack" data-nested-feature-node style="transform:${featureNodeTransform};">
                                        <div class="nested-breadcrumb">${crumbHtml}</div>
                                        <div class="nested-feature-card">
                                            <div class="nested-stage">
                                                <svg class="nested-stage-svg ${expandIndex && expandBody ? '' : 'is-hidden'}" data-nested-stage-svg></svg>
                                                <div class="nested-surface nested-surface-index ${expandIndex ? 'is-expanded' : ''}">
                                                    <div class="nested-panel-head ${expandIndex ? 'is-expanded' : ''}">
                                                        <span class="nested-pill-btn" data-nested-toggle="index">
                                                            <span class="nested-pill-label">Index Node</span>
                                                            <span class="nested-pill-count">${indexCount}</span>
                                                            <span class="nested-toggle-arrow ${expandIndex ? 'expanded' : ''}">▾</span>
                                                        </span>
                                                    </div>
                                                    <div class="nested-expand-shell ${expandIndex ? 'expanded' : 'collapsed'}" data-nested-expand-shell="index">
                                                        <div class="nested-expand-inner">
                                                            <div class="nested-grid" style="grid-template-columns:repeat(${Math.max(1, indexCount)}, minmax(0,1fr));">
                                                                ${indexNodesHtml || '<div class="nested-muted-note">No index nodes</div>'}
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                                <div class="nested-surface nested-surface-body ${expandBody ? 'is-expanded' : ''}">
                                                    <div class="nested-expand-shell nested-expand-shell-bottom ${expandBody ? 'expanded' : 'collapsed'}" data-nested-expand-shell="body">
                                                        <div class="nested-expand-inner">
                                                            <div class="nested-grid" style="grid-template-columns:repeat(${Math.max(1, indexCount)}, minmax(0,1fr));">
                                                                ${bodyGroupsHtml || '<div class="nested-muted-note">No body nodes</div>'}
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <div class="nested-panel-head nested-panel-head-bottom ${expandBody ? 'is-expanded' : ''}">
                                                        <span class="nested-pill-btn" data-nested-toggle="body">
                                                            <span class="nested-pill-label">Body Node</span>
                                                            <span class="nested-pill-count">${bodyCount}</span>
                                                            <span class="nested-toggle-arrow ${expandBody ? 'expanded' : ''}">▾</span>
                                                        </span>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;

                renderHost.querySelectorAll('[data-nested-toggle]').forEach((el) => {
                    el.addEventListener('click', () => {
                        const key = String(el.getAttribute('data-nested-toggle') || '');
                        const { featureNodeEl } = getNestedDom(renderHost);
                        const beforeRect = featureNodeEl?.getBoundingClientRect() || null;
                        const next = !this.state.nestedExpandState?.[key];
                        this.state.nestedExpandState = {
                            ...this.state.nestedExpandState,
                            [key]: next,
                        };
                        const shell = renderHost.querySelector(`[data-nested-expand-shell="${key}"]`);
                        const head = el.closest('.nested-panel-head');
                        const surface = el.closest('.nested-surface');
                        shell?.classList.toggle('expanded', next);
                        shell?.classList.toggle('collapsed', !next);
                        head?.classList.toggle('is-expanded', next);
                        surface?.classList.toggle('is-expanded', next);
                        const arrow = el.querySelector('.nested-toggle-arrow');
                        arrow?.classList.toggle('expanded', next);
                        this._preserveNestedFeatureNodePosition(renderHost, beforeRect);
                    });
                });

                renderHost.querySelectorAll('[data-nested-bc-idx]').forEach((el) => {
                    el.addEventListener('click', () => {
                        const idx = Number(el.getAttribute('data-nested-bc-idx'));
                        this.state.nestedSemanticFocus = null;
                        if (Number.isNaN(idx) || idx < 0) {
                            this.state.nestedFocusPath = [];
                            this.state.nestedPathStack = [{ key: 'ROOT', data: this._nestedBuildRootFeature() }];
                        } else {
                            this.state.nestedFocusPath = this.state.nestedFocusPath.slice(0, idx + 1);
                            const stack = Array.isArray(this.state.nestedPathStack) ? this.state.nestedPathStack : [];
                            this.state.nestedPathStack = stack.length ? stack.slice(0, idx + 2) : [{ key: 'ROOT', data: this._nestedBuildRootFeature() }];
                        }
                        this.state.nestedExpandState = { index: false, body: false };
                        this.updateTreeViewModeButtons();
                        this.renderNestedFeatureView({ preserveViewport: false });
                    });
                });

                renderHost.querySelectorAll('[data-nested-body-idx]').forEach((el) => {
                    el.addEventListener('click', () => {
                        if (String(el.dataset.dragMoved || '') === '1') return;
                        const idx = Number(el.getAttribute('data-nested-body-idx'));
                        if (Number.isNaN(idx) || idx < 0) return;
                        const target = bodyNodes[idx];
                        if (target && target.isNested && target.nestedData) {
                            this.state.nestedSemanticFocus = null;
                            const nextCrumbLabel = this._nestedBreadcrumbLabelForTarget(target);
                            this.state.nestedFocusPath = [...this.state.nestedFocusPath, nextCrumbLabel];
                            const stack = Array.isArray(this.state.nestedPathStack) && this.state.nestedPathStack.length
                                ? this.state.nestedPathStack
                                : [{ key: 'ROOT', data: this._nestedBuildRootFeature() }];
                            this.state.nestedPathStack = [...stack, { key: nextCrumbLabel, data: target.nestedData }];
                            this.state.nestedExpandState = { index: false, body: false };
                            this.updateTreeViewModeButtons();
                            this.renderNestedFeatureView({ preserveViewport: false });
                        }
                    });
                });

                this._bindNestedColumnGlow(renderHost);
                this._bindNestedDrag(renderHost);
                this._bindNestedViewportZoom(renderHost);
                this._bindNestedViewportPan(renderHost);
                this._bindNestedViewportScroll(renderHost);
                this._bindNestedViewportResize();
                this._bindNestedStageMotion(renderHost);
                this._applyNestedFeatureNodeTransform(renderHost);
                this._applyNestedViewportZoom(renderHost);
                if (shouldCenterViewport) {
                    this.state.nestedViewportScroll = null;
                }
                this._scheduleNestedViewportRestore(renderHost, shouldCenterViewport ? null : prevViewport);
                this._scheduleNestedConnectorRefresh(renderHost);

                requestAnimationFrame(() => {
                    if (!this._isNestedPanelActive()) return;
                    const next = this._getNestedViewportMetrics(renderHost);
                    if (Math.abs(next.viewportWidth - viewportWidth) > 8 || Math.abs(next.viewportHeight - viewportHeight) > 8) {
                        this.renderNestedFeatureView({ preserveViewport });
                    }
                });
            },

            async loadNestedFeatureData() {
                const query = this.state.activeId ? `?conversation_id=${encodeURIComponent(this.state.activeId)}` : '';
                const res = await fetch(`/api/tree-nested-data${query}`);
                const data = await res.json();
                if (!data?.success) return false;
                this.state.nestedRawData = data.data || null;
                this.state.nestedRawDataConversationId = this.state.activeId || null;
                this.state.nestedFocusPath = [];
                this.state.nestedPathStack = [{ key: 'ROOT', data: this._nestedBuildRootFeature() }];
                this.state.nestedExpandState = { index: false, body: false };
                this.state.nestedSemanticFocus = null;
                this.renderNestedFeatureView({ preserveViewport: false });
                this.updateTreeViewModeButtons();
                return true;
            },

            enterNestedLevel() {
                if (this.state.treeViewMode !== 'nested') return;
                const cur = this._getNestedFeatureAtPath();
                if (!cur || typeof cur !== 'object' || Array.isArray(cur)) return;
                const parsed = this._nestedParseFeatureLevel(cur);
                const first = parsed.bodyNodes.find((node) => node && node.isNested && node.nestedData);
                if (!first) return;
                this.state.nestedSemanticFocus = null;
                const nextCrumbLabel = this._nestedBreadcrumbLabelForTarget(first);
                this.state.nestedFocusPath = [...this.state.nestedFocusPath, String(nextCrumbLabel)];
                const stack = Array.isArray(this.state.nestedPathStack) && this.state.nestedPathStack.length
                    ? this.state.nestedPathStack
                    : [{ key: 'ROOT', data: this._nestedBuildRootFeature() }];
                this.state.nestedPathStack = [...stack, { key: String(nextCrumbLabel), data: first.nestedData }];
                this.state.nestedExpandState = { index: false, body: false };
                this.updateTreeViewModeButtons();
                this.renderNestedFeatureView({ preserveViewport: false });
            },

            backNestedLevel() {
                if (this.state.treeViewMode !== 'nested') return;
                if (!Array.isArray(this.state.nestedFocusPath) || !this.state.nestedFocusPath.length) return;
                this.state.nestedFocusPath = this.state.nestedFocusPath.slice(0, -1);
                if (Array.isArray(this.state.nestedPathStack) && this.state.nestedPathStack.length > 1) {
                    this.state.nestedPathStack = this.state.nestedPathStack.slice(0, -1);
                } else {
                    this.state.nestedPathStack = [{ key: 'ROOT', data: this._nestedBuildRootFeature() }];
                }
                this.state.nestedSemanticFocus = null;
                this.state.nestedExpandState = { index: false, body: false };
                this.updateTreeViewModeButtons();
                this.renderNestedFeatureView({ preserveViewport: false });
            },
        });
    };
})();

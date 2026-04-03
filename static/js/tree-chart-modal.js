(function installTreeChartModalModule() {
    // Trading-terminal style: long-ish green, short-ish red, net neutral cyan, then accents
    const COLOR_POOL = ['#34d399', '#f87171', '#38bdf8', '#fbbf24', '#a78bfa', '#fb7185'];
    const CHART_BG = '#060a0c';
    const GRID_H = 'rgba(52, 211, 153, 0.10)';
    const GRID_V = 'rgba(100, 116, 139, 0.12)';
    const AXIS_STROKE = 'rgba(71, 85, 105, 0.85)';
    const TICK_COLOR = '#94a3b8';
    const FONT_MONO = '11px ui-monospace, "Cascadia Code", "SF Mono", Menlo, Monaco, Consolas, monospace';
    const FONT_MONO_SM = '10px ui-monospace, "Cascadia Code", "SF Mono", Menlo, Monaco, Consolas, monospace';
    const FONT_MONO_MD = '12px ui-monospace, "Cascadia Code", "SF Mono", Menlo, Monaco, Consolas, monospace';
    const FONT_TITLE = '600 13px ui-monospace, "Cascadia Code", "SF Mono", Menlo, Monaco, Consolas, monospace';

    const isPlainObject = (value) => !!value && typeof value === 'object' && !Array.isArray(value);

    const normalizeText = (value) => String(value || '')
        .toLowerCase()
        .replace(/\s+/g, ' ')
        .trim();

    const tokenize = (value) => {
        const text = normalizeText(value);
        if (!text) return [];
        const tokens = text.match(/[a-z0-9]+|[\u4e00-\u9fff]+/g) || [];
        return Array.from(new Set(tokens.filter(Boolean)));
    };

    const parseNumeric = (value) => {
        const raw = String(value === null || value === undefined ? '' : value).trim();
        if (!raw) return null;
        const cleaned = raw.replace(/,/g, '');
        if (!/^[-+]?\d*\.?\d+$/.test(cleaned)) return null;
        const num = Number(cleaned);
        return Number.isFinite(num) ? num : null;
    };

    const isNumericArray = (arr) => Array.isArray(arr) && arr.length > 0 && arr.every((item) => parseNumeric(item) !== null);
    const isNonNumericArray = (arr) => Array.isArray(arr) && arr.length > 0 && arr.every((item) => parseNumeric(item) === null);
    const isTotalLikeKey = (key) => /(total|grand\s*total|合计|总计|汇总|总)/i.test(String(key || '').trim());

    const collectChartCandidates = (root, path = []) => {
        const out = [];
        if (!isPlainObject(root)) return out;
        const entries = Object.entries(root);
        const allArray = entries.length >= 2 && entries.every(([, value]) => Array.isArray(value));
        if (allArray) {
            out.push({
                path: path.slice(),
                label: String(path[path.length - 1] || '').trim(),
                obj: root
            });
        }
        entries.forEach(([key, value]) => {
            if (isPlainObject(value)) {
                out.push(...collectChartCandidates(value, path.concat([String(key)])));
            }
        });
        return out;
    };

    const scoreCandidate = (targetName, candidateLabel) => {
        const t = normalizeText(targetName);
        const c = normalizeText(candidateLabel);
        if (!t || !c) return 0;
        if (t === c) return 1000;
        let score = 0;
        if (c.includes(t)) score += 400;
        if (t.includes(c)) score += 240;
        const tTokens = tokenize(t);
        const cTokens = tokenize(c);
        const cSet = new Set(cTokens);
        const overlap = tTokens.filter((token) => cSet.has(token)).length;
        score += overlap * 35;
        return score;
    };

    const buildChartSeriesFromObject = (obj, options = {}) => {
        const includeTotalColumns = options.includeTotalColumns !== false;
        const entries = Object.entries(obj || {});
        if (!entries.length) {
            return { ok: false, error: 'No matched object found' };
        }
        if (!entries.every(([, value]) => Array.isArray(value))) {
            return { ok: false, error: 'No matched object found' };
        }

        const lengths = entries.map(([, value]) => value.length);
        const uniqueLengths = Array.from(new Set(lengths));
        if (uniqueLengths.length !== 1 || uniqueLengths[0] <= 0) {
            return { ok: false, error: 'Column lengths are inconsistent' };
        }
        const seriesLen = uniqueLengths[0];

        const nonNumericEntries = entries.filter(([, value]) => isNonNumericArray(value));
        if (nonNumericEntries.length > 1) {
            return { ok: false, error: 'More than one non-numeric key detected' };
        }

        let numericEntries = entries.filter(([, value]) => isNumericArray(value));
        if (!includeTotalColumns) {
            numericEntries = numericEntries.filter(([key]) => !isTotalLikeKey(key));
        }
        if (!numericEntries.length) {
            return { ok: false, error: 'No usable numeric column' };
        }

        const labels = nonNumericEntries.length === 1
            ? nonNumericEntries[0][1].map((item) => String(item || '').trim() || '-')
            : Array.from({ length: seriesLen }, (_, idx) => `[${idx + 1}]`);

        const xLabels = numericEntries.map(([key]) => String(key || '').trim() || '-');
        const lines = labels.map((label, lineIdx) => ({
            name: label,
            values: numericEntries.map(([, arr]) => parseNumeric(arr[lineIdx])),
        }));

        const hasNull = lines.some((line) => line.values.some((item) => item === null));
        if (hasNull) {
            return { ok: false, error: 'No usable numeric column' };
        }

        return {
            ok: true,
            data: {
                xLabels,
                lines
            }
        };
    };

    window.installTreeChartModal = function installTreeChartModal(app) {
        Object.assign(app, {
            setupTreeChartControls() {
                const openBtn = document.getElementById('tree-generate-chart-btn');
                const closeBtn = document.getElementById('tree-chart-modal-close');
                const fullscreenBtn = document.getElementById('tree-chart-fullscreen-btn');
                const saveBtn = document.getElementById('tree-chart-save-btn');
                const overlay = document.getElementById('tree-chart-modal-overlay');
                const modal = document.getElementById('tree-chart-modal');
                const totalYesBtn = document.getElementById('tree-chart-total-yes');
                const totalNoBtn = document.getElementById('tree-chart-total-no');

                if (openBtn) {
                    openBtn.addEventListener('click', () => this.generateChartFromTreeSelection());
                }
                if (closeBtn) {
                    closeBtn.addEventListener('click', () => this.closeTreeChartModal());
                }
                if (fullscreenBtn) {
                    fullscreenBtn.addEventListener('click', () => this.toggleTreeChartFullscreen());
                }
                if (overlay) {
                    overlay.addEventListener('click', () => this.closeTreeChartModal());
                }
                if (saveBtn) {
                    saveBtn.addEventListener('click', () => this.saveTreeChartImage());
                }
                if (totalYesBtn) {
                    totalYesBtn.addEventListener('click', () => this.resolveTreeChartTotalConfirm(true));
                }
                if (totalNoBtn) {
                    totalNoBtn.addEventListener('click', () => this.resolveTreeChartTotalConfirm(false));
                }
                document.addEventListener('keydown', (ev) => {
                    if (ev.key === 'Escape' && modal && !modal.classList.contains('hidden')) {
                        this.closeTreeChartModal();
                    }
                });
            },

            closeTreeChartModal() {
                const modal = document.getElementById('tree-chart-modal');
                if (!modal) return;
                this.resolveTreeChartTotalConfirm(false);
                this.toggleTreeChartFullscreen(false);
                modal.classList.add('hidden');
            },

            openTreeChartModal() {
                const modal = document.getElementById('tree-chart-modal');
                if (!modal) return;
                modal.classList.remove('hidden');
            },

            toggleTreeChartFullscreen(force = null) {
                const card = document.getElementById('tree-chart-modal-card');
                const btn = document.getElementById('tree-chart-fullscreen-btn');
                if (!card || !btn) return;
                const next = force === null ? !this.state.treeChartFullscreen : !!force;
                this.state.treeChartFullscreen = next;
                card.classList.toggle('max-w-7xl', !next);
                card.classList.toggle('max-h-[94vh]', !next);
                card.classList.toggle('w-[98vw]', next);
                card.classList.toggle('h-[96vh]', next);
                card.classList.toggle('max-w-none', next);
                card.classList.toggle('max-h-none', next);
                btn.textContent = next ? 'Exit Fullscreen' : 'Fullscreen';
                if (this._lastTreeChartData) {
                    setTimeout(() => this.drawLineChartOnCanvas(this._lastTreeChartData), 50);
                }
            },

            resolveTreeChartTotalConfirm(choice) {
                const wrap = document.getElementById('tree-chart-total-confirm');
                if (wrap) wrap.classList.add('hidden');
                const resolver = this._treeChartTotalConfirmResolver;
                this._treeChartTotalConfirmResolver = null;
                if (typeof resolver === 'function') {
                    resolver(!!choice);
                }
            },

            askIncludeTotalColumns(totalKeys = []) {
                const wrap = document.getElementById('tree-chart-total-confirm');
                const text = document.getElementById('tree-chart-total-confirm-text');
                if (!wrap || !text) {
                    return Promise.resolve(true);
                }
                const previewKeys = totalKeys.slice(0, 2).map((item) => String(item || '').trim()).filter(Boolean);
                const keyText = previewKeys.length ? `（${previewKeys.join('、')}）` : '';
                text.textContent = `Detected Total-like columns ${keyText}. Include them in the chart?`;
                wrap.classList.remove('hidden');
                return new Promise((resolve) => {
                    this._treeChartTotalConfirmResolver = resolve;
                });
            },

            setTreeChartError(message) {
                const errorEl = document.getElementById('tree-chart-error');
                const canvas = document.getElementById('tree-chart-canvas');
                const saveBtn = document.getElementById('tree-chart-save-btn');
                if (errorEl) {
                    errorEl.textContent = String(message || 'Failed to generate chart');
                    errorEl.classList.remove('hidden');
                }
                if (canvas) {
                    canvas.classList.add('hidden');
                }
                if (saveBtn) {
                    saveBtn.disabled = true;
                    saveBtn.classList.add('opacity-50', 'cursor-not-allowed');
                }
            },

            clearTreeChartError() {
                const errorEl = document.getElementById('tree-chart-error');
                const canvas = document.getElementById('tree-chart-canvas');
                if (errorEl) {
                    errorEl.textContent = '';
                    errorEl.classList.add('hidden');
                }
                if (canvas) {
                    canvas.classList.remove('hidden');
                }
            },

            saveTreeChartImage() {
                const canvas = document.getElementById('tree-chart-canvas');
                const saveBtn = document.getElementById('tree-chart-save-btn');
                if (!canvas || !saveBtn || saveBtn.disabled) return;
                const link = document.createElement('a');
                const activeId = String(this.state.activeId || '').trim() || 'chart';
                link.download = `tree-line-chart-${activeId}.png`;
                link.href = canvas.toDataURL('image/png');
                link.click();
            },

            getSelectedTreeNodeName() {
                const win = this.getTreeIframeWindow && this.getTreeIframeWindow();
                if (!win) return '';
                const selectedIds = typeof win.getSelectedNodeIds === 'function' ? win.getSelectedNodeIds() : [];
                const selectedId = Array.isArray(selectedIds) && selectedIds.length
                    ? String(selectedIds[selectedIds.length - 1] || '').trim()
                    : '';
                if (!selectedId) return '';
                const roots = typeof win.getTreeData === 'function' ? win.getTreeData() : [];
                const walk = (nodes) => {
                    for (const node of (Array.isArray(nodes) ? nodes : [])) {
                        if (!node || typeof node !== 'object') continue;
                        if (String(node.id || '') === selectedId) return String(node.name || '').trim();
                        const found = walk(node.children || []);
                        if (found) return found;
                    }
                    return '';
                };
                return walk(roots) || '';
            },

            findBestChartObjectByNodeName(rawColumn, nodeName) {
                const root = isPlainObject(rawColumn) ? rawColumn : null;
                if (!root) return null;
                const candidates = collectChartCandidates(root, []);
                if (!candidates.length) return null;
                const target = String(nodeName || '').trim();
                if (!target) return null;

                let best = null;
                candidates.forEach((candidate) => {
                    const label = candidate.label || (candidate.path[candidate.path.length - 1] || '');
                    const score = scoreCandidate(target, label);
                    if (!best || score > best.score) {
                        best = { ...candidate, score };
                    }
                });
                if (!best || best.score < 120) return null;
                return best.obj;
            },

            findTotalLikeNumericKeys(obj) {
                return Object.entries(obj || {})
                    .filter(([key, value]) => isNumericArray(value) && isTotalLikeKey(key))
                    .map(([key]) => String(key || '').trim())
                    .filter(Boolean);
            },

            drawLineChartOnCanvas(chartData) {
                const canvas = document.getElementById('tree-chart-canvas');
                const saveBtn = document.getElementById('tree-chart-save-btn');
                if (!canvas || !chartData) return;
                this._lastTreeChartData = chartData;
                this.clearTreeChartError();
                if (saveBtn) {
                    saveBtn.disabled = false;
                    saveBtn.classList.remove('opacity-50', 'cursor-not-allowed');
                }

                const dpr = Math.max(1, window.devicePixelRatio || 1);
                const xLabels = chartData.xLabels || [];
                const lines = chartData.lines || [];
                const shouldTiltXLabels = xLabels.length > 5 || xLabels.some((label) => String(label || '').length > 6);
                const wrapper = canvas.parentElement;
                const wrapperWidth = Number((wrapper && wrapper.clientWidth) || 1040);
                const cssWidth = Math.max(880, Math.min(1620, wrapperWidth - 16));
                const cssHeight = shouldTiltXLabels
                    ? Math.max(640, Math.round(cssWidth * 0.62))
                    : Math.max(540, Math.round(cssWidth * 0.54));
                canvas.width = Math.floor(cssWidth * dpr);
                canvas.height = Math.floor(cssHeight * dpr);
                canvas.style.width = `${cssWidth}px`;
                canvas.style.height = `${cssHeight}px`;
                const ctx = canvas.getContext('2d');
                if (!ctx) return;
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                ctx.clearRect(0, 0, cssWidth, cssHeight);

                ctx.fillStyle = CHART_BG;
                ctx.fillRect(0, 0, cssWidth, cssHeight);

                const allValues = lines.flatMap((line) => line.values || []);
                let minY = Math.min(...allValues);
                let maxY = Math.max(...allValues);
                if (!Number.isFinite(minY) || !Number.isFinite(maxY)) {
                    this.setTreeChartError('No usable numeric column');
                    return;
                }
                if (minY === maxY) {
                    minY -= 1;
                    maxY += 1;
                }
                const pad = (maxY - minY) * 0.08;
                minY -= pad;
                maxY += pad;
                // For non-negative datasets, keep y-axis baseline at 0.
                if (allValues.every((value) => Number(value) >= 0)) {
                    minY = Math.max(0, minY);
                }

                const formatNumber = (value) => {
                    const num = Number(value);
                    if (!Number.isFinite(num)) return String(value);
                    if (Math.abs(num) >= 1000) return Math.round(num).toLocaleString('en-US');
                    return Number.isInteger(num) ? String(num) : num.toFixed(2);
                };

                // Dynamic left margin from Y tick width + extra room for slanted X labels at first tick
                ctx.font = FONT_MONO;
                let maxYTickW = 0;
                for (let i = 0; i <= 4; i += 1) {
                    const ratio = i / 4;
                    const tickVal = maxY - (maxY - minY) * ratio;
                    const w = ctx.measureText(formatNumber(tickVal)).width;
                    if (w > maxYTickW) maxYTickW = w;
                }
                const leftBase = Math.ceil(maxYTickW) + 28;
                const margin = {
                    top: 72,
                    right: 40,
                    bottom: shouldTiltXLabels ? 150 : 72,
                    left: Math.max(shouldTiltXLabels ? 100 : 84, leftBase),
                };
                const plotW = cssWidth - margin.left - margin.right;
                const plotH = cssHeight - margin.top - margin.bottom;

                // Horizontal inset so first/last slanted date labels are not clipped
                const xEdgeGutter = shouldTiltXLabels ? 28 : 10;
                const plotInnerW = Math.max(40, plotW - 2 * xEdgeGutter);
                const toX = (idx) => {
                    if (xLabels.length <= 1) return margin.left + plotW / 2;
                    return margin.left + xEdgeGutter + (idx * plotInnerW) / (xLabels.length - 1);
                };
                const toY = (v) => margin.top + ((maxY - v) / (maxY - minY)) * plotH;

                // Horizontal grid (subtle green tint, dashed)
                ctx.strokeStyle = GRID_H;
                ctx.lineWidth = 1;
                ctx.setLineDash([4, 6]);
                for (let i = 0; i <= 4; i += 1) {
                    const y = margin.top + (plotH * i) / 4;
                    ctx.beginPath();
                    ctx.moveTo(margin.left, y);
                    ctx.lineTo(margin.left + plotW, y);
                    ctx.stroke();
                }
                // Vertical grid on x ticks
                ctx.strokeStyle = GRID_V;
                for (let i = 0; i < xLabels.length; i += 1) {
                    const x = toX(i);
                    ctx.beginPath();
                    ctx.moveTo(x, margin.top);
                    ctx.lineTo(x, margin.top + plotH);
                    ctx.stroke();
                }
                ctx.setLineDash([]);

                ctx.strokeStyle = AXIS_STROKE;
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(margin.left, margin.top);
                ctx.lineTo(margin.left, margin.top + plotH);
                ctx.lineTo(margin.left + plotW, margin.top + plotH);
                ctx.stroke();

                lines.forEach((line, lineIdx) => {
                    const color = COLOR_POOL[lineIdx % COLOR_POOL.length];
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2.2;
                    ctx.beginPath();
                    (line.values || []).forEach((value, idx) => {
                        const x = toX(idx);
                        const y = toY(value);
                        if (idx === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    });
                    ctx.stroke();

                    ctx.fillStyle = color;
                    (line.values || []).forEach((value, idx) => {
                        const x = toX(idx);
                        const y = toY(value);
                        ctx.beginPath();
                        ctx.arc(x, y, 3, 0, Math.PI * 2);
                        ctx.fill();

                        // Value labels near points
                        const labelText = formatNumber(value);
                        ctx.font = FONT_MONO_SM;
                        ctx.textAlign = 'center';
                        const yOffset = lineIdx % 2 === 0 ? -10 : 14;
                        const tx = x;
                        const ty = y + yOffset;
                        const textW = ctx.measureText(labelText).width;
                        const bgW = textW + 8;
                        const bgH = 16;
                        ctx.fillStyle = 'rgba(6, 10, 12, 0.92)';
                        ctx.fillRect(tx - (bgW / 2), ty - 12, bgW, bgH);
                        ctx.strokeStyle = 'rgba(52, 211, 153, 0.25)';
                        ctx.lineWidth = 1;
                        ctx.strokeRect(tx - (bgW / 2), ty - 12, bgW, bgH);
                        ctx.fillStyle = '#e5e7eb';
                        ctx.fillText(labelText, tx, ty);
                        ctx.fillStyle = color;
                    });
                });

                ctx.fillStyle = TICK_COLOR;
                ctx.font = FONT_MONO_MD;
                xLabels.forEach((label, idx) => {
                    const x = toX(idx);
                    const text = String(label || '');
                    if (!shouldTiltXLabels) {
                        ctx.textAlign = 'center';
                        ctx.fillText(text, x, margin.top + plotH + 22);
                        return;
                    }
                    // Draw x-axis labels in a slanted style for dense timelines.
                    ctx.save();
                    ctx.translate(x - 4, margin.top + plotH + 30);
                    ctx.rotate(-Math.PI / 3); // ~60deg tilt, close to financial chart style
                    ctx.textAlign = 'right';
                    ctx.fillText(text, 0, 0);
                    ctx.restore();
                });

                ctx.textAlign = 'right';
                ctx.fillStyle = TICK_COLOR;
                ctx.font = FONT_MONO;
                for (let i = 0; i <= 4; i += 1) {
                    const ratio = i / 4;
                    const value = maxY - (maxY - minY) * ratio;
                    const y = margin.top + plotH * ratio;
                    ctx.fillText(formatNumber(value), margin.left - 8, y + 4);
                }

                ctx.textAlign = 'left';
                ctx.fillStyle = '#cbd5e1';
                ctx.font = FONT_TITLE;
                ctx.fillText('Exposure Chart', margin.left, 30);

                let legendX = margin.left;
                let legendY = 48;
                lines.forEach((line, lineIdx) => {
                    const color = COLOR_POOL[lineIdx % COLOR_POOL.length];
                    const text = String(line.name || `[${lineIdx + 1}]`);
                    ctx.fillStyle = color;
                    ctx.fillRect(legendX, legendY - 8, 12, 12);
                    ctx.fillStyle = '#cbd5e1';
                    ctx.font = FONT_MONO_MD;
                    ctx.fillText(text, legendX + 18, legendY + 2);
                    legendX += Math.min(240, Math.max(90, text.length * 9 + 40));
                    if (legendX > cssWidth - margin.right - 140) {
                        legendX = margin.left;
                        legendY += 20;
                    }
                });
            },

            async generateChartFromTreeSelection() {
                this.openTreeChartModal();

                if ((this.state.treeViewMode || 'flat') !== 'flat') {
                    this.setTreeChartError('Only flat view is supported (row/column)');
                    return;
                }

                const nodeName = this.getSelectedTreeNodeName();
                if (!nodeName) {
                    this.setTreeChartError('Please select a node in the tree first');
                    return;
                }

                const activeId = String(this.state.activeId || '').trim();
                if (!activeId) {
                    this.setTreeChartError('Conversation data not found');
                    return;
                }

                try {
                    const query = `?conversation_id=${encodeURIComponent(activeId)}`;
                    const res = await fetch(`/api/tree-nested-data${query}`);
                    const payload = await res.json();
                    if (!payload || !payload.success || !payload.data) {
                        this.setTreeChartError('No matched object found');
                        return;
                    }

                    const targetObject = this.findBestChartObjectByNodeName(payload.data, nodeName);
                    if (!targetObject) {
                        this.setTreeChartError('No matched object found');
                        return;
                    }

                    let includeTotalColumns = true;
                    const totalLikeKeys = this.findTotalLikeNumericKeys(targetObject);
                    if (totalLikeKeys.length) {
                        includeTotalColumns = await this.askIncludeTotalColumns(totalLikeKeys);
                    }

                    const parsed = buildChartSeriesFromObject(targetObject, { includeTotalColumns });
                    if (!parsed.ok) {
                        this.setTreeChartError(parsed.error || 'Failed to generate chart');
                        return;
                    }
                    this.drawLineChartOnCanvas(parsed.data);
                } catch (e) {
                    this.setTreeChartError('Failed to generate chart');
                }
            },
        });
    };
})();

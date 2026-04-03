(function initAppTraceSyncModule(global) {
    // Trace synchronization and playback glue logic.
    // Data sources still come from /api/chain and iframe methods installed by /static/js/trace-stage-viewer.js.
    global.AppTraceSyncMethods = {
        updateTracePlaybackButton() {
            const btn = document.getElementById('play-trace-btn');
            if (!btn) return;
            btn.textContent = this.state.isTracePlaying ? 'Stop Replay' : 'Replay Path';
        },

        updateStrictTraceWarning(isStrictTrace) {
            const tip = document.getElementById('strict-trace-warning');
            if (!tip) return;
            tip.classList.toggle('hidden', !!isStrictTrace);
        },

        updateTraceDebugOverlay(traceInfo, backendTrace = null) {
            const wrap = document.getElementById('trace-debug-overlay');
            const content = document.getElementById('trace-debug-content');
            if (!wrap || !content) return;
            // 调试输出面板已关闭，避免干扰主界面。
            wrap.classList.add('hidden');
            content.textContent = '';
        },

        getTreeIframeWindow() {
            const iframe = document.querySelector('#tree-content iframe');
            if (!iframe || !iframe.contentWindow) return null;
            return iframe.contentWindow;
        },

        setTracePlaybackSpeed(speed) {
            const parsed = Number(speed);
            if (!Number.isFinite(parsed) || parsed <= 0) return;
            this.state.tracePlaybackSpeed = parsed;
            const win = this.getTreeIframeWindow();
            if (win && typeof win.setTracePlaybackSpeed === 'function') {
                try {
                    win.setTracePlaybackSpeed(parsed);
                } catch (e) {
                    // ignore
                }
            }
        },

        stopTraceStageAppend() {
            if (this.state.traceStageAppendTimer) {
                clearTimeout(this.state.traceStageAppendTimer);
                this.state.traceStageAppendTimer = null;
            }
        },

        toggleTracePlayback() {
            const win = this.getTreeIframeWindow();
            if (!win) return;
            try {
                this.setTracePlaybackSpeed(this.state.tracePlaybackSpeed);
                if (this.state.isTracePlaying && typeof win.stopTracePlayback === 'function') {
                    win.stopTracePlayback();
                    this.state.isTracePlaying = false;
                    this.stopTraceStageAppend();
                    this.updateTracePlaybackButton();
                    return;
                }
                if (typeof win.startTracePlayback === 'function') {
                    const res = win.startTracePlayback();
                    this.state.isTracePlaying = !!res?.started;
                    if (this.state.isTracePlaying) {
                        this.toggleTraceStageDrawer(true);
                        this.renderTraceStageResults(this.state.lastExecutionTrace || [], { append: true });
                    }
                    this.updateTracePlaybackButton();
                    if (this.state.isTracePlaying) {
                        const interval = Number(res?.interval_ms || (420 / (this.state.tracePlaybackSpeed || 1)));
                        const duration = Math.max(900, (Number(res?.steps || 0) * interval) + 260);
                        setTimeout(() => {
                            this.state.isTracePlaying = false;
                            this.updateTracePlaybackButton();
                        }, duration);
                    }
                }
            } catch (e) {
                this.state.isTracePlaying = false;
                this.updateTracePlaybackButton();
            }
        },

        autoPlayTrace() {
            const win = this.getTreeIframeWindow();
            if (!win || typeof win.startTracePlayback !== 'function') return;
            try {
                this.setTracePlaybackSpeed(this.state.tracePlaybackSpeed);
                const res = win.startTracePlayback();
                this.state.isTracePlaying = !!res?.started;
                if (this.state.isTracePlaying) {
                    if (this.state.traceStageDrawerOpen) {
                        this.toggleTraceStageDrawer(true);
                    }
                    this.renderTraceStageResults(this.state.lastExecutionTrace || [], { append: true });
                }
                this.updateTracePlaybackButton();
                if (this.state.isTracePlaying) {
                    const interval = Number(res?.interval_ms || (420 / (this.state.tracePlaybackSpeed || 1)));
                    const duration = Math.max(900, (Number(res?.steps || 0) * interval) + 260);
                    setTimeout(() => {
                        this.state.isTracePlaying = false;
                        this.updateTracePlaybackButton();
                    }, duration);
                }
            } catch (e) {
                this.state.isTracePlaying = false;
                this.updateTracePlaybackButton();
            }
        },

        clearQaTrace() {
            this.state.pendingQaTrace = null;
            this.state.lastQaTrace = null;
            this.state.isTracePlaying = false;
            this.state.lastExecutionTrace = [];
            this.stopTraceStageAppend();
            const win = this.getTreeIframeWindow();
            if (!win || typeof win.clearQaTrace !== 'function') {
                this.updateTracePlaybackButton();
                return;
            }
            try {
                if (typeof win.stopTracePlayback === 'function') {
                    win.stopTracePlayback();
                }
                win.clearQaTrace();
            } catch (e) {
                // ignore
            }
            this.updateTracePlaybackButton();
        },

        async debugHighlightExact(canonicalId, options = {}) {
            const cid = String(canonicalId || '').trim();
            if (!cid) return { success: false, error: 'canonicalId required' };
            const treeViewMode = String(this.state.treeViewMode || '').trim().toLowerCase();
            const flatMode = String(this.state.flatSubMode || 'column').trim().toLowerCase();
            const viewMode = treeViewMode === 'flat'
                ? (flatMode === 'row' ? 'row' : 'column')
                : 'row';
            const params = new URLSearchParams();
            params.set('conversation_id', this.state.activeId || '');
            params.set('view_mode', viewMode);
            params.set('canonical_id', cid);
            const endpoint = `/api/debug/highlight-exact?${params.toString()}`;
            try {
                const res = await fetch(endpoint);
                const data = await res.json();
                if (!data?.success) return data || { success: false, error: 'request failed' };
                const payload = data.highlight_payload || null;
                const win = await this.ensureTracePlaybackSurface({ resetTrace: true, reapplyTrace: false });
                if (!win || !payload) return { ...data, applied: false, reason: 'iframe not ready' };
                const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
                const tryApply = () => {
                    if (typeof win.flashTraceSegment === 'function') {
                        return win.flashTraceSegment(payload, { duration_ms: 1800 });
                    }
                    if (typeof win.previewTraceSegment === 'function') {
                        return win.previewTraceSegment(payload);
                    }
                    if (typeof win.playTraceSegment === 'function') {
                        return win.playTraceSegment(payload);
                    }
                    return null;
                };
                let applied = tryApply();
                let attempts = 1;
                // 某些情况下 iframe 已就绪但树节点尚未落位，做短重试。
                while ((!applied || applied.applied === false || applied.flashed === false) && attempts < 4) {
                    await sleep(160 * attempts);
                    applied = tryApply();
                    attempts += 1;
                }
                if ((!applied || applied.applied === false || applied.flashed === false) && typeof win.focusNodeById === 'function') {
                    const targetNodeId = String((data && data.target_node_id) || '').trim();
                    if (targetNodeId) {
                        try {
                            win.focusNodeById(targetNodeId);
                        } catch (e) {
                            // ignore
                        }
                    }
                }
                return { ...data, applied: applied || null, attempts };
            } catch (e) {
                return { success: false, error: String(e || 'unknown error') };
            }
        },

        async fetchThinkingChain() {
            try {
                const params = new URLSearchParams();
                if (this.state.activeId) {
                    params.set('conversation_id', this.state.activeId);
                }
                const treeViewMode = String(this.state.treeViewMode || '').trim().toLowerCase();
                const flatMode = String(this.state.flatSubMode || 'column').trim().toLowerCase();
                const viewMode = treeViewMode === 'flat'
                    ? (flatMode === 'row' ? 'row' : 'column')
                    : 'row';
                params.set('view_mode', viewMode);
                const query = params.toString() ? `?${params.toString()}` : '';
                const res = await fetch(`/api/chain${query}`);
                const data = await res.json();
                if (data && data.success) {
                    return {
                        chain: data.chain || {},
                        trace: data.trace || null,
                        executionTrace: data.execution_trace || [],
                        traceV2: data.trace_v2 || null,
                        traceTreeFingerprint: data.trace_tree_fingerprint || data.traceTreeFingerprint || ''
                    };
                }
                return { chain: {}, trace: null, executionTrace: [], traceV2: null, traceTreeFingerprint: '' };
            } catch (e) {
                return { chain: {}, trace: null, executionTrace: [], traceV2: null, traceTreeFingerprint: '' };
            }
        },

        applyPendingQaTrace() {
            if (!this.state.pendingQaTrace) return;
            const win = this.getTreeIframeWindow();
            if (!win || typeof win.applyQaTrace !== 'function') return;
            try {
                const traceInfo = win.applyQaTrace(this.state.pendingQaTrace);
                this.state.lastQaTrace = traceInfo || null;
                this.state.lastExecutionTrace = Array.isArray(this.state.pendingQaTrace?.executionTrace) ? this.state.pendingQaTrace.executionTrace : [];
                this.state.lastPrimitiveSteps = Array.isArray(this.state.pendingQaTrace?.traceV2?.primitiveSteps) ? this.state.pendingQaTrace.traceV2.primitiveSteps : [];
                this.updateStrictTraceWarning(String(traceInfo?.traceMode || '').includes('strict'));
                this.updateTraceDebugOverlay(traceInfo, this.state.pendingQaTrace?.trace || null);
                this.renderTraceStageResults(this.state.pendingQaTrace?.executionTrace || [], { primitiveSteps: this.state.lastPrimitiveSteps });
                this.state.pendingQaTrace = null;
                this.state.isTracePlaying = false;
                this.updateTracePlaybackButton();
                this.autoPlayTrace();
            } catch (e) {
                // ignore
            }
        },

        async syncQaTrace(answerText = "") {
            const tracePayload = await this.fetchThinkingChain();
            const executionTrace = Array.isArray(tracePayload?.executionTrace) ? tracePayload.executionTrace : [];
            const traceV2 = tracePayload?.traceV2 || null;
            const visitNodeOrder = executionTrace
                .map((ev) => String(ev?.canonical_id || ev?.canonical_trace_id || ev?.frontend_node_id || '').trim())
                .filter((nodeId, index) => String(executionTrace[index]?.event_type || '') === 'visit' && nodeId);
            const visitNodeSet = Array.from(new Set(visitNodeOrder));
            const visitEdgeOrder = [];
            for (let i = 1; i < visitNodeOrder.length; i += 1) {
                visitEdgeOrder.push(`${visitNodeOrder[i - 1]}->${visitNodeOrder[i]}`);
            }
            const backendTrace = tracePayload?.trace || null;
            const answerNorm = String(answerText || '').toLowerCase().replace(/\s+/g, '').trim();
            let answerNodeFromVisit = null;
            if (answerNorm) {
                for (let i = executionTrace.length - 1; i >= 0; i -= 1) {
                    const ev = executionTrace[i];
                    const nodeText = String(ev?.node_value || '').toLowerCase().replace(/\s+/g, '').trim();
                    if (!nodeText) continue;
                    if (answerNorm.includes(nodeText) || nodeText.includes(answerNorm)) {
                        answerNodeFromVisit = ev?.canonical_id || ev?.canonical_trace_id || ev?.frontend_node_id || null;
                        if (answerNodeFromVisit) break;
                    }
                }
            }
            const mergedTrace = {
                ...(backendTrace || {}),
                mode: visitNodeOrder.length ? 'strict_visit' : (backendTrace?.mode || 'inferred'),
                matched_node_ids: visitNodeSet.length ? visitNodeSet : (backendTrace?.matched_node_ids || []),
                path_node_order: visitNodeOrder.length ? visitNodeOrder : (backendTrace?.path_node_order || []),
                path_edge_order: visitEdgeOrder.length ? visitEdgeOrder : (backendTrace?.path_edge_order || []),
                answer_node_id: answerNodeFromVisit || backendTrace?.answer_node_id || null
            };
            const payload = {
                chain: tracePayload?.chain || {},
                trace: mergedTrace,
                executionTrace,
                traceV2,
                traceTreeFingerprint: tracePayload?.traceTreeFingerprint || '',
                answerText
            };
            const win = this.getTreeIframeWindow();
            if (win && typeof win.applyQaTrace === 'function') {
                try {
                    const traceInfo = win.applyQaTrace(payload);
                    this.state.lastQaTrace = traceInfo || null;
                    this.state.lastExecutionTrace = Array.isArray(executionTrace) ? executionTrace : [];
                    this.state.lastPrimitiveSteps = Array.isArray(traceV2?.primitiveSteps) ? traceV2.primitiveSteps : [];
                    this.updateStrictTraceWarning(String(traceInfo?.traceMode || '').includes('strict'));
                    this.updateTraceDebugOverlay(traceInfo, payload?.trace || null);
                    this.renderTraceStageResults(executionTrace, { primitiveSteps: this.state.lastPrimitiveSteps });
                    this.state.pendingQaTrace = null;
                    this.state.isTracePlaying = false;
                    this.updateTracePlaybackButton();
                    this.autoPlayTrace();
                    return this.state.lastQaTrace;
                } catch (e) {
                    // fall through to pending
                }
            }
            this.state.pendingQaTrace = payload;
            this.state.lastQaTrace = null;
            return null;
        },

        async refreshChain() {
            const data = await this.fetchThinkingChain();
            const chain = data?.chain || {};
            const executionTrace = Array.isArray(data?.executionTrace) ? data.executionTrace : [];
            document.getElementById('chain-content').textContent = JSON.stringify(chain, null, 2);
            this.renderThoughtHierarchy(chain);
            this.renderExecutionTracePanel(executionTrace);
        },

        renderThoughtHierarchy(chain) {
            const panel = document.getElementById('thought-hierarchy');
            if (!panel) return;
            const qa = chain?.question_answering || {};
            const decomposed = qa?.query_decomposition?.decomposed_queries || [];
            const flags = qa?.query_decomposition?.retrieve_flag || [];
            const subs = qa?.subqueries || [];
            const blocks = [];
            blocks.push(`<div class="cot-step"><strong>Q0</strong>: ${this.escapeHtml(String(qa?.raw_query || ''))}</div>`);
            if (decomposed.length) {
                blocks.push(`<div class="cot-step"><strong>Query Decomposition</strong><br>${decomposed.map((q, i) => `${i + 1}. ${this.escapeHtml(String(q))} ${flags[i] ? '(retrieval)' : '(direct)'}`).join('<br>')}</div>`);
            }
            subs.forEach((sq) => {
                const rp = Array.isArray(sq?.reasoning_path) ? sq.reasoning_path.join(' | ') : String(sq?.reasoning_path || '');
                blocks.push(`<div class="cot-step"><strong>SubQ${sq?.index || '?'}</strong>: ${this.escapeHtml(String(sq?.query || ''))}<br><span class="text-white/70">Primitive:</span> ${this.escapeHtml(rp)}<br><span class="text-white/70">Answer:</span> ${this.escapeHtml(String(sq?.answer || ''))}</div>`);
            });
            if (qa?.final_answer !== undefined) {
                blocks.push(`<div class="cot-step"><strong>Final Answer</strong>: ${this.escapeHtml(String(qa.final_answer))}</div>`);
            }
            panel.innerHTML = blocks.join('') || '<div class="text-xs text-white/50">No reasoning hierarchy data.</div>';
        },

        renderExecutionTracePanel(executionTrace) {
            const panel = document.getElementById('execution-trace-panel');
            if (!panel) return;
            if (!executionTrace.length) {
                panel.innerHTML = '<div class="text-xs text-white/50">No node visit events.</div>';
                return;
            }
            const existingDetails = panel.querySelector('details[data-trace-details="1"]');
            const keepOpen = existingDetails ? existingDetails.open : !!this.state.executionTraceDetailsOpen;
            const keyTypes = new Set(['subquery_start', 'primitive_generated', 'primitive_execute', 'retrieval_result', 'subquery_end']);
            const keyLines = executionTrace
                .filter((ev) => keyTypes.has(String(ev?.event_type || '')))
                .slice(-80)
                .map((ev) => `<div class="trace-key-line">#${ev.step} [${this.escapeHtml(String(ev.event_type || ''))}] ${this.escapeHtml(String(ev.node_value || ''))}</div>`)
                .join('');
            const detailLines = executionTrace
                .slice(-300)
                .map((ev) => {
                    const ctx = ev?.context || {};
                    const loc = [ctx?.subquery_index ? `subQ${ctx.subquery_index}` : '', ctx?.depth ? `d${ctx.depth}` : ''].filter(Boolean).join(' · ');
                    const fidValue = ev?.canonical_id || ev?.canonical_trace_id || ev?.frontend_node_id || '';
                    const fid = fidValue ? ` -> ${fidValue}` : '';
                    return `<div class="trace-detail-item">#${ev.step} [${this.escapeHtml(String(ev.event_type || ''))}] ${this.escapeHtml(String(ev.node_type || ''))}: ${this.escapeHtml(String(ev.node_value || ''))}${this.escapeHtml(fid)} ${loc ? `<span class="text-white/50">(${this.escapeHtml(loc)})</span>` : ''}</div>`;
                })
                .join('');
            panel.innerHTML = `
                <div class="text-[11px] text-white/60 mb-2">Key steps are shown by default. Expand to view all visit events.</div>
                ${keyLines || '<div class="text-xs text-white/50">No key steps.</div>'}
                <details class="mt-2" data-trace-details="1" ${keepOpen ? 'open' : ''}>
                    <summary class="text-xs text-white/60 cursor-pointer">Expand all visit events</summary>
                    <div class="mt-2">${detailLines}</div>
                </details>
            `;
            const detailsEl = panel.querySelector('details[data-trace-details="1"]');
            if (detailsEl) {
                detailsEl.addEventListener('toggle', () => {
                    this.state.executionTraceDetailsOpen = detailsEl.open;
                });
            }
        },

        renderTraceStageResults(executionTrace, options = {}) {
            const list = document.getElementById('trace-stage-list');
            if (!list) return;
            this.stopTraceStageAppend();
            const events = Array.isArray(executionTrace) ? executionTrace : [];
            const primitiveSteps = Array.isArray(options?.primitiveSteps) ? options.primitiveSteps : [];
            if (primitiveSteps.length) {
                const stepHtml = primitiveSteps.slice(-80).map((step) => {
                    const args = Array.isArray(step?.args) ? step.args.map((x) => this.escapeHtml(String(x))).join(', ') : '';
                    const visited = Array.isArray(step?.visitedNodeIds) ? step.visitedNodeIds.length : 0;
                    const retrieved = Array.isArray(step?.retrievalNodeIds) ? step.retrievalNodeIds.length : 0;
                    return `<div class="trace-stage-item"><div class="trace-stage-title">#${this.escapeHtml(String(step?.index ?? '-'))} [${this.escapeHtml(String(step?.type || ''))}]</div><div class="trace-stage-sub">args: ${args || '-'} · visited: ${visited} · retrieved: ${retrieved}</div></div>`;
                }).join('');
                if (!events.length) {
                    list.innerHTML = stepHtml;
                    return;
                }
                list.innerHTML = `${stepHtml}<details class="mt-2"><summary class="text-xs text-white/70 cursor-pointer">Expand event details</summary><div id="trace-stage-detail-lines" class="mt-2"></div></details>`;
                const detailBox = document.getElementById('trace-stage-detail-lines');
                if (detailBox) {
                    detailBox.innerHTML = events.slice(-300).map((ev) => {
                        const ctx = ev?.context || {};
                        const loc = [ctx?.subquery_index ? `subQ${ctx.subquery_index}` : '', ctx?.depth ? `d${ctx.depth}` : ''].filter(Boolean).join(' · ');
                        const fidValue = ev?.canonical_id || ev?.canonical_trace_id || ev?.frontend_node_id || '';
                        const fid = fidValue ? ` -> ${fidValue}` : '';
                        return `<div class="trace-detail-item">#${this.escapeHtml(String(ev?.step ?? '-'))} [${this.escapeHtml(String(ev?.event_type || ''))}] ${this.escapeHtml(String(ev?.node_type || ''))}: ${this.escapeHtml(String(ev?.node_value || ''))}${this.escapeHtml(fid)} ${loc ? `<span class="text-white/50">(${this.escapeHtml(loc)})</span>` : ''}</div>`;
                    }).join('');
                }
                return;
            }
            if (!events.length) {
                list.innerHTML = '<div class="text-xs text-white/45">No stage results.</div>';
                return;
            }
            const appendMode = !!options?.append;
            const keyTypes = new Set(['subquery_start', 'primitive_generated', 'primitive_execute', 'retrieval_result', 'subquery_end']);
            const formatCoreText = (ev) => {
                const t = String(ev?.event_type || '');
                const value = String(ev?.node_value || '');
                const nodeType = String(ev?.node_type || '');
                if (t === 'subquery_start' || t === 'subquery_end') return value ? `subquery: ${value}` : 'subquery';
                if (t === 'primitive_generated' || t === 'primitive_execute') return value ? `primitive: ${value}` : 'primitive';
                if (t === 'retrieval_result') return value ? `result: ${value}` : 'result';
                if (t === 'retrieval_item') return value ? `data: ${value}` : 'data';
                if (t === 'visit') return `${nodeType ? `${nodeType}: ` : ''}${value}`.trim();
                return `${nodeType ? `${nodeType}: ` : ''}${value}`.trim() || '-';
            };
            const formatCtx = (ev) => {
                const ctx = ev?.context || {};
                const loc = [ctx?.subquery_index ? `subQ${ctx.subquery_index}` : '', ctx?.depth ? `d${ctx.depth}` : ''].filter(Boolean).join(' · ');
                return loc ? ` (${this.escapeHtml(loc)})` : '';
            };

            list.innerHTML = `
                <div class="text-[11px] text-white/60 mb-2">Key steps are shown by default. Expand to view all visit events.</div>
                <div id="trace-stage-key-lines"></div>
                <details class="mt-2">
                    <summary class="text-xs text-white/70 cursor-pointer">Expand all visit events</summary>
                    <div id="trace-stage-detail-lines" class="mt-2"></div>
                </details>
            `;
            const keyBox = document.getElementById('trace-stage-key-lines');
            const detailBox = document.getElementById('trace-stage-detail-lines');
            if (!keyBox || !detailBox) return;

            const clippedEvents = events.slice(-500);
            const appendKeyLine = (ev) => {
                const row = document.createElement('div');
                row.className = 'trace-stage-item';
                row.innerHTML = `<div class="trace-stage-title">#${this.escapeHtml(String(ev?.step ?? '-'))} [${this.escapeHtml(String(ev?.event_type || ''))}] ${this.escapeHtml(formatCoreText(ev))}</div>`;
                keyBox.appendChild(row);
            };
            const appendDetailLine = (ev) => {
                const row = document.createElement('div');
                row.className = 'trace-detail-item';
                const fidValue = ev?.canonical_id || ev?.canonical_trace_id || ev?.frontend_node_id || '';
                const fid = fidValue ? ` -> ${fidValue}` : '';
                row.innerHTML = `#${this.escapeHtml(String(ev?.step ?? '-'))} [${this.escapeHtml(String(ev?.event_type || ''))}] ${this.escapeHtml(formatCoreText(ev))}${this.escapeHtml(fid)}${formatCtx(ev)}`;
                detailBox.appendChild(row);
            };

            if (!appendMode) {
                const keyEvents = clippedEvents.filter((ev) => keyTypes.has(String(ev?.event_type || ''))).slice(-120);
                if (!keyEvents.length) {
                    keyBox.innerHTML = '<div class="text-xs text-white/45">No key steps.</div>';
                } else {
                    keyEvents.forEach((ev) => appendKeyLine(ev));
                }
                if (!clippedEvents.length) {
                    detailBox.innerHTML = '<div class="text-xs text-white/45">No event details.</div>';
                } else {
                    clippedEvents.forEach((ev) => appendDetailLine(ev));
                }
                return;
            }

            if (!clippedEvents.some((ev) => keyTypes.has(String(ev?.event_type || '')))) {
                keyBox.innerHTML = '<div class="text-xs text-white/45">No key steps.</div>';
            }
            let idx = 0;
            const speed = Number(this.state.tracePlaybackSpeed || 1);
            const tickMs = Math.max(90, Math.round(220 / (speed > 0 ? speed : 1)));
            const pump = () => {
                if (!this.state.traceStageDrawerOpen) {
                    this.state.traceStageAppendTimer = null;
                    return;
                }
                if (idx >= clippedEvents.length) {
                    this.state.traceStageAppendTimer = null;
                    return;
                }
                const ev = clippedEvents[idx];
                if (keyTypes.has(String(ev?.event_type || ''))) {
                    appendKeyLine(ev);
                }
                appendDetailLine(ev);
                idx += 1;
                this.state.traceStageAppendTimer = setTimeout(pump, tickMs);
            };
            this.state.traceStageAppendTimer = setTimeout(pump, tickMs);
        }
    };
})(window);

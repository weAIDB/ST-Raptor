(function installTraceStageViewerModule() {
    const DEFAULT_SUBTREE_EMPTY = '<div class="trace-empty-state">No subtree preview is available for this step.</div>';

    const summarizeValue = (value) => {
        if (value === null || value === undefined || value === '') return '-';
        if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return String(value);
        if (Array.isArray(value)) return value.map((item) => summarizeValue(item)).join(' | ');
        if (typeof value === 'object') {
            if (typeof value.resultCount === 'number' && Array.isArray(value.results)) {
                const previews = value.results.slice(0, 3).map((item) => summarizeValue(item)).filter(Boolean);
                return previews.length ? `Matched ${value.resultCount} items: ${previews.join(' | ')}` : `Matched ${value.resultCount} items`;
            }
            if (typeof value.preview === 'string' && value.preview.trim()) return value.preview;
            if (typeof value.value === 'string' && value.value.trim()) return value.value;
            if (Array.isArray(value.results)) return value.results.map((item) => summarizeValue(item)).join(' | ');
            try {
                return JSON.stringify(value);
            } catch (e) {
                return String(value);
            }
        }
        return String(value);
    };

    const cleanTraceText = (value) => {
        let text = String(value === null || value === undefined ? '' : value).trim();
        if (!text) return '';
        text = text.replace(/^#+\s*/, '').replace(/\s*#+$/, '').trim();
        if (text === 'None') return 'No result';
        return text;
    };

    const dedupeTextLines = (items) => {
        const seen = new Set();
        return (Array.isArray(items) ? items : []).map((item) => cleanTraceText(item)).filter((item) => {
            if (!item || seen.has(item)) return false;
            seen.add(item);
            return true;
        });
    };

    const tryParseLiteral = (value) => {
        const text = String(value === null || value === undefined ? '' : value).trim();
        if (!text) return null;
        try {
            return JSON.parse(text);
        } catch (e) {
            try {
                return JSON.parse(text.replace(/'/g, '"'));
            } catch (err) {
                return null;
            }
        }
    };

    const hasValue = (value) => value !== undefined && value !== null;
    const getValue = (obj, key, fallback) => (obj && hasValue(obj[key]) ? obj[key] : fallback);
    const getArray = (obj, key) => (obj && Array.isArray(obj[key]) ? obj[key] : []);
    const hitClosest = (target, selector) => !!(target && typeof target.closest === 'function' && target.closest(selector));

    window.installTraceStageViewer = function installTraceStageViewer(app) {
        const originalRenderTraceStageResults = typeof app.renderTraceStageResults === 'function'
            ? app.renderTraceStageResults
            : function noop() {};
        const originalToggleTracePlayback = typeof app.toggleTracePlayback === 'function'
            ? app.toggleTracePlayback
            : function noop() {};
        const originalAutoPlayTrace = typeof app.autoPlayTrace === 'function'
            ? app.autoPlayTrace
            : function noop() {};
        const originalClearQaTrace = typeof app.clearQaTrace === 'function'
            ? app.clearQaTrace
            : function noop() {};

        Object.assign(app, {
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
                        const canonicalProjectionMap = data.canonical_projection_map || data.canonicalProjectionMap || null;
                        const nestedIndexProjectionMap = data.nested_index_projection_map
                            || data.nestedIndexProjectionMap
                            || (canonicalProjectionMap && (canonicalProjectionMap.nestedIndexProjectionMap || canonicalProjectionMap.nested_index_projection_map))
                            || null;
                        return {
                            chain: data.chain || {},
                            trace: data.trace || null,
                            executionTrace: data.execution_trace || [],
                            traceV2: data.trace_v2 || null,
                            traceV3: data.trace_v3 || null,
                            traceTreeFingerprint: data.trace_tree_fingerprint || data.traceTreeFingerprint || '',
                            canonicalProjectionMap,
                            nestedIndexProjectionMap
                        };
                    }
                    return {
                        chain: {},
                        trace: null,
                        executionTrace: [],
                        traceV2: null,
                        traceV3: null,
                        traceTreeFingerprint: '',
                        canonicalProjectionMap: null,
                        nestedIndexProjectionMap: null
                    };
                } catch (e) {
                    return {
                        chain: {},
                        trace: null,
                        executionTrace: [],
                        traceV2: null,
                        traceV3: null,
                        traceTreeFingerprint: '',
                        canonicalProjectionMap: null,
                        nestedIndexProjectionMap: null
                    };
                }
            },

            closeTraceSubtreePreview() {
                this._traceSubtreeResolveToken = Number(this._traceSubtreeResolveToken || 0) + 1;
                this.state.subtreePreviewState = {
                    open: false,
                    title: '',
                    data: null,
                    context: null,
                    loading: false,
                    resolvedFeature: null,
                    resolvedPath: [],
                    matchScore: 0,
                    error: ''
                };
                this.renderTraceSubtreePreview();
            },

            async ensureNestedRawDataForSubtree() {
                const activeId = String(this.state.activeId || '').trim();
                if (!activeId) return null;
                if (this.state.nestedRawData && String(this.state.nestedRawDataConversationId || '') === activeId) {
                    return this.state.nestedRawData;
                }
                try {
                    const query = `?conversation_id=${encodeURIComponent(activeId)}`;
                    const res = await fetch(`/api/tree-nested-data${query}`);
                    const data = await res.json();
                    if (data && data.success) {
                        this.state.nestedRawData = data.data || null;
                        this.state.nestedRawDataConversationId = activeId;
                        return this.state.nestedRawData;
                    }
                } catch (e) {
                    // ignore and fall through
                }
                return this.state.nestedRawData || null;
            },

            getNestedIndexProjectionMap(payload = null) {
                const source = payload && typeof payload === 'object'
                    ? payload
                    : (this.state.lastTracePayload || {});
                const direct = source.nestedIndexProjectionMap || source.nested_index_projection_map || null;
                const projection = source.canonicalProjectionMap || source.canonical_projection_map || null;
                const fromProjection = projection && typeof projection === 'object'
                    ? (projection.nestedIndexProjectionMap || projection.nested_index_projection_map || null)
                    : null;
                const map = direct || fromProjection || null;
                if (!map || typeof map !== 'object' || Array.isArray(map)) return null;
                return map;
            },

            normalizeNestedFeaturePath(pathValue) {
                if (!Array.isArray(pathValue)) return [];
                return pathValue
                    .map((part) => String(part === null || part === undefined ? '' : part).trim())
                    .filter(Boolean);
            },

            buildNestedStackFromFeaturePath(featurePath = []) {
                const rootFeature = typeof this._nestedBuildRootFeature === 'function'
                    ? this._nestedBuildRootFeature()
                    : null;
                if (!rootFeature || typeof rootFeature !== 'object' || Array.isArray(rootFeature)) {
                    return null;
                }
                const normalizedPath = this.normalizeNestedFeaturePath(featurePath);
                const stack = [{ key: 'ROOT', data: rootFeature }];
                let featureObj = rootFeature;
                for (const key of normalizedPath) {
                    if (!featureObj || typeof featureObj !== 'object' || Array.isArray(featureObj)) {
                        return null;
                    }
                    if (!(key in featureObj)) {
                        return null;
                    }
                    const next = featureObj[key];
                    if (!next || typeof next !== 'object' || Array.isArray(next)) {
                        return null;
                    }
                    stack.push({ key, data: next });
                    featureObj = next;
                }
                return { stack, feature: featureObj };
            },

            collectSubtreeSemanticIds(context = {}) {
                const ids = [];
                const push = (value) => {
                    const text = String(value || '').trim();
                    if (!text || ids.includes(text)) return;
                    ids.push(text);
                };

                const ctx = context && typeof context === 'object' ? context : {};
                const operation = ctx.operation
                    || (typeof this.getSelectedTraceOperation === 'function'
                        ? this.getSelectedTraceOperation(this.state.lastTraceV3 || null)
                        : null)
                    || null;
                if (!operation || typeof operation !== 'object') return ids;
                const operationKind = String(getValue(operation, 'kind', '') || '').trim();

                const playback = getValue(operation, 'playback', {}) || {};
                getArray(playback, 'canonicalNodeIds').forEach((value) => push(value));
                push(getValue(playback, 'canonicalAnswerNodeId', ''));

                // child/father 语义定位只使用 playback 顺序，不混入其他 semantic targets。
                if (operationKind === 'child_lookup' || operationKind === 'father_lookup') {
                    return ids;
                }

                const semanticTargets = getValue(operation, 'semanticTargets', {}) || {};
                const semanticExact = getValue(semanticTargets, 'semantic', {}) || {};
                ['tableNodeId', 'rowItemNodeId', 'rowAnchorNodeId', 'columnAnchorNodeId', 'resultNodeId']
                    .forEach((key) => push(getValue(semanticExact, key, '')));

                ['flatRow', 'flatColumn'].forEach((scope) => {
                    const scopedTargets = getValue(semanticTargets, scope, {}) || {};
                    ['tableNodeId', 'rowItemNodeId', 'rowAnchorNodeId', 'columnAnchorNodeId', 'resultNodeId']
                        .forEach((key) => push(getValue(scopedTargets, key, '')));
                });
                return ids;
            },

            resolveSubtreeBySemanticProjection(context = {}) {
                const projectionMap = this.getNestedIndexProjectionMap();
                if (!projectionMap) return null;
                const tracePayload = this.state.lastTracePayload || {};
                const canonicalProjectionMap = tracePayload.canonicalProjectionMap
                    || tracePayload.canonical_projection_map
                    || {};
                const rowCanonicalToSemantic = canonicalProjectionMap.rowCanonicalToSemantic
                    || canonicalProjectionMap.row_canonical_to_semantic
                    || {};
                const columnCanonicalToSemantic = canonicalProjectionMap.columnCanonicalToSemantic
                    || canonicalProjectionMap.column_canonical_to_semantic
                    || {};

                const operation = context && typeof context === 'object'
                    ? (
                        context.operation
                        || (typeof this.getSelectedTraceOperation === 'function'
                            ? this.getSelectedTraceOperation(this.state.lastTraceV3 || null)
                            : null)
                    )
                    : null;
                const operationKind = String(getValue(operation, 'kind', '') || '').trim();
                const operationArgs = getArray(operation, 'args').map((item) => String(item || '').trim()).filter(Boolean);

                const semanticIds = this.collectSubtreeSemanticIds(context);
                if (!semanticIds.length) return null;
                const projectedSemanticIds = [];
                const projectedSeen = new Set();
                const pushProjected = (value) => {
                    const text = String(value || '').trim();
                    if (!text || projectedSeen.has(text)) return;
                    projectedSeen.add(text);
                    projectedSemanticIds.push(text);
                };
                semanticIds.forEach((rawId) => {
                    const text = String(rawId || '').trim();
                    if (!text) return;
                    pushProjected(text);
                    if (text.startsWith('ct_tree_') || text.startsWith('ct_tree_group_')) {
                        pushProjected(rowCanonicalToSemantic[text] || '');
                        pushProjected(columnCanonicalToSemantic[text] || '');
                    }
                });

                const candidates = projectedSemanticIds.map((semanticId) => {
                    const entry = projectionMap[semanticId];
                    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return null;
                    const candidatePath = this.normalizeNestedFeaturePath(
                        entry.path || entry.indexPath || entry.nestedIndexPath || []
                    );
                    if (!candidatePath.length) return null;
                    return { semanticId, entry, path: candidatePath };
                }).filter(Boolean);
                if (!candidates.length) return null;

                const normalize = (value) => String(value || '').toLowerCase().replace(/\s+/g, '').trim();
                let hitSemanticId = '';
                let hitEntry = null;
                if (operationKind === 'child_lookup' && operationArgs.length) {
                    const wanted = normalize(operationArgs[0]);
                    const byChildArg = candidates.find((item) => {
                        const indexName = normalize(item && item.entry ? item.entry.indexName : '');
                        return !!wanted && !!indexName && (indexName.includes(wanted) || wanted.includes(indexName));
                    });
                    if (byChildArg) {
                        hitSemanticId = byChildArg.semanticId;
                        hitEntry = byChildArg.entry;
                    }
                }
                if (!hitEntry) {
                    // Follow semantic order strictly: use the first mapped semantic id.
                    hitSemanticId = candidates[0].semanticId;
                    hitEntry = candidates[0].entry;
                }
                if (!hitEntry) return null;

                const indexPath = this.normalizeNestedFeaturePath(
                    hitEntry.path || hitEntry.indexPath || hitEntry.nestedIndexPath || []
                );
                if (!indexPath.length) return null;
                const indexKey = String(indexPath[indexPath.length - 1] || '').trim();
                if (!indexKey) return null;

                const parentFeaturePath = indexPath.slice(0, -1);
                const parentResolved = this.buildNestedStackFromFeaturePath(parentFeaturePath);
                if (!parentResolved || !parentResolved.feature) return null;

                const drilldownCandidates = Array.isArray(hitEntry.drilldownBodyCandidates) ? hitEntry.drilldownBodyCandidates : [];
                let drillPath = this.normalizeNestedFeaturePath(
                    hitEntry.childFeaturePath || hitEntry.child_path || hitEntry.nextPath || []
                );
                if (!drillPath.length) {
                    const firstDrillCandidate = drilldownCandidates.find((item) => {
                        if (!item || typeof item !== 'object') return false;
                        return !!item.isDrilldown && Array.isArray(item.nextPath);
                    });
                    if (firstDrillCandidate) {
                        drillPath = this.normalizeNestedFeaturePath(firstDrillCandidate.nextPath || []);
                    }
                }
                const drillableByEntry = !!hitEntry.drillable || drilldownCandidates.some((item) => !!(item && item.isDrilldown));
                if (!drillPath.length && drillableByEntry) {
                    drillPath = indexPath.slice();
                }

                const childResolved = drillPath.length ? this.buildNestedStackFromFeaturePath(drillPath) : null;
                const drillable = !!(drillableByEntry && childResolved && childResolved.feature);
                const previewFeature = drillable
                    ? childResolved.feature
                    : parentResolved.feature;
                const previewPath = drillable ? drillPath : parentFeaturePath;

                return {
                    semanticId: hitSemanticId,
                    indexPath,
                    indexKey,
                    parentFeaturePath,
                    drillPath,
                    drillable,
                    previewFeature,
                    previewPath,
                };
            },

            async navigateNestedBySemanticProjection(resolution, options = {}) {
                if (!resolution || typeof resolution !== 'object') return false;
                const indexPath = this.normalizeNestedFeaturePath(resolution.indexPath || []);
                if (!indexPath.length) return false;

                const drillIfPossible = options.drillIfPossible !== false;
                let targetFeaturePath = this.normalizeNestedFeaturePath(
                    resolution.parentFeaturePath || indexPath.slice(0, -1)
                );
                if (drillIfPossible && resolution.drillable) {
                    const candidatePath = this.normalizeNestedFeaturePath(resolution.drillPath || indexPath);
                    if (candidatePath.length) {
                        targetFeaturePath = candidatePath;
                    }
                }

                // Keep flat tree on the right; nested feature view renders in left dock.
                if ((this.state.treeViewMode || 'nested') !== 'flat') {
                    await this.setTreeViewMode('flat');
                }
                if (typeof this.setNestedDockVisible === 'function') {
                    this.setNestedDockVisible(true);
                }
                await this.ensureNestedRawDataForSubtree();

                let resolved = this.buildNestedStackFromFeaturePath(targetFeaturePath);
                if ((!resolved || !resolved.stack) && drillIfPossible && resolution.drillable) {
                    // Fallback: simulate clicking the nested "sub feature tree" body on the located index column.
                    const parentPath = this.normalizeNestedFeaturePath(resolution.parentFeaturePath || indexPath.slice(0, -1));
                    const parentResolved = this.buildNestedStackFromFeaturePath(parentPath);
                    if (parentResolved && parentResolved.feature && typeof this._nestedParseFeatureLevel === 'function') {
                        const parsed = this._nestedParseFeatureLevel(parentResolved.feature);
                        const indexNodes = Array.isArray(parsed && parsed.indexNodes) ? parsed.indexNodes : [];
                        const bodyNodes = Array.isArray(parsed && parsed.bodyNodes) ? parsed.bodyNodes : [];
                        const targetIndexNode = indexNodes.find((item) => String((item && item.key) || '') === String(resolution.indexKey || ''));
                        const targetBody = (targetIndexNode && Array.isArray(targetIndexNode.bodyIndices))
                            ? targetIndexNode.bodyIndices
                                .map((idx) => bodyNodes[idx])
                                .find((node) => !!(node && node.isNested && node.nestedData))
                            : null;
                        if (targetBody && typeof this._nestedBreadcrumbLabelForTarget === 'function') {
                            const nextCrumbLabel = this._nestedBreadcrumbLabelForTarget(targetBody);
                            targetFeaturePath = [...parentPath, String(nextCrumbLabel || '').trim()].filter(Boolean);
                            resolved = this.buildNestedStackFromFeaturePath(targetFeaturePath);
                        }
                    }
                }
                if (!resolved || !resolved.stack) return false;

                this.state.nestedFocusPath = targetFeaturePath.slice();
                this.state.nestedPathStack = resolved.stack;
                this.state.nestedExpandState = { index: true, body: true };
                this.state.nestedSemanticFocus = {
                    semanticId: String(resolution.semanticId || '').trim(),
                    indexPath: indexPath.slice(),
                    indexKey: String(resolution.indexKey || '').trim(),
                };
                this.updateTreeViewModeButtons();
                if (typeof this.renderNestedFeatureView === 'function') {
                    this.renderNestedFeatureView({ preserveViewport: false });
                }
                return true;
            },

            collectSubtreeHintTokens(previewData, context = {}) {
                const hints = [];
                const push = (value) => {
                    const text = cleanTraceText(String(value || '')).trim();
                    if (!text || text.length < 2) return;
                    if (text === 'table' || text === 'none') return;
                    if (!hints.includes(text)) hints.push(text);
                };

                const operation = context && typeof context === 'object' ? (context.operation || null) : null;
                const frame = context && typeof context === 'object' ? (context.frame || null) : null;
                getArray(operation, 'args').forEach((item) => push(item));
                push(getValue(operation, 'title', ''));
                push(getValue(frame, 'title', ''));

                const previewObj = previewData && typeof previewData === 'object' ? previewData : null;
                if (previewObj && !Array.isArray(previewObj)) {
                    Object.keys(previewObj).forEach((key) => push(key));
                    Object.values(previewObj).forEach((value) => {
                        if (Array.isArray(value)) {
                            value.slice(0, 12).forEach((item) => push(item));
                        }
                    });
                }
                return hints.slice(0, 40);
            },

            findNestedFeatureMatch(previewData, context = {}) {
                const root = this.state.nestedRawData;
                if (!root || typeof root !== 'object' || Array.isArray(root)) return null;
                const normalize = (value) => String(value || '').toLowerCase().replace(/\s+/g, '').trim();
                const hintNorms = this.collectSubtreeHintTokens(previewData, context).map((item) => normalize(item)).filter(Boolean);
                const previewObj = previewData && typeof previewData === 'object' && !Array.isArray(previewData) ? previewData : {};
                const previewKeys = Object.keys(previewObj).map((item) => normalize(item)).filter(Boolean);
                const schemaHeaders = Array.isArray(previewObj.table)
                    ? previewObj.table.map((item) => normalize(item)).filter(Boolean)
                    : [];

                const scoreCandidate = (obj, pathParts = []) => {
                    if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return 0;
                    const keyNorms = Object.keys(obj).map((key) => normalize(key)).filter(Boolean);
                    const pathNorms = pathParts.map((item) => normalize(item)).filter(Boolean);
                    let score = 0;

                    hintNorms.forEach((hint) => {
                        if (!hint) return;
                        if (pathNorms.some((part) => part === hint)) score += 18;
                        if (keyNorms.some((part) => part === hint)) score += 18;
                        if (pathNorms.some((part) => part.includes(hint) || hint.includes(part))) score += 6;
                        if (keyNorms.some((part) => part.includes(hint) || hint.includes(part))) score += 6;
                    });

                    previewKeys.forEach((key) => {
                        if (keyNorms.includes(key)) score += 8;
                    });

                    if (schemaHeaders.length) {
                        const matchedHeaderCount = schemaHeaders.filter((item) => keyNorms.includes(item)).length;
                        if (matchedHeaderCount > 0) {
                            score += matchedHeaderCount * 9;
                            if (matchedHeaderCount >= 2) score += 18;
                        }
                    }

                    if (pathParts.length) score += 1;
                    return score;
                };

                let best = null;
                const visit = (obj, pathParts = []) => {
                    if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return;
                    const score = scoreCandidate(obj, pathParts);
                    if (!best || score > best.score) {
                        best = { data: obj, path: pathParts.slice(), score };
                    }
                    Object.entries(obj).forEach(([key, value]) => {
                        if (value && typeof value === 'object' && !Array.isArray(value)) {
                            visit(value, pathParts.concat([String(key)]));
                        }
                    });
                };
                visit(root, []);
                if (!best || best.score <= 0) return null;
                return best;
            },

            renderTraceSubtreeFeatureView(featureObj, resolvedPath = [], matchScore = 0) {
                if (!featureObj || typeof featureObj !== 'object' || Array.isArray(featureObj)) {
                    return `<div class="trace-subtree-root">${this.renderTraceSubtreeNode(featureObj, 'ROOT', 0)}</div>`;
                }
                const parser = typeof this._nestedParseFeatureLevel === 'function'
                    ? this._nestedParseFeatureLevel(featureObj)
                    : { indexNodes: [], bodyNodes: [] };
                const indexNodes = Array.isArray(parser && parser.indexNodes) ? parser.indexNodes : [];
                const bodyNodes = Array.isArray(parser && parser.bodyNodes) ? parser.bodyNodes : [];
                if (!indexNodes.length) {
                    return `<div class="trace-subtree-root">${this.renderTraceSubtreeNode(featureObj, 'ROOT', 0)}</div>`;
                }
                const pathText = ['ROOT', ...resolvedPath.map((part) => String(part || '').trim()).filter(Boolean)].join(' > ');
                const columnsHtml = indexNodes.map((indexNode) => {
                    const bodyIndices = Array.isArray(indexNode && indexNode.bodyIndices) ? indexNode.bodyIndices : [];
                    const chips = bodyIndices.map((bodyIdx) => {
                        const body = bodyNodes[bodyIdx] || {};
                        const bodyName = String(body && body.name !== undefined ? body.name : '').trim() || '-';
                        const suffix = String(body && body.meta ? body.meta : '').trim();
                        const text = `${bodyName}${suffix ? ` ${suffix}` : ''}${body && body.isNested ? ' ↘' : ''}`;
                        return `<div class="trace-subtree-feature-chip ${body && body.isNested ? 'is-nested' : ''}">${this.escapeHtml(text)}</div>`;
                    }).join('') || '<div class="trace-subtree-feature-chip is-empty">-</div>';
                    return `
                        <div class="trace-subtree-feature-col">
                            <div class="trace-subtree-feature-col-head">${this.escapeHtml(String(indexNode && indexNode.key !== undefined ? indexNode.key : '-'))}</div>
                            <div class="trace-subtree-feature-col-body">${chips}</div>
                        </div>
                    `;
                }).join('');
                const scoreText = typeof matchScore === 'string'
                    ? matchScore
                    : String(matchScore || 0);
                return `
                    <div class="trace-subtree-feature-wrap">
                        <div class="trace-subtree-feature-meta">
                            <span class="trace-subtree-feature-path">${this.escapeHtml(pathText)}</span>
                            <span class="trace-subtree-feature-score">match ${this.escapeHtml(scoreText)}</span>
                        </div>
                        <div class="trace-subtree-feature-grid" style="grid-template-columns:repeat(${Math.max(1, indexNodes.length)}, minmax(0,1fr));">
                            ${columnsHtml}
                        </div>
                    </div>
                `;
            },

            openTraceSubtreePreview(previewData, title = 'Subtree Preview', context = null) {
                if (!previewData || typeof previewData !== 'object') {
                    this.closeTraceSubtreePreview();
                    return;
                }
                const contextPayload = context && typeof context === 'object' ? context : {};
                const reqOpId = String(
                    getValue(contextPayload && contextPayload.operation, 'operationId', '')
                    || getValue(contextPayload && contextPayload.operation, 'id', '')
                    || ''
                );
                const reqFrameId = String(
                    getValue(contextPayload && contextPayload.frame, 'frameId', '')
                    || getValue(contextPayload && contextPayload.frame, 'id', '')
                    || ''
                );
                const reqSubquery = String(
                    getValue(contextPayload && contextPayload.subquery, 'index', '')
                    || ''
                );
                const requestKey = `${String(title || 'Subtree Preview')}|${reqSubquery}|${reqFrameId}|${reqOpId}`;
                const existing = this.state.subtreePreviewState || null;
                if (existing && existing.requestKey === requestKey && existing.loading) {
                    // A same-source preview is already resolving; skip duplicate open to avoid flicker.
                    return;
                }
                if (!this.state.traceStageDrawerOpen) {
                    this.toggleTraceStageDrawer(true);
                }
                const resolveToken = Number(this._traceSubtreeResolveToken || 0) + 1;
                this._traceSubtreeResolveToken = resolveToken;
                this.state.subtreePreviewState = {
                    open: true,
                    title: String(title || 'Subtree Preview'),
                    data: previewData,
                    context: contextPayload,
                    requestKey,
                    loading: true,
                    resolvedFeature: null,
                    resolvedPath: [],
                    matchScore: 0,
                    error: ''
                };
                this.renderTraceSubtreePreview();

                const finalize = (patch = {}) => {
                    if (Number(this._traceSubtreeResolveToken || 0) !== resolveToken) return;
                    this.state.subtreePreviewState = {
                        ...(this.state.subtreePreviewState || {}),
                        ...patch
                    };
                    this.renderTraceSubtreePreview();
                };

                (async () => {
                    try {
                        await this.ensureNestedRawDataForSubtree();
                        const semanticResolved = this.resolveSubtreeBySemanticProjection(contextPayload);
                        if (semanticResolved) {
                            if (contextPayload.navigateNested) {
                                await this.navigateNestedBySemanticProjection(semanticResolved, { drillIfPossible: true });
                            }
                            finalize({
                                loading: false,
                                resolvedFeature: semanticResolved.previewFeature || null,
                                resolvedPath: Array.isArray(semanticResolved.previewPath) ? semanticResolved.previewPath : [],
                                matchScore: 'semantic',
                                error: ''
                            });
                            return;
                        }

                        const matched = this.findNestedFeatureMatch(previewData, contextPayload);
                        finalize({
                            loading: false,
                            resolvedFeature: matched && matched.data ? matched.data : null,
                            resolvedPath: matched && Array.isArray(matched.path) ? matched.path : [],
                            matchScore: Number((matched && matched.score) || 0),
                            error: ''
                        });
                    } catch (e) {
                        finalize({
                            loading: false,
                            error: e && e.message ? String(e.message) : 'Failed to locate subtree'
                        });
                    }
                })();
            },

            renderTraceSubtreePreview() {
                const wrap = document.getElementById('trace-subtree-preview');
                const body = document.getElementById('trace-subtree-body');
                const title = document.getElementById('trace-subtree-title');
                if (!wrap || !body || !title) return;
                const preview = this.state.subtreePreviewState || {};
                const isOpen = !!preview.open && preview.data && this.state.traceStageDrawerOpen;
                wrap.classList.toggle('hidden', !isOpen);
                if (!isOpen) {
                    body.innerHTML = DEFAULT_SUBTREE_EMPTY;
                    title.textContent = 'Subtree Preview';
                    return;
                }
                title.textContent = String(preview.title || 'Subtree Preview');
                if (preview.loading) {
                    body.innerHTML = '<div class="trace-empty-state">Locating nested feature tree...</div>';
                    return;
                }
                if (preview.resolvedFeature && typeof preview.resolvedFeature === 'object' && !Array.isArray(preview.resolvedFeature)) {
                    body.innerHTML = this.renderTraceSubtreeFeatureView(
                        preview.resolvedFeature,
                        Array.isArray(preview.resolvedPath) ? preview.resolvedPath : [],
                        preview.matchScore
                    );
                    return;
                }
                if (preview.error) {
                    body.innerHTML = `<div class="trace-empty-state">${this.escapeHtml(String(preview.error || 'Failed to locate subtree'))}</div>`;
                    return;
                }
                body.innerHTML = `<div class="trace-subtree-root">${this.renderTraceSubtreeNode(preview.data, 'ROOT', 0)}</div>`;
            },

            renderTraceSubtreeNode(value, label = 'ROOT', depth = 0) {
                const safeLabel = this.escapeHtml(String(label || 'ROOT'));
                if (depth >= 5) {
                    return `<div class="trace-subtree-branch"><div class="trace-subtree-branch-label">${safeLabel}</div><div class="trace-subtree-leaf">...</div></div>`;
                }
                if (Array.isArray(value)) {
                    const children = value.slice(0, 8).map((item, index) => this.renderTraceSubtreeNode(item, `[${index}]`, depth + 1)).join('');
                    return `<div class="trace-subtree-branch"><div class="trace-subtree-branch-label">${safeLabel}</div><div class="trace-subtree-children">${children || '<div class="trace-subtree-leaf">[]</div>'}</div></div>`;
                }
                if (value && typeof value === 'object') {
                    const entries = Object.entries(value).slice(0, 10);
                    const children = entries.map(([key, item]) => this.renderTraceSubtreeNode(item, key, depth + 1)).join('');
                    return `<div class="trace-subtree-branch"><div class="trace-subtree-branch-label">${safeLabel}</div><div class="trace-subtree-children">${children || '<div class="trace-subtree-leaf">{}</div>'}</div></div>`;
                }
                return `<div class="trace-subtree-branch"><div class="trace-subtree-branch-label">${safeLabel}</div><div class="trace-subtree-leaf">${this.escapeHtml(String(value === null || value === undefined ? '' : value))}</div></div>`;
            },

            getTraceV3Subqueries(traceV3 = null) {
                const source = traceV3 || this.state.lastTraceV3;
                return getArray(source, 'subqueries');
            },

            ensureTraceStageSelection(traceV3 = null) {
                const subqueries = this.getTraceV3Subqueries(traceV3);
                if (!subqueries.length) {
                    this.state.selectedTraceSubquery = null;
                    this.state.selectedTraceFrame = null;
                    this.state.selectedTraceOperation = null;
                    return;
                }
                let selectedSubquery = subqueries.find((item) => Number(getValue(item, 'index', 0)) === Number(this.state.selectedTraceSubquery));
                if (!selectedSubquery) {
                    selectedSubquery = subqueries[0];
                    this.state.selectedTraceSubquery = Number(getValue(selectedSubquery, 'index', 0) || 0);
                }
                const frames = getArray(selectedSubquery, 'frames');
                let selectedFrame = frames.find((item) => String(getValue(item, 'frameId', '')) === String(this.state.selectedTraceFrame || ''));
                if (!selectedFrame) {
                    selectedFrame = frames[0] || null;
                    this.state.selectedTraceFrame = getValue(selectedFrame, 'frameId', null);
                }
                const operations = getArray(selectedFrame, 'operations');
                let selectedOperation = operations.find((item) => String(getValue(item, 'operationId', '')) === String(this.state.selectedTraceOperation || ''));
                if (!selectedOperation) {
                    selectedOperation = operations[0] || null;
                    this.state.selectedTraceOperation = getValue(selectedOperation, 'operationId', null);
                }
            },

            getSelectedTraceSubquery(traceV3 = null) {
                this.ensureTraceStageSelection(traceV3);
                return this.getTraceV3Subqueries(traceV3).find((item) => Number(getValue(item, 'index', 0)) === Number(this.state.selectedTraceSubquery)) || null;
            },

            getSelectedTraceFrame(traceV3 = null) {
                const subquery = this.getSelectedTraceSubquery(traceV3);
                const frames = getArray(subquery, 'frames');
                return frames.find((item) => String(getValue(item, 'frameId', '')) === String(this.state.selectedTraceFrame || '')) || null;
            },

            getSelectedTraceOperation(traceV3 = null) {
                const frame = this.getSelectedTraceFrame(traceV3);
                const operations = getArray(frame, 'operations');
                return operations.find((item) => String(getValue(item, 'operationId', '')) === String(this.state.selectedTraceOperation || '')) || null;
            },

            stopOperationPlayback() {
                if (this.state.operationPlaybackTimer) {
                    clearTimeout(this.state.operationPlaybackTimer);
                    this.state.operationPlaybackTimer = null;
                }
                this.state.currentPlaybackQueue = [];
                this.state.currentPlaybackIndex = -1;
                const win = this.getTreeIframeWindow();
                if (win && typeof win.stopTracePlayback === 'function') {
                    try {
                        win.stopTracePlayback();
                    } catch (e) {
                        // ignore
                    }
                }
                this.state.isTracePlaying = false;
                this.updateTracePlaybackButton();
            },

            async waitForTreeIframeReady(timeoutMs = 2600) {
                const startedAt = Date.now();
                return await new Promise((resolve) => {
                    const check = () => {
                        const win = this.getTreeIframeWindow();
                        if (win && typeof win.applyQaTrace === 'function') {
                            resolve(win);
                            return;
                        }
                        if ((Date.now() - startedAt) >= timeoutMs) {
                            resolve(win || null);
                            return;
                        }
                        setTimeout(check, 80);
                    };
                    check();
                });
            },

            applyTracePayloadToIframe(payload) {
                const win = this.getTreeIframeWindow();
                if (!win || typeof win.applyQaTrace !== 'function' || !payload) return null;
                try {
                    const projectionMap = payload.canonicalProjectionMap || payload.canonical_projection_map || null;
                    if (projectionMap && typeof win.setTraceCanonicalProjectionMap === 'function') {
                        try {
                            win.setTraceCanonicalProjectionMap(projectionMap);
                        } catch (e) {
                            // ignore
                        }
                    }
                    const traceInfo = win.applyQaTrace(payload);
                    this.state.lastQaTrace = traceInfo || null;
                    this.updateStrictTraceWarning(String((traceInfo && traceInfo.traceMode) || '').includes('strict'));
                    this.updateTraceDebugOverlay(traceInfo, payload.trace || null);
                    return traceInfo || null;
                } catch (e) {
                    return null;
                }
            },

            async ensureTracePlaybackSurface(options = {}) {
                const opts = options || {};
                if (this.state.view !== 'spanning-tree') {
                    await this.setView('spanning-tree');
                }
                if ((this.state.treeViewMode || 'nested') !== 'flat') {
                    await this.setTreeViewMode('flat');
                }
                const win = await this.waitForTreeIframeReady();
                const projectionMap = (this.state.lastTracePayload && (
                    this.state.lastTracePayload.canonicalProjectionMap
                    || this.state.lastTracePayload.canonical_projection_map
                )) || null;
                if (win && projectionMap && typeof win.setTraceCanonicalProjectionMap === 'function') {
                    try {
                        win.setTraceCanonicalProjectionMap(projectionMap);
                    } catch (e) {
                        // ignore
                    }
                }
                if (win && opts.resetTrace !== false && typeof win.clearQaTrace === 'function') {
                    try {
                        win.clearQaTrace();
                    } catch (e) {
                        // ignore
                    }
                }
                if (opts.reapplyTrace === true && this.state.lastTracePayload) {
                    this.applyTracePayloadToIframe(this.state.lastTracePayload);
                }
                return this.getTreeIframeWindow();
            },

            buildTraceSegmentFromOperation(operation) {
                const playback = operation && operation.playback ? operation.playback : {};
                const flatMode = String(this.state.flatSubMode || 'column').trim().toLowerCase();
                const viewMode = flatMode === 'row' ? 'row' : 'column';
                const semanticNodeIds = Array.isArray(playback.semanticNodeIds) ? playback.semanticNodeIds.map((x) => String(x || '').trim()).filter(Boolean) : [];
                const semanticEdgeIds = Array.isArray(playback.semanticEdgeIds) ? playback.semanticEdgeIds.map((x) => String(x || '').trim()).filter(Boolean) : [];
                const canonicalNodeIds = Array.isArray(playback.canonicalNodeIds) ? playback.canonicalNodeIds.map((x) => String(x || '').trim()).filter(Boolean) : [];
                const canonicalEdgeIds = Array.isArray(playback.canonicalEdgeIds) ? playback.canonicalEdgeIds.map((x) => String(x || '').trim()).filter(Boolean) : [];
                const frontendNodeIds = Array.isArray(playback.nodeIds) ? playback.nodeIds.map((x) => String(x || '').trim()).filter(Boolean) : [];
                const frontendEdgeIds = Array.isArray(playback.edgeIds) ? playback.edgeIds.map((x) => String(x || '').trim()).filter(Boolean) : [];
                const nodeIds = semanticNodeIds.length ? semanticNodeIds : (canonicalNodeIds.length ? canonicalNodeIds : frontendNodeIds);
                const edgeIds = semanticEdgeIds.length ? semanticEdgeIds : (canonicalEdgeIds.length ? canonicalEdgeIds : frontendEdgeIds);
                const answerNodeId = String(
                    playback.semanticAnswerNodeId
                    || playback.canonicalAnswerNodeId
                    || playback.answerNodeId
                    || ''
                ).trim() || null;
                if (!nodeIds.length) {
                    try {
                        console.warn('[trace-debug] empty playback ids', {
                            operationId: getValue(operation, 'operationId', ''),
                            kind: getValue(operation, 'kind', ''),
                            playback
                        });
                    } catch (e) {
                        // ignore
                    }
                }
                return {
                    nodeIds,
                    edgeIds,
                    answerNodeId,
                    viewMode
                };
            },

            extractOperationResultLabels(operation) {
                const summary = operation ? getValue(operation, 'resultSummary', null) : null;
                if (Array.isArray(summary)) {
                    return summary.map((item) => cleanTraceText(String(item || ''))).filter(Boolean);
                }
                if (summary && typeof summary === 'object') {
                    if (Array.isArray(summary.results)) {
                        return summary.results.map((item) => cleanTraceText(String(item || ''))).filter(Boolean);
                    }
                    if (hasValue(summary.value)) {
                        return [cleanTraceText(String(summary.value || ''))].filter(Boolean);
                    }
                    if (hasValue(summary.preview)) {
                        return [cleanTraceText(String(summary.preview || ''))].filter(Boolean);
                    }
                }
                const text = cleanTraceText(String(summary || ''));
                if (text && text !== '-' && text !== 'None') return [text];
                const detailLines = getArray(operation, 'details');
                for (const line of detailLines) {
                    const rawLine = String(line || '').trim();
                    const match = rawLine.match(/^Retrieved Data:\s*(.+)$/i);
                    if (!match) continue;
                    const parsed = tryParseLiteral(match[1]);
                    if (Array.isArray(parsed)) {
                        const labels = parsed.map((item) => cleanTraceText(String(item || ''))).filter(Boolean);
                        if (labels.length) return labels;
                    }
                    const inline = cleanTraceText(match[1]);
                    if (inline) return [inline];
                }
                return [];
            },

            buildTraceSegmentsFromOperation(operation) {
                const fallbackSegment = this.buildTraceSegmentFromOperation(operation);
                if (!operation || String(getValue(operation, 'kind', '')) !== 'extract_lookup') {
                    return [fallbackSegment];
                }

                const semanticTargets = getValue(operation, 'semanticTargets', {}) || {};
                const flatMode = String(this.state.flatSubMode || 'column').trim().toLowerCase();
                const viewMode = flatMode === 'row' ? 'row' : 'column';
                const scopeKey = flatMode === 'row' ? 'flatRow' : 'flatColumn';
                const scopedTargets = getValue(semanticTargets, scopeKey, {}) || {};
                const semanticExactTargets = getValue(semanticTargets, 'semantic', {}) || {};
                // 严格按当前视图(row/column)优先选择 scoped 目标，避免跨视图锚点错位。
                const tableNodeId = String(
                    getValue(scopedTargets, 'tableNodeId', '') || getValue(semanticExactTargets, 'tableNodeId', '')
                ).trim();
                const rowItemNodeId = String(
                    getValue(scopedTargets, 'rowItemNodeId', '') || getValue(semanticExactTargets, 'rowItemNodeId', '')
                ).trim();
                const rowAnchorNodeId = String(
                    getValue(scopedTargets, 'rowAnchorNodeId', '') || getValue(semanticExactTargets, 'rowAnchorNodeId', '')
                ).trim();
                const columnAnchorNodeId = String(
                    getValue(scopedTargets, 'columnAnchorNodeId', '') || getValue(semanticExactTargets, 'columnAnchorNodeId', '')
                ).trim();
                const resultNodeId = String(
                    getValue(scopedTargets, 'resultNodeId', '') || getValue(semanticExactTargets, 'resultNodeId', '')
                ).trim();
                const exactAnchorNodeIds = [rowAnchorNodeId, columnAnchorNodeId, resultNodeId]
                    .map((x) => String(x || '').trim())
                    .filter((x, idx, arr) => !!x && arr.indexOf(x) === idx);
                const exactNodeIds = (exactAnchorNodeIds.length ? exactAnchorNodeIds : [tableNodeId, rowItemNodeId, rowAnchorNodeId, columnAnchorNodeId, resultNodeId])
                    .map((x) => String(x || '').trim())
                    .filter((x, idx, arr) => !!x && arr.indexOf(x) === idx);

                if (!exactNodeIds.length) {
                    return [fallbackSegment];
                }
                const exactSegment = {
                    nodeIds: exactNodeIds,
                    edgeIds: [],
                    deriveEdges: false,
                    answerNodeId: resultNodeId || null,
                    focusNodeId: resultNodeId || columnAnchorNodeId || rowAnchorNodeId || null,
                    viewMode
                };
                // EXT 回放优先展示语义精确命中（行锚点/列锚点/结果值），
                // 避免先播放大范围 fallback 路径导致“Visited nodes: 10”类噪声。
                return [exactSegment];
            },

            summarizeOperationSubtitle(operation) {
                if (!operation) return '-';
                if (String(getValue(operation, 'status', '')) === 'skipped') return 'Skipped';
                const summary = summarizeValue(getValue(operation, 'resultSummary', getValue(operation, 'raw', '-')));
                return cleanTraceText(summary || '-');
            },

            getTraceNarrativePreview(subquery) {
                const hiddenPrefixes = [
                    'Answering Subquery',
                    'Final Check for Query',
                    'Final Check Passed and Final Answer',
                    'Back Verification',
                    'Query List:',
                    'Reliabillity:',
                    'Answer succeeded',
                    'Cost time:'
                ];
                return dedupeTextLines(getArray(subquery, 'narrative')).filter((line) => {
                    return !hiddenPrefixes.some((prefix) => line.indexOf(prefix) === 0);
                }).slice(-8);
            },

            selectTraceSubquery(subqueryIndex, options = {}) {
                const subqueries = this.getTraceV3Subqueries();
                const subquery = subqueries.find((item) => Number(getValue(item, 'index', 0)) === Number(subqueryIndex));
                if (!subquery) return;
                if (options.openDrawer === true) {
                    this.toggleTraceStageDrawer(true);
                }
                this.state.selectedTraceSubquery = Number(subquery.index || 0);
                const subqueryFrames = getArray(subquery, 'frames');
                const firstFrame = subqueryFrames[0] || null;
                const firstFrameOperations = getArray(firstFrame, 'operations');
                this.state.selectedTraceFrame = getValue(firstFrame, 'frameId', null);
                this.state.selectedTraceOperation = getValue(firstFrameOperations[0], 'operationId', null);
                this.renderTraceStageResults(this.state.lastExecutionTrace || [], {
                    primitiveSteps: this.state.lastPrimitiveSteps || [],
                    traceV3: this.state.lastTraceV3
                });
                if (options.play) {
                    const queue = [];
                    subqueryFrames.forEach((frame) => {
                        getArray(frame, 'operations').forEach((operation) => {
                            queue.push({
                                subqueryIndex: subquery.index,
                                frameId: frame.frameId,
                                operationId: operation.operationId,
                                operation
                            });
                        });
                    });
                    this.playOperationQueue(queue);
                }
            },

            async selectTraceFrame(subqueryIndex, frameId, options = {}) {
                const subquery = this.getTraceV3Subqueries().find((item) => Number(getValue(item, 'index', 0)) === Number(subqueryIndex));
                const frame = getArray(subquery, 'frames').find((item) => String(getValue(item, 'frameId', '')) === String(frameId || ''));
                if (!subquery || !frame) return;
                if (options.openDrawer === true) {
                    this.toggleTraceStageDrawer(true);
                }
                this.state.selectedTraceSubquery = Number(subquery.index || 0);
                this.state.selectedTraceFrame = frame.frameId;
                this.state.selectedTraceOperation = getValue(getArray(frame, 'operations')[0], 'operationId', null);
                this.renderTraceStageResults(this.state.lastExecutionTrace || [], {
                    primitiveSteps: this.state.lastPrimitiveSteps || [],
                    traceV3: this.state.lastTraceV3
                });
                if (options.play) {
                    const queue = getArray(frame, 'operations').map((operation) => ({
                        subqueryIndex: subquery.index,
                        frameId: frame.frameId,
                        operationId: operation.operationId,
                        operation
                    }));
                    await this.playOperationQueue(queue);
                }
            },

            async selectTraceOperation(subqueryIndex, frameId, operationId, options = {}) {
                const subquery = this.getTraceV3Subqueries().find((item) => Number(getValue(item, 'index', 0)) === Number(subqueryIndex));
                const frame = getArray(subquery, 'frames').find((item) => String(getValue(item, 'frameId', '')) === String(frameId || ''));
                const operation = getArray(frame, 'operations').find((item) => String(getValue(item, 'operationId', '')) === String(operationId || ''));
                if (!subquery || !frame || !operation) return;
                if (options.openDrawer === true) {
                    this.toggleTraceStageDrawer(true);
                }
                this.state.selectedTraceSubquery = Number(subquery.index || 0);
                this.state.selectedTraceFrame = frame.frameId;
                this.state.selectedTraceOperation = operation.operationId;
                if (options.openSubtree && operation.hasSubtreePreview && operation.subtreePreviewData) {
                    this.openTraceSubtreePreview(
                        operation.subtreePreviewData,
                        operation.subtreePreviewTitle || operation.title || 'Subtree Preview',
                        {
                            operation,
                            frame,
                            subquery,
                            navigateNested: options.navigateNested === true
                        }
                    );
                }
                this.renderTraceStageResults(this.state.lastExecutionTrace || [], {
                    primitiveSteps: this.state.lastPrimitiveSteps || [],
                    traceV3: this.state.lastTraceV3
                });
                if (options.play) {
                    await this.playOperationQueue([{
                        subqueryIndex: subquery.index,
                        frameId: frame.frameId,
                        operationId: operation.operationId,
                        operation
                    }]);
                    return;
                }
                if (options.preview !== false) {
                    await this.ensureTracePlaybackSurface({ resetTrace: true, reapplyTrace: false });
                    this.playTraceOperation(operation, {
                        play: false,
                        openSubtree: options.openSubtree === true
                    });
                }
            },

            playTraceOperation(operation, options = {}) {
                if (!operation) return { started: false, duration: 700 };
                const segments = this.buildTraceSegmentsFromOperation(operation);
                const segment = segments[segments.length - 1] || this.buildTraceSegmentFromOperation(operation);
                const win = this.getTreeIframeWindow();
                let duration = 820;
                let started = false;
                try {
                    console.info('[trace-debug] playTraceOperation', {
                        operationId: getValue(operation, 'operationId', ''),
                        kind: getValue(operation, 'kind', ''),
                        title: getValue(operation, 'title', ''),
                        segments,
                        selectedSegment: segment,
                        flatMode: this.state.flatSubMode || 'column'
                    });
                } catch (e) {
                    // ignore
                }
                if (operation.hasSubtreePreview && operation.subtreePreviewData && options.openSubtree !== false) {
                    this.openTraceSubtreePreview(
                        operation.subtreePreviewData,
                        operation.subtreePreviewTitle || operation.title || 'Subtree Preview',
                        {
                            operation,
                            navigateNested: options.navigateNested === true
                        }
                    );
                }
                if ((this.state.treeViewMode || 'nested') === 'flat' && win) {
                    try {
                        if (segments.length > 1 && typeof win.flashTraceSequence === 'function') {
                            const sequenceRes = win.flashTraceSequence(segments, {
                                step_duration_ms: options.play === false ? 1200 : 980,
                                final_duration_ms: options.play === false ? 1700 : 1450,
                                gap_ms: 180
                            });
                            try {
                                console.info('[trace-debug] flashTraceSequence result', sequenceRes);
                            } catch (e) {
                                // ignore
                            }
                            started = !!(sequenceRes && sequenceRes.flashed);
                            duration = Math.max(duration, Number((sequenceRes && sequenceRes.duration_ms) || 0));
                        } else if (segment.nodeIds.length && typeof win.flashTraceSegment === 'function') {
                            const flashDuration = options.play === false ? 2400 : 1700;
                            const flashRes = win.flashTraceSegment(segment, {
                                duration_ms: flashDuration
                            });
                            try {
                                console.info('[trace-debug] flashTraceSegment result', flashRes);
                            } catch (e) {
                                // ignore
                            }
                            started = true;
                            duration = Math.max(duration, flashDuration);
                        }
                        if (segment.nodeIds.length && typeof win.previewTraceSegment === 'function' && typeof win.flashTraceSegment !== 'function') {
                            const previewRes = win.previewTraceSegment(segment);
                            try {
                                console.info('[trace-debug] previewTraceSegment result', previewRes);
                            } catch (e) {
                                // ignore
                            }
                            started = !!(previewRes && (previewRes.applied || previewRes.flashed || previewRes.started));
                        }
                    } catch (e) {
                        try {
                            console.warn('[trace-debug] playTraceOperation exception', e);
                        } catch (_e) {}
                        // ignore
                    }
                }
                if (!segment.nodeIds.length) {
                    duration = 700;
                } else if (!started) {
                    duration = Math.max(1400, segment.nodeIds.length * 260);
                }
                this.state.isTracePlaying = options.play !== false;
                this.updateTracePlaybackButton();
                return { started, duration };
            },

            async playOperationQueue(queue = []) {
                const steps = Array.isArray(queue) ? queue.filter(Boolean) : [];
                if (!steps.length) return false;
                await this.ensureTracePlaybackSurface({ resetTrace: true, reapplyTrace: false });
                this.stopOperationPlayback();
                this.state.currentPlaybackQueue = steps;
                this.state.currentPlaybackIndex = -1;
                this.state.isTracePlaying = true;
                this.updateTracePlaybackButton();

                const runStep = (index) => {
                    if (index >= steps.length) {
                        this.stopOperationPlayback();
                        return;
                    }
                    const step = steps[index];
                    this.state.currentPlaybackIndex = index;
                    this.state.selectedTraceSubquery = Number((step && step.subqueryIndex) || this.state.selectedTraceSubquery || 0);
                    this.state.selectedTraceFrame = (step && step.frameId) || this.state.selectedTraceFrame || null;
                    this.state.selectedTraceOperation = (step && step.operationId) || this.state.selectedTraceOperation || null;
                    this.renderTraceStageResults(this.state.lastExecutionTrace || [], {
                        primitiveSteps: this.state.lastPrimitiveSteps || [],
                        traceV3: this.state.lastTraceV3
                    });
                    const result = this.playTraceOperation((step && step.operation) || null, { play: true, openSubtree: false });
                    const delay = Math.max(650, Number((result && result.duration) || 820));
                    this.state.operationPlaybackTimer = setTimeout(() => runStep(index + 1), delay);
                };

                runStep(0);
                return true;
            },

            playSelectedTracePath() {
                const traceV3 = this.state.lastTraceV3;
                const subqueries = this.getTraceV3Subqueries(traceV3);
                if (!subqueries.length) return false;
                this.ensureTraceStageSelection(traceV3);
                const selectedSubquery = this.getSelectedTraceSubquery(traceV3);
                const selectedFrame = this.getSelectedTraceFrame(traceV3);
                const selectedFrameOperations = getArray(selectedFrame, 'operations');
                if (selectedFrameOperations.length) {
                    const queue = selectedFrameOperations.map((operation) => ({
                        subqueryIndex: getValue(selectedSubquery, 'index', 0),
                        frameId: selectedFrame.frameId,
                        operationId: operation.operationId,
                        operation
                    }));
                    return this.playOperationQueue(queue);
                }
                const selectedSubqueryFrames = getArray(selectedSubquery, 'frames');
                if (selectedSubqueryFrames.length) {
                    const queue = [];
                    selectedSubqueryFrames.forEach((frame) => {
                        getArray(frame, 'operations').forEach((operation) => {
                            queue.push({
                                subqueryIndex: selectedSubquery.index,
                                frameId: frame.frameId,
                                operationId: operation.operationId,
                                operation
                            });
                        });
                    });
                    return this.playOperationQueue(queue);
                }
                return false;
            },

            async applyPendingQaTrace() {
                if (!this.state.pendingQaTrace) return;
                const win = this.getTreeIframeWindow();
                if (!win || typeof win.applyQaTrace !== 'function') return;
                try {
                    this.state.lastTracePayload = this.state.pendingQaTrace;
                    const traceInfo = this.applyTracePayloadToIframe(this.state.pendingQaTrace);
                    this.state.lastExecutionTrace = Array.isArray(this.state.pendingQaTrace && this.state.pendingQaTrace.executionTrace) ? this.state.pendingQaTrace.executionTrace : [];
                    this.state.lastPrimitiveSteps = Array.isArray(this.state.pendingQaTrace && this.state.pendingQaTrace.traceV2 && this.state.pendingQaTrace.traceV2.primitiveSteps) ? this.state.pendingQaTrace.traceV2.primitiveSteps : [];
                    this.state.lastTraceV3 = (this.state.pendingQaTrace && this.state.pendingQaTrace.traceV3) || null;
                    this.renderTraceStageResults((this.state.pendingQaTrace && this.state.pendingQaTrace.executionTrace) || [], {
                        primitiveSteps: this.state.lastPrimitiveSteps,
                        traceV3: this.state.lastTraceV3
                    });
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
                const executionTrace = Array.isArray(tracePayload && tracePayload.executionTrace) ? tracePayload.executionTrace : [];
                const traceV2 = (tracePayload && tracePayload.traceV2) || null;
                const traceV3 = (tracePayload && tracePayload.traceV3) || null;
                const visitNodeOrder = executionTrace
                    .filter((ev) => String((ev && ev.event_type) || '') === 'visit')
                    .map((ev) => String((ev && (ev.canonical_id || ev.canonical_trace_id || ev.frontend_node_id)) || '').trim())
                    .filter(Boolean);
                const visitNodeSet = Array.from(new Set(visitNodeOrder));
                const visitEdgeOrder = [];
                for (let i = 1; i < visitNodeOrder.length; i += 1) {
                    visitEdgeOrder.push(`${visitNodeOrder[i - 1]}->${visitNodeOrder[i]}`);
                }
                const backendTrace = (tracePayload && tracePayload.trace) || null;
                const answerNorm = String(answerText || '').toLowerCase().replace(/\s+/g, '').trim();
                let answerNodeFromVisit = null;
                if (answerNorm) {
                    for (let i = executionTrace.length - 1; i >= 0; i -= 1) {
                        const ev = executionTrace[i];
                        const nodeText = String((ev && ev.node_value) || '').toLowerCase().replace(/\s+/g, '').trim();
                        if (!nodeText) continue;
                        if (answerNorm.includes(nodeText) || nodeText.includes(answerNorm)) {
                            answerNodeFromVisit = (ev && (ev.canonical_id || ev.canonical_trace_id || ev.frontend_node_id)) || null;
                            if (answerNodeFromVisit) break;
                        }
                    }
                }
                const mergedTrace = {
                    ...(backendTrace || {}),
                    mode: visitNodeOrder.length ? 'strict_visit' : getValue(backendTrace, 'mode', 'inferred'),
                    matched_node_ids: visitNodeSet.length ? visitNodeSet : getValue(backendTrace, 'matched_node_ids', []),
                    path_node_order: visitNodeOrder.length ? visitNodeOrder : getValue(backendTrace, 'path_node_order', []),
                    path_edge_order: visitEdgeOrder.length ? visitEdgeOrder : getValue(backendTrace, 'path_edge_order', []),
                    answer_node_id: answerNodeFromVisit || getValue(backendTrace, 'answer_node_id', null)
                };
                const payload = {
                    chain: (tracePayload && tracePayload.chain) || {},
                    trace: mergedTrace,
                    executionTrace,
                    traceV2,
                    traceV3,
                    traceTreeFingerprint: (tracePayload && tracePayload.traceTreeFingerprint) || '',
                    canonicalProjectionMap: (tracePayload && (
                        tracePayload.canonicalProjectionMap
                        || tracePayload.canonical_projection_map
                    )) || null,
                    nestedIndexProjectionMap: (tracePayload && tracePayload.nestedIndexProjectionMap) || null,
                    answerText
                };
                const win = this.getTreeIframeWindow();
                this.state.lastTracePayload = payload;
                this.state.lastExecutionTrace = executionTrace;
                this.state.lastPrimitiveSteps = Array.isArray(traceV2 && traceV2.primitiveSteps) ? traceV2.primitiveSteps : [];
                this.state.lastTraceV3 = traceV3 || null;
                this.renderTraceStageResults(executionTrace, {
                    primitiveSteps: this.state.lastPrimitiveSteps,
                    traceV3: this.state.lastTraceV3
                });

                if (win && typeof win.applyQaTrace === 'function') {
                    try {
                        this.applyTracePayloadToIframe(payload);
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

            renderLegacyTraceStageResults(executionTrace, options = {}) {
                return originalRenderTraceStageResults.call(this, executionTrace, options);
            },

            renderTraceStageResults(executionTrace, options = {}) {
                const list = document.getElementById('trace-stage-list');
                if (!list) return;
                this.stopTraceStageAppend();
                const traceV3 = (options && options.traceV3) || this.state.lastTraceV3 || null;
                if (!getArray(traceV3, 'subqueries').length) {
                    this.renderLegacyTraceStageResults(executionTrace, options);
                    this.renderTraceSubtreePreview();
                    return;
                }

                this.state.lastTraceV3 = traceV3;
                this.ensureTraceStageSelection(traceV3);
                const subqueries = this.getTraceV3Subqueries(traceV3);
                const selectedSubquery = this.getSelectedTraceSubquery(traceV3);
                const selectedFrame = this.getSelectedTraceFrame(traceV3);
                const selectedOperation = this.getSelectedTraceOperation(traceV3);

                const subqueryTabsHtml = subqueries.map((subquery) => {
                    const active = Number(getValue(subquery, 'index', 0)) === Number(this.state.selectedTraceSubquery);
                    const label = `SubQ${this.escapeHtml(String(subquery && subquery.index !== undefined && subquery.index !== null ? subquery.index : '?'))}`;
                    return `<button class="trace-subquery-tab ${active ? 'is-active' : ''}" data-trace-subquery="${this.escapeHtml(String(subquery && subquery.index !== undefined && subquery.index !== null ? subquery.index : ''))}">${label}</button>`;
                }).join('');

                const subquerySummaryHtml = selectedSubquery ? `
                    <div class="trace-subquery-summary">
                        <div><strong>Question</strong>: ${this.escapeHtml(String(getValue(selectedSubquery, 'query', '-')))}</div>
                        <div><strong>Answer</strong>: ${this.escapeHtml(String(selectedSubquery && selectedSubquery.answer !== undefined && selectedSubquery.answer !== null ? selectedSubquery.answer : '-'))}</div>
                        <div><strong>Type</strong>: ${this.escapeHtml(String(getValue(selectedSubquery, 'reasoningType', '-')))}</div>
                        <div><strong>Frame Count</strong>: ${this.escapeHtml(String(getArray(selectedSubquery, 'frames').length))}</div>
                        ${this.getTraceNarrativePreview(selectedSubquery).length ? `
                            <div class="trace-subquery-narrative">
                                <div class="trace-subquery-narrative-title">Narrative Excerpt</div>
                                ${this.getTraceNarrativePreview(selectedSubquery).map((line) => `<div class="trace-subquery-narrative-line">${this.escapeHtml(line)}</div>`).join('')}
                            </div>
                        ` : ''}
                    </div>
                ` : '<div class="trace-empty-state">No subquery data.</div>';

                const framesHtml = getArray(selectedSubquery, 'frames').map((frame) => {
                    const active = String(getValue(frame, 'frameId', '')) === String(this.state.selectedTraceFrame || '');
                    const frameSummary = summarizeValue(getValue(frame, 'outputSummary', getValue(frame, 'inputSummary', '')));
                    const frameButtons = `
                        <div class="trace-frame-actions">
                            <button type="button" class="trace-mini-btn" data-trace-frame-play="${this.escapeHtml(String(getValue(frame, 'frameId', '')))}" data-trace-subquery-play="${this.escapeHtml(String(getValue(selectedSubquery, 'index', '')))}">Play Frame</button>
                            ${frame.hasSubtreePreview && frame.subtreePreviewData ? `<button type="button" class="trace-mini-btn" data-trace-frame-subtree="${this.escapeHtml(String(getValue(frame, 'frameId', '')))}" data-trace-subquery-subtree="${this.escapeHtml(String(getValue(selectedSubquery, 'index', '')))}">Subtree</button>` : ''}
                        </div>
                    `;
                    const operationsHtml = getArray(frame, 'operations').map((operation) => {
                        const opActive = String(getValue(operation, 'operationId', '')) === String(this.state.selectedTraceOperation || '');
                        const detailLines = dedupeTextLines(getArray(operation, 'details')).map((item) => `<div class="trace-operation-detail">${this.escapeHtml(item)}</div>`).join('');
                        const statusText = cleanTraceText(String(getValue(operation, 'status', '')));
                        const opButtons = `
                            <div class="trace-operation-actions">
                                <button type="button" class="trace-mini-btn" data-trace-operation-play="${this.escapeHtml(String(getValue(operation, 'operationId', '')))}" data-trace-frame-op="${this.escapeHtml(String(getValue(frame, 'frameId', '')))}" data-trace-subquery-op="${this.escapeHtml(String(getValue(selectedSubquery, 'index', '')))}">Play Step</button>
                                ${operation.hasSubtreePreview && operation.subtreePreviewData ? `<button type="button" class="trace-mini-btn" data-trace-operation-subtree="${this.escapeHtml(String(getValue(operation, 'operationId', '')))}" data-trace-frame-op="${this.escapeHtml(String(getValue(frame, 'frameId', '')))}" data-trace-subquery-op="${this.escapeHtml(String(getValue(selectedSubquery, 'index', '')))}">Subtree</button>` : ''}
                            </div>
                        `;
                        return `
                            <div class="trace-operation-card ${opActive ? 'is-active' : ''}" data-trace-operation="${this.escapeHtml(String(getValue(operation, 'operationId', '')))}" data-trace-operation-frame="${this.escapeHtml(String(getValue(frame, 'frameId', '')))}" data-trace-subquery="${this.escapeHtml(String(getValue(selectedSubquery, 'index', '')))}">
                                <div class="flex items-start justify-between gap-3">
                                    <div class="min-w-0">
                                        <div class="trace-operation-title-row">
                                            <div class="trace-operation-title">${this.escapeHtml(String(getValue(operation, 'title', 'Operation')))}</div>
                                            ${statusText ? `<span class="trace-status-pill">${this.escapeHtml(statusText)}</span>` : ''}
                                        </div>
                                        <div class="trace-operation-sub">${this.escapeHtml(String(this.summarizeOperationSubtitle(operation)))}</div>
                                    </div>
                                    ${opButtons}
                                </div>
                                ${detailLines ? `<div class="trace-operation-detail-list">${detailLines}</div>` : ''}
                            </div>
                        `;
                    }).join('') || '<div class="trace-empty-state">No operations to display for this frame.</div>';
                    return `
                        <div class="trace-frame-card ${active ? 'is-active' : ''}">
                            <div class="trace-frame-header" role="button" tabindex="0" data-trace-frame="${this.escapeHtml(String(getValue(frame, 'frameId', '')))}" data-trace-subquery="${this.escapeHtml(String(getValue(selectedSubquery, 'index', '')))}">
                                <div class="min-w-0">
                                    <div class="trace-frame-title">${this.escapeHtml(String((frame && frame.title) || 'Frame'))} · depth ${this.escapeHtml(String(frame && frame.depth !== undefined && frame.depth !== null ? frame.depth : '-'))}</div>
                                    <div class="trace-frame-meta">${this.escapeHtml(String(frameSummary || '-'))}</div>
                                </div>
                                ${frameButtons}
                            </div>
                            <div class="trace-operations">${operationsHtml}</div>
                        </div>
                    `;
                }).join('') || '<div class="trace-empty-state">No frames under this subquery.</div>';

                list.innerHTML = `
                    <div class="trace-subquery-tabs">${subqueryTabsHtml}</div>
                    ${subquerySummaryHtml}
                    <div class="trace-frames">${framesHtml}</div>
                `;

                list.querySelectorAll('[data-trace-subquery]').forEach((btn) => {
                    btn.addEventListener('click', (ev) => {
                        ev.preventDefault();
                        const index = btn.getAttribute('data-trace-subquery');
                        this.selectTraceSubquery(index);
                    });
                });
                list.querySelectorAll('.trace-frame-header[data-trace-frame]').forEach((btn) => {
                    btn.addEventListener('click', (ev) => {
                        if (hitClosest(ev.target, '[data-trace-frame-play],[data-trace-frame-subtree],[data-trace-operation-play],[data-trace-operation-subtree]')) return;
                        const frameId = btn.getAttribute('data-trace-frame');
                        const subqueryIndex = btn.getAttribute('data-trace-subquery');
                        this.selectTraceFrame(subqueryIndex, frameId, { play: true });
                    });
                    btn.addEventListener('keydown', (ev) => {
                        if (ev.key !== 'Enter' && ev.key !== ' ') return;
                        ev.preventDefault();
                        const frameId = btn.getAttribute('data-trace-frame');
                        const subqueryIndex = btn.getAttribute('data-trace-subquery');
                        this.selectTraceFrame(subqueryIndex, frameId, { play: true });
                    });
                });
                list.querySelectorAll('[data-trace-frame-play]').forEach((btn) => {
                    btn.addEventListener('click', (ev) => {
                        ev.preventDefault();
                        ev.stopPropagation();
                        const frameId = btn.getAttribute('data-trace-frame-play');
                        const subqueryIndex = btn.getAttribute('data-trace-subquery-play');
                        this.selectTraceFrame(subqueryIndex, frameId, { play: true });
                    });
                });
                list.querySelectorAll('[data-trace-frame-subtree]').forEach((btn) => {
                    btn.addEventListener('click', (ev) => {
                        ev.preventDefault();
                        ev.stopPropagation();
                        const frameId = btn.getAttribute('data-trace-frame-subtree');
                        const subqueryIndex = btn.getAttribute('data-trace-subquery-subtree');
                        const subquery = this.getTraceV3Subqueries().find((item) => Number(getValue(item, 'index', 0)) === Number(subqueryIndex));
                        const frame = getArray(subquery, 'frames').find((item) => String(getValue(item, 'frameId', '')) === String(frameId || ''));
                        if (frame && frame.subtreePreviewData) {
                            const frameOperation = getArray(frame, 'operations').find((item) => !!(item && item.hasSubtreePreview && item.subtreePreviewData))
                                || getArray(frame, 'operations')[0]
                                || null;
                            this.openTraceSubtreePreview(
                                frame.subtreePreviewData,
                                frame.subtreePreviewTitle || frame.title || 'Subtree Preview',
                                {
                                    frame,
                                    subquery,
                                    operation: frameOperation,
                                    navigateNested: true
                                }
                            );
                        }
                    });
                });
                list.querySelectorAll('[data-trace-operation]').forEach((card) => {
                    card.addEventListener('click', (ev) => {
                        if (hitClosest(ev.target, '[data-trace-operation-play],[data-trace-operation-subtree]')) return;
                        const operationId = card.getAttribute('data-trace-operation');
                        const frameId = card.getAttribute('data-trace-operation-frame');
                        const subqueryIndex = card.getAttribute('data-trace-subquery');
                        this.selectTraceOperation(subqueryIndex, frameId, operationId, { play: true, openSubtree: true });
                    });
                });
                list.querySelectorAll('[data-trace-operation-play]').forEach((btn) => {
                    btn.addEventListener('click', (ev) => {
                        ev.preventDefault();
                        ev.stopPropagation();
                        const operationId = btn.getAttribute('data-trace-operation-play');
                        const frameId = btn.getAttribute('data-trace-frame-op');
                        const subqueryIndex = btn.getAttribute('data-trace-subquery-op');
                        this.selectTraceOperation(subqueryIndex, frameId, operationId, { play: true, openSubtree: true });
                    });
                });
                list.querySelectorAll('[data-trace-operation-subtree]').forEach((btn) => {
                    btn.addEventListener('click', (ev) => {
                        ev.preventDefault();
                        ev.stopPropagation();
                        const operationId = btn.getAttribute('data-trace-operation-subtree');
                        const frameId = btn.getAttribute('data-trace-frame-op');
                        const subqueryIndex = btn.getAttribute('data-trace-subquery-op');
                        this.selectTraceOperation(subqueryIndex, frameId, operationId, {
                            openSubtree: true,
                            preview: false,
                            navigateNested: true
                        });
                    });
                });

                if (this.state.subtreePreviewState && this.state.subtreePreviewState.open) {
                    const candidate = (selectedOperation && selectedOperation.subtreePreviewData) || (selectedFrame && selectedFrame.subtreePreviewData) || null;
                    if (candidate) {
                        const title = (selectedOperation && selectedOperation.subtreePreviewTitle) || (selectedFrame && selectedFrame.subtreePreviewTitle) || 'Subtree Preview';
                        this.openTraceSubtreePreview(candidate, title, {
                            operation: selectedOperation || null,
                            frame: selectedFrame || null,
                            subquery: selectedSubquery || null
                        });
                    } else {
                        this.closeTraceSubtreePreview();
                    }
                } else {
                    this.renderTraceSubtreePreview();
                }
            },

            toggleTracePlayback() {
                if (getArray(this.state.lastTraceV3, 'subqueries').length) {
                    if (this.state.isTracePlaying) {
                        this.stopOperationPlayback();
                        return;
                    }
                    // User explicitly clicked "路径回放": keep replay card visible.
                    this.toggleTraceStageDrawer(true);
                    if (this.playSelectedTracePath()) return;
                }
                originalToggleTracePlayback.call(this);
            },

            autoPlayTrace() {
                if (getArray(this.state.lastTraceV3, 'subqueries').length) {
                    return;
                }
                originalAutoPlayTrace.call(this);
            },

            clearQaTrace() {
                this.stopOperationPlayback();
                this.state.lastTracePayload = null;
                this.state.lastTraceV3 = null;
                this.state.selectedTraceSubquery = null;
                this.state.selectedTraceFrame = null;
                this.state.selectedTraceOperation = null;
                this.closeTraceSubtreePreview();
                if (typeof this.setNestedDockVisible === 'function') {
                    this.setNestedDockVisible(false);
                }
                originalClearQaTrace.call(this);
            }
        });
    };
})();

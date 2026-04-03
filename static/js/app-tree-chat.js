(function initAppTreeChatModule(global) {
    global.AppTreeChatMethods = {
        treeChatPlaceholderHtml() {
            return '<p class="text-xs text-white/40 text-center mt-6">No messages yet. Ask a question to start.</p>';
        },

        persistCurrentTreeChat(immediate = false) {
            const convId = this.state.activeId;
            if (!convId) return;
            const list = document.getElementById('tree-chat-list');
            if (!list) return;
            const html = String(list.innerHTML || '').trim();
            this.state.treeChatByConversation[convId] = html;
            const saveFn = async () => {
                try {
                    await fetch(`/api/history/${encodeURIComponent(convId)}/tree-chat`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ chat_html: html })
                    });
                } catch (e) {
                    // ignore
                }
            };
            if (immediate) {
                if (this.state.treeChatPersistTimer) {
                    clearTimeout(this.state.treeChatPersistTimer);
                    this.state.treeChatPersistTimer = null;
                }
                saveFn();
                return;
            }
            if (this.state.treeChatPersistTimer) {
                clearTimeout(this.state.treeChatPersistTimer);
            }
            this.state.treeChatPersistTimer = setTimeout(() => {
                this.state.treeChatPersistTimer = null;
                saveFn();
            }, 260);
        },

        async restoreTreeChatForConversation(conversationId) {
            const list = document.getElementById('tree-chat-list');
            if (!list) return;
            if (!conversationId) {
                list.innerHTML = this.treeChatPlaceholderHtml();
                return;
            }
            const token = (this.state.treeChatRestoreToken || 0) + 1;
            this.state.treeChatRestoreToken = token;
            const cached = String(this.state.treeChatByConversation[conversationId] || '').trim();
            list.innerHTML = cached || this.treeChatPlaceholderHtml();
            list.scrollTop = list.scrollHeight;
            try {
                const res = await fetch(`/api/history/${encodeURIComponent(conversationId)}/tree-chat`);
                const data = await res.json();
                if (token !== this.state.treeChatRestoreToken) return;
                if (!data?.success) return;
                const serverHtml = String(data?.chat_html || '').trim();
                this.state.treeChatByConversation[conversationId] = serverHtml;
                if (this.state.activeId !== conversationId) return;
                list.innerHTML = serverHtml || this.treeChatPlaceholderHtml();
                list.scrollTop = list.scrollHeight;
            } catch (e) {
                // ignore
            }
        },

        extractPlainLog(logHtml = "") {
            if (!logHtml) return "";
            const withBreaks = String(logHtml)
                .replace(/<br\s*\/?>/gi, "\n")
                .replace(/<\/pre>/gi, "\n");
            const tmp = document.createElement('div');
            tmp.innerHTML = withBreaks;
            return (tmp.textContent || tmp.innerText || "").trim();
        },

        async fetchPlainLogs() {
            try {
                const res = await fetch('/api/logs');
                const data = await res.json();
                if (!data.success || !data.logs) return "";
                const plain = this.extractPlainLog(data.logs);
                return plain || "";
            } catch (e) {
                return "";
            }
        },

        extractDeltaLogs(beforeLogs = "", afterLogs = "") {
            const beforeLines = String(beforeLogs || "").split('\n').map(s => s.trim()).filter(Boolean);
            const afterLines = String(afterLogs || "").split('\n').map(s => s.trim()).filter(Boolean);
            if (!afterLines.length) return "";
            if (!beforeLines.length) return afterLines.slice(-80).join('\n');
            const beforeLast = beforeLines[beforeLines.length - 1];
            let anchor = -1;
            for (let i = afterLines.length - 1; i >= 0; i -= 1) {
                if (afterLines[i] === beforeLast) {
                    anchor = i;
                    break;
                }
            }
            const delta = anchor >= 0 ? afterLines.slice(anchor + 1) : afterLines.slice(-80);
            return delta.slice(-80).join('\n');
        },

        createTreeAssistantStreamingMessage() {
            const list = document.getElementById('tree-chat-list');
            if (!list) return null;
            if (list.children.length === 1 && list.firstElementChild?.tagName === 'P') {
                list.innerHTML = '';
            }
            const msg = document.createElement('div');
            msg.className = 'tree-chat-msg assistant';
            msg.innerHTML = `
                <details class="tree-thinking-inline" open>
                    <summary>Thinking (Live Logs)</summary>
                    <div data-thinking-log class="tree-thinking-log-stream">
                        <div class="tree-thinking-log-line">Waiting for logs...</div>
                    </div>
                </details>
                <div data-assistant-answer class="opacity-80">Generating answer...</div>
            `;
            list.appendChild(msg);
            list.scrollTop = list.scrollHeight;
            this.persistCurrentTreeChat();
            return {
                root: msg,
                logEl: msg.querySelector('[data-thinking-log]'),
                answerEl: msg.querySelector('[data-assistant-answer]'),
                detailsEl: msg.querySelector('details'),
                summaryEl: msg.querySelector('summary'),
                logLines: []
            };
        },

        updateTreeAssistantStreamingLogs(streamingMessage, logsText) {
            if (!streamingMessage?.logEl) return;
            const incomingLines = String(logsText || "")
                .split('\n')
                .map(line => line.trim())
                .filter(Boolean);

            if (!incomingLines.length) {
                if (!streamingMessage.logLines?.length) {
                    streamingMessage.logEl.innerHTML = '<div class="tree-thinking-log-line">Waiting for logs...</div>';
                }
                return;
            }

            if (!Array.isArray(streamingMessage.logLines)) {
                streamingMessage.logLines = [];
            }

            const prevLines = streamingMessage.logLines;
            let needReset = incomingLines.length < prevLines.length;
            if (!needReset) {
                for (let i = 0; i < prevLines.length; i += 1) {
                    if (incomingLines[i] !== prevLines[i]) {
                        needReset = true;
                        break;
                    }
                }
            }

            if (needReset) {
                streamingMessage.logEl.innerHTML = '';
                incomingLines.forEach((line) => {
                    const row = document.createElement('div');
                    row.className = 'tree-thinking-log-line';
                    row.textContent = line;
                    streamingMessage.logEl.appendChild(row);
                });
                streamingMessage.logLines = incomingLines;
                streamingMessage.logEl.scrollTop = streamingMessage.logEl.scrollHeight;
                return;
            }

            for (let i = prevLines.length; i < incomingLines.length; i += 1) {
                const row = document.createElement('div');
                row.className = 'tree-thinking-log-line';
                row.textContent = incomingLines[i];
                streamingMessage.logEl.appendChild(row);
            }
            streamingMessage.logLines = incomingLines;
            streamingMessage.logEl.scrollTop = streamingMessage.logEl.scrollHeight;
        },

        finalizeTreeAssistantStreamingMessage(streamingMessage, {
            answer = '',
            answerNodeId = null,
            answerNodeName = '',
            logsText = ''
        } = {}) {
            if (!streamingMessage?.root) return;
            const safeAnswer = this.formatContent(answer || 'Done.');
            const safeName = this.escapeHtml(answerNodeName || 'Answer Node');
            const hasLocate = !!answerNodeId;
            const locateButtonHtml = hasLocate
                ? `<div class="mt-2"><button onclick="app.focusAnswerNode('${answerNodeId}')" class="text-[11px] px-2.5 py-1 rounded-md bg-white/12 hover:bg-white/18 text-white/90 transition-colors border border-white/15">Locate Node: ${safeName}</button></div>`
                : '';
            this.updateTreeAssistantStreamingLogs(streamingMessage, logsText);
            if (streamingMessage.summaryEl) {
                streamingMessage.summaryEl.textContent = 'Thinking Process (This Turn)';
            }
            if (streamingMessage.detailsEl) {
                streamingMessage.detailsEl.open = false;
            }
            if (streamingMessage.answerEl) {
                streamingMessage.answerEl.className = '';
                streamingMessage.answerEl.innerHTML = `${safeAnswer}${locateButtonHtml}`;
            }
            const list = document.getElementById('tree-chat-list');
            if (list) list.scrollTop = list.scrollHeight;
            this.persistCurrentTreeChat();
        },

        addTreeChatMessage(role, content, answerNodeId = null, answerNodeName = "", thinkingLogs = "") {
            const list = document.getElementById('tree-chat-list');
            if (!list) return;
            if (list.children.length === 1 && list.firstElementChild?.tagName === 'P') {
                list.innerHTML = '';
            }
            const msg = document.createElement('div');
            msg.className = `tree-chat-msg ${role === 'user' ? 'user' : 'assistant'}`;
            const safeContent = this.formatContent(content);
            let locateButtonHtml = "";
            let thinkingHtml = "";
            if (role === "assistant" && thinkingLogs) {
                const safeLogs = this.escapeHtml(thinkingLogs);
                thinkingHtml = `
                    <details class="tree-thinking-inline">
                        <summary>Thinking Process (This Turn)</summary>
                        <pre>${safeLogs}</pre>
                    </details>
                `;
            }
            if (role === "assistant" && answerNodeId) {
                const safeName = this.escapeHtml(answerNodeName || "Answer Node");
                locateButtonHtml = `
                    <div class="mt-2">
                        <button onclick="app.focusAnswerNode('${answerNodeId}')" class="text-[11px] px-2.5 py-1 rounded-md bg-white/12 hover:bg-white/18 text-white/90 transition-colors border border-white/15">
                            Locate Node: ${safeName}
                        </button>
                    </div>
                `;
            }
            msg.innerHTML = `${thinkingHtml}${safeContent}${locateButtonHtml}`;
            list.appendChild(msg);
            list.scrollTop = list.scrollHeight;
            this.persistCurrentTreeChat();
        },

        async sendTreeMessage() {
            const input = document.getElementById('tree-chat-input');
            if (!input || this.state.isTreeSending) return;
            const message = input.value.trim();
            if (!message) return;

            this.clearQaTrace();
            const logsBefore = await this.fetchPlainLogs();
            input.value = '';
            this.addTreeChatMessage('user', message);
            this.state.isTreeSending = true;
            const streamingMessage = this.createTreeAssistantStreamingMessage();
            let latestLogs = '';

            const formData = new FormData();
            formData.append('message', message);
            formData.append('conversation_id', this.state.activeId || '');

            try {
                const response = await fetch('/api/chat-stream', { method: 'POST', body: formData });
                if (!response.ok || !response.body) {
                    throw new Error(`HTTP ${response.status}`);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let donePayload = null;

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';

                    for (const rawLine of lines) {
                        const line = rawLine.trim();
                        if (!line) continue;
                        let payload = null;
                        try {
                            payload = JSON.parse(line);
                        } catch (e) {
                            continue;
                        }
                        if (payload.type === 'log') {
                            const plain = this.extractPlainLog(payload.logs || "");
                            latestLogs = this.extractDeltaLogs(logsBefore, plain);
                            this.updateTreeAssistantStreamingLogs(streamingMessage, latestLogs);
                        } else if (payload.type === 'log_lines') {
                            const incoming = Array.isArray(payload.lines) ? payload.lines : [];
                            if (payload.reset) {
                                latestLogs = incoming.join('\n');
                                this.updateTreeAssistantStreamingLogs(streamingMessage, latestLogs);
                            } else if (incoming.length) {
                                latestLogs = latestLogs ? `${latestLogs}\n${incoming.join('\n')}` : incoming.join('\n');
                                const clipped = latestLogs.split('\n').filter(Boolean).slice(-160).join('\n');
                                latestLogs = clipped;
                                this.updateTreeAssistantStreamingLogs(streamingMessage, latestLogs);
                            }
                        } else if (payload.type === 'done') {
                            donePayload = payload;
                        }
                    }
                }

                if (!donePayload) {
                    throw new Error('Stream closed without final payload');
                }

                if (donePayload.success) {
                    if (!this.state.activeId && donePayload.conversation_id) {
                        this.state.activeId = donePayload.conversation_id;
                        await this.loadHistory();
                    }
                    this.updateActiveConversationTitle(this.state.activeId);
                    const finalMsg = donePayload.message || 'Done.';
                    const traceInfo = await this.syncQaTrace(finalMsg);
                    this.finalizeTreeAssistantStreamingMessage(streamingMessage, {
                        answer: finalMsg,
                        answerNodeId: traceInfo?.answerNodeId || this.state.lastQaTrace?.answerNodeId || null,
                        answerNodeName: traceInfo?.answerNodeName || this.state.lastQaTrace?.answerNodeName || "",
                        logsText: latestLogs
                    });
                } else {
                    this.finalizeTreeAssistantStreamingMessage(streamingMessage, {
                        answer: `Error: ${donePayload.error || 'unknown error'}`,
                        logsText: latestLogs
                    });
                }
            } catch (err) {
                this.finalizeTreeAssistantStreamingMessage(streamingMessage, {
                    answer: 'Error connecting to server.',
                    logsText: latestLogs
                });
            } finally {
                this.state.isTreeSending = false;
            }
        }
    };
})(window);

(function initAppMainChatModule(global) {
    // Main chat area rendering helpers (left panel / center welcome flow).
    global.AppMainChatMethods = {
        pushModelOutput(message) {
            const feed = document.getElementById('model-output-feed');
            if (!feed || !message) return;
            const safe = this.escapeHtml(String(message));
            feed.innerHTML = `
                <div class="model-output-item">
                    <div class="model-output-avatar">
                        <img src="/static/images/image.png" alt="ST-Raptor" onerror="this.src='https://via.placeholder.com/56?text=AI'">
                    </div>
                    <div class="model-output-bubble">${safe}</div>
                </div>
            `;
            feed.scrollIntoView({ block: 'nearest' });
        },

        startNewChat() {
            this.persistCurrentTreeChat(true);
            this.state.activeId = null;
            document.getElementById('view-header').classList.add('hidden');
            this.setView('chat');
            this.updateActiveConversationTitle(null);
            this.pushModelOutput('Started a new workspace. Upload a file or open a history tree model.');
        },

        sendSuggestion(text) {
            this.pushModelOutput(text);
        },

        addMessage(role, content, answerNodeId = null, answerNodeName = '') {
            const prefix = role === 'assistant' ? '[MODEL]' : '[USER]';
            this.pushModelOutput(`${prefix} ${String(content || '')}`);
            if (answerNodeId) {
                this.pushModelOutput(`Answer node: ${answerNodeName || answerNodeId}`);
            }
        },

        formatContent(text) {
            return text
                .replace(/\n/g, '<br>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>');
        },

        updateActiveConversationTitle(conversationId) {
            const titleEl = document.getElementById('active-conversation-title');
            if (!titleEl) return;
            if (!conversationId) {
                titleEl.textContent = 'New Conversation';
                return;
            }
            const record = this.state.historyMap[conversationId];
            titleEl.textContent = record?.summary || 'New Conversation';
        },

        showChatAfterUpload() {
            this.setView('chat');
            document.getElementById('view-header')?.classList.toggle('hidden', !this.state.activeId);
            this.pushModelOutput('File uploaded successfully. Preparing the tree model...');
        },

        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    };
})(window);

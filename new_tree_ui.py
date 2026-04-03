import json
import os
import base64
import html


def load_initial_tree_data(path="cache/temp.ui.tree.json"):
    """从本地文件加载初始树数据，失败或缺失则返回空列表。"""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] 读取初始树数据失败: {e}")
    return []


# 前端树模板（暗色主题版本）
NEW_TREE_HTML_TEMPLATE = r"""
<style>
  * { box-sizing: border-box; }
  html, body { height: 100%; overflow: hidden; }
  body { margin: 0; padding: 0; background: #0a0a0f; }
  .nt-container { 
    background: #3A3939; 
    border: none; 
    border-radius: 12px; 
    padding: 10px; 
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; 
    position: relative; 
    height: 100%;
    min-height: 100%; 
    max-width: 100%; 
    margin: 0;
    box-shadow: 
      0 15px 50px rgba(0, 0, 0, 0.5),
      0 5px 20px rgba(0, 0, 0, 0.4),
      0 0 80px rgba(0, 0, 0, 0.3);
  }
  .nt-tree-box { 
    border: none; 
    border-radius: 8px; 
    padding: 12px; 
    background: #3A3939; 
    height: 100%;
    min-height: 100%;
    overflow: auto; 
    position: relative;
    box-shadow: none;
    cursor: grab;
    scrollbar-width: none;
    -ms-overflow-style: none;
  }
  .nt-tree-box::-webkit-scrollbar { display: none; }
  .nt-tree-box.dragging {
    cursor: grabbing;
    user-select: none;
  }
  .nt-tree { position: relative; min-height: 320px; }
  .nt-abs-node { position: absolute; transform: translate(-50%, -50%); transform-origin: center center; }
  .nt-image-item {
    position: absolute;
    transform: translate(-50%, -50%);
    border: 1px solid rgba(148, 163, 184, 0.65);
    border-radius: 10px;
    overflow: hidden;
    background: rgba(15, 23, 42, 0.82);
    box-shadow: 0 8px 26px rgba(0, 0, 0, 0.36);
    cursor: move;
    z-index: 6;
  }
  .nt-image-item img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    pointer-events: none;
    user-select: none;
  }
  .nt-image-cap {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    padding: 2px 6px;
    font-size: 10px;
    color: rgba(241, 245, 249, 0.95);
    background: linear-gradient(180deg, rgba(2, 6, 23, 0) 0%, rgba(2, 6, 23, 0.84) 100%);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    pointer-events: none;
  }
  .nt-image-resize-handle {
    position: absolute;
    right: 0;
    bottom: 0;
    width: 16px;
    height: 16px;
    border-top: 1px solid rgba(191, 219, 254, 0.7);
    border-left: 1px solid rgba(191, 219, 254, 0.7);
    border-top-left-radius: 6px;
    background: rgba(15, 23, 42, 0.86);
    cursor: se-resize;
    z-index: 1;
  }
  .nt-image-resize-handle::after {
    content: "";
    position: absolute;
    right: 3px;
    bottom: 3px;
    width: 7px;
    height: 7px;
    border-right: 2px solid rgba(226, 232, 240, 0.9);
    border-bottom: 2px solid rgba(226, 232, 240, 0.9);
  }
  .nt-image-delete {
    position: absolute;
    top: 6px;
    right: 6px;
    width: 20px;
    height: 20px;
    border-radius: 999px;
    border: 1px solid rgba(248, 113, 113, 0.8);
    background: rgba(127, 29, 29, 0.85);
    color: #fff;
    font-size: 14px;
    line-height: 1;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    z-index: 2;
  }
  .nt-image-delete:hover {
    background: rgba(185, 28, 28, 0.95);
    border-color: rgba(252, 165, 165, 0.95);
  }
  .nt-image-preview-mask {
    position: fixed;
    inset: 0;
    background: rgba(2, 6, 23, 0.78);
    backdrop-filter: blur(3px);
    z-index: 1200;
    display: none;
    align-items: center;
    justify-content: center;
    padding: 30px;
  }
  .nt-image-preview-mask.show {
    display: flex;
  }
  .nt-image-preview-panel {
    position: relative;
    max-width: min(92vw, 1400px);
    max-height: 90vh;
    border: 1px solid rgba(148, 163, 184, 0.45);
    border-radius: 14px;
    background: rgba(15, 23, 42, 0.96);
    box-shadow: 0 18px 54px rgba(0, 0, 0, 0.56);
    overflow: hidden;
  }
  .nt-image-preview-close {
    position: absolute;
    top: 8px;
    right: 8px;
    width: 34px;
    height: 34px;
    border: 1px solid rgba(148, 163, 184, 0.5);
    border-radius: 999px;
    background: rgba(2, 6, 23, 0.75);
    color: #e2e8f0;
    font-size: 20px;
    line-height: 1;
    cursor: pointer;
    z-index: 2;
  }
  .nt-image-preview-close:hover {
    background: rgba(15, 23, 42, 0.92);
    border-color: rgba(191, 219, 254, 0.72);
  }
  .nt-image-preview-img {
    display: block;
    max-width: min(92vw, 1400px);
    max-height: calc(90vh - 34px);
    width: auto;
    height: auto;
    object-fit: contain;
    background: rgba(2, 6, 23, 0.2);
  }
  .nt-image-preview-cap {
    padding: 8px 12px;
    font-size: 12px;
    color: rgba(226, 232, 240, 0.96);
    background: rgba(2, 6, 23, 0.7);
    border-top: 1px solid rgba(148, 163, 184, 0.28);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .nt-lines { position: absolute; top: 0; left: 0; pointer-events: none; }
  .nt-line {
    stroke-width: 2.5 !important;
    stroke-dasharray: 8 6;
    filter: drop-shadow(0 0 2px rgba(255, 255, 255, 0.25));
  }
  .nt-hover-tip {
    position: absolute;
    min-width: 180px;
    max-width: 320px;
    padding: 10px 12px;
    border: 1px solid rgba(106, 125, 227, 0.35);
    border-radius: 10px;
    background: rgba(20, 22, 36, 0.96);
    backdrop-filter: blur(12px);
    color: #ececf2;
    font-size: 12px;
    line-height: 1.45;
    box-shadow:
      0 8px 22px rgba(0, 0, 0, 0.45),
      0 0 26px rgba(106, 125, 227, 0.16);
    pointer-events: auto;
    display: none;
    z-index: 999;
  }
  .nt-tip-title {
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(186, 196, 255, 0.8);
    margin-bottom: 6px;
  }
  .nt-tip-body {
    word-break: break-word;
  }
  .nt-tip-meta {
    margin-top: 6px;
    font-size: 11px;
    color: rgba(210, 214, 235, 0.76);
  }
  .nt-tip-copy {
    margin-top: 8px;
    border: 1px solid rgba(132, 150, 248, 0.45);
    background: rgba(95, 113, 211, 0.2);
    color: #f2f4ff;
    border-radius: 8px;
    font-size: 11px;
    padding: 5px 9px;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .nt-tip-copy:hover {
    background: rgba(112, 131, 232, 0.35);
    border-color: rgba(132, 150, 248, 0.7);
  }
  .nt-node.trace-hit {
    border-color: #f59e0b !important;
    box-shadow:
      0 0 0 2px rgba(245, 158, 11, 0.28),
      0 0 20px rgba(245, 158, 11, 0.42) !important;
  }
  .nt-node.trace-answer {
    border-color: #22c55e !important;
    box-shadow:
      0 0 0 2px rgba(34, 197, 94, 0.35),
      0 0 24px rgba(34, 197, 94, 0.5) !important;
    background: linear-gradient(135deg, #466f5a 0%, #4f8a67 100%) !important;
  }
  .nt-line.trace-hit {
    stroke: #f59e0b !important;
    stroke-width: 3px !important;
    opacity: 0.95;
  }
  .nt-line.trace-answer {
    stroke: #22c55e !important;
    stroke-width: 3.2px !important;
    opacity: 1;
  }
  .nt-node.trace-playing {
    border-color: #ef4444 !important;
    background: linear-gradient(135deg, #5a2b2b 0%, #7a3232 100%) !important;
    box-shadow:
      0 0 0 2px rgba(239, 68, 68, 0.55),
      0 0 28px rgba(239, 68, 68, 0.75) !important;
  }
  .nt-line.trace-playing {
    stroke: #ef4444 !important;
    stroke-width: 4px !important;
    opacity: 1;
  }
  .nt-node.trace-flash {
    border-color: #f59e0b !important;
    background: linear-gradient(135deg, #61411d 0%, #7a4a16 100%) !important;
    box-shadow:
      0 0 0 2px rgba(245, 158, 11, 0.55),
      0 0 30px rgba(245, 158, 11, 0.82),
      0 0 52px rgba(251, 191, 36, 0.38) !important;
    animation: ntTraceFlash 1.8s ease-in-out;
  }
  .nt-line.trace-flash {
    stroke: #f59e0b !important;
    stroke-width: 4px !important;
    opacity: 1;
    filter: drop-shadow(0 0 10px rgba(245, 158, 11, 0.7));
    animation: ntTraceFlashLine 1.8s ease-in-out;
  }
  .nt-node.trace-pulse {
    animation: ntPulse 1s ease;
  }
  @keyframes ntPulse {
    0% { transform: translateY(-2px) scale(1.05); box-shadow: 0 0 0 0 rgba(34,197,94,0.45); }
    50% { transform: translateY(-2px) scale(1.1); box-shadow: 0 0 0 8px rgba(34,197,94,0.08); }
    100% { transform: translateY(-2px) scale(1.05); box-shadow: 0 0 0 0 rgba(34,197,94,0); }
  }
  @keyframes ntTraceFlash {
    0% { opacity: 0.45; transform: translateY(-2px) scale(1.02); }
    18% { opacity: 1; transform: translateY(-2px) scale(1.1); }
    36% { opacity: 0.5; transform: translateY(-2px) scale(1.04); }
    54% { opacity: 1; transform: translateY(-2px) scale(1.08); }
    100% { opacity: 1; transform: translateY(-2px) scale(1.05); }
  }
  @keyframes ntTraceFlashLine {
    0% { opacity: 0.25; }
    18% { opacity: 1; }
    36% { opacity: 0.35; }
    54% { opacity: 1; }
    100% { opacity: 0.95; }
  }
  .nt-history {
    position: absolute;
    left: 10px;
    top: 58px;
    width: 250px;
    max-height: 45%;
    display: flex;
    flex-direction: column;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    background: rgba(12, 14, 24, 0.9);
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 26px rgba(0,0,0,0.35);
    z-index: 12;
  }
  .nt-history-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }
  .nt-history-title {
    font-size: 11px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: rgba(220, 228, 255, 0.9);
  }
  .nt-history-actions {
    display: flex;
    gap: 6px;
  }
  .nt-history-btn {
    border: 1px solid rgba(132,150,248,0.35);
    background: rgba(95,113,211,0.18);
    color: #f0f4ff;
    font-size: 11px;
    border-radius: 8px;
    padding: 4px 8px;
    cursor: pointer;
  }
  .nt-history-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .nt-history-list {
    list-style: none;
    margin: 0;
    padding: 6px;
    overflow-y: auto;
    max-height: 240px;
  }
  .nt-history-item {
    margin: 0;
    padding: 0;
  }
  .nt-history-item button {
    width: 100%;
    border: 0;
    background: transparent;
    color: rgba(225, 230, 245, 0.86);
    text-align: left;
    padding: 6px 8px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 11px;
    line-height: 1.35;
  }
  .nt-history-item button:hover {
    background: rgba(141, 163, 255, 0.14);
  }
  .nt-history-item.active button {
    background: rgba(110, 132, 232, 0.3);
    color: #fff;
  }
  .nt-history-time {
    display: block;
    opacity: 0.66;
    font-size: 10px;
    margin-top: 2px;
  }
  .nt-node { 
    display: inline-flex; 
    align-items: center; 
    gap: 8px; 
    padding: 8px 14px; 
    border-radius: 10px; 
    cursor: pointer; 
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); 
    background: #4F5F58; 
    color: #ffffff; 
    border: 1px solid rgba(79, 95, 88, 0.8);
    box-shadow: 0 2px 8px rgba(79, 95, 88, 0.4);
    position: relative;
  }
  .nt-node.m-node {
    background: linear-gradient(135deg, #475569 0%, #334155 100%);
    border-color: rgba(148, 163, 184, 0.82);
    box-shadow: 0 2px 10px rgba(51, 65, 85, 0.45);
  }
  .nt-node.b-node {
    background: linear-gradient(135deg, #0f766e 0%, #115e59 100%);
    border-color: rgba(45, 212, 191, 0.75);
    box-shadow: 0 2px 10px rgba(15, 118, 110, 0.45);
  }
  .nt-visit-badge {
    position: absolute;
    top: -8px;
    right: -8px;
    min-width: 20px;
    height: 20px;
    padding: 0 5px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.35);
    background: rgba(245, 158, 11, 0.95);
    color: #fff;
    font-size: 10px;
    font-weight: 700;
    line-height: 18px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(245, 158, 11, 0.45);
    pointer-events: none;
    z-index: 2;
  }
  .nt-node:hover { 
    background: #5A6B64; 
    border-color: rgba(79, 95, 88, 1);
    transform: translateY(-2px) scale(1.05);
    box-shadow: 
      0 4px 15px rgba(79, 95, 88, 0.6),
      0 0 20px rgba(79, 95, 88, 0.4);
  }
  .nt-node.selected { 
    background: linear-gradient(135deg, #4F5F58 0%, #5A6B64 100%); 
    color: #ffffff; 
    font-weight: 600; 
    border-color: #4F5F58;
    box-shadow: 
      0 4px 20px rgba(79, 95, 88, 0.7),
      0 0 30px rgba(79, 95, 88, 0.5),
      inset 0 0 20px rgba(255, 255, 255, 0.1);
  }
  .nt-node.multi-selected {
    outline: 2px solid rgba(96, 165, 250, 0.92);
    outline-offset: 1px;
  }
  .nt-node.search-hit {
    box-shadow:
      0 0 0 2px rgba(251, 191, 36, 0.58),
      0 4px 14px rgba(251, 191, 36, 0.35);
  }
  .nt-node.search-active {
    box-shadow:
      0 0 0 2px rgba(250, 204, 21, 0.9),
      0 0 26px rgba(250, 204, 21, 0.56);
  }
  .nt-tools {
    position: absolute;
    top: calc(58px + 45% + 10px);
    left: 14px;
    display: none;
    gap: 8px;
    z-index: 21;
    pointer-events: auto;
  }
  .nt-tool-btn {
    border: 1px solid rgba(148, 163, 184, 0.42);
    background: rgba(15, 23, 42, 0.82);
    color: rgba(226, 232, 240, 0.95);
    border-radius: 8px;
    font-size: 11px;
    padding: 5px 9px;
    cursor: pointer;
  }
  .nt-tool-btn:hover {
    border-color: rgba(191, 219, 254, 0.8);
    background: rgba(30, 41, 59, 0.92);
  }
  .nt-search-bar {
    position: absolute;
    top: 58px;
    right: 14px;
    z-index: 21;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 7px 8px;
    border-radius: 10px;
    border: 1px solid rgba(148, 163, 184, 0.42);
    background: rgba(2, 6, 23, 0.78);
    backdrop-filter: blur(8px);
    display: none;
  }
  .nt-search-input {
    width: 180px;
    border: 1px solid rgba(100, 116, 139, 0.5);
    border-radius: 6px;
    padding: 4px 8px;
    background: rgba(15, 23, 42, 0.84);
    color: #e2e8f0;
    font-size: 12px;
    outline: none;
  }
  .nt-search-input:focus {
    border-color: rgba(125, 211, 252, 0.82);
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25);
  }
  .nt-search-btn {
    border: 1px solid rgba(148, 163, 184, 0.42);
    border-radius: 6px;
    background: rgba(15, 23, 42, 0.9);
    color: #e2e8f0;
    font-size: 12px;
    line-height: 1;
    width: 26px;
    height: 24px;
    cursor: pointer;
  }
  .nt-search-btn:hover {
    border-color: rgba(191, 219, 254, 0.8);
  }
  .nt-search-meta {
    min-width: 54px;
    text-align: center;
    font-size: 11px;
    color: rgba(191, 219, 254, 0.92);
  }
  .nt-minimap {
    position: absolute;
    left: 14px;
    bottom: 14px;
    width: 190px;
    height: 128px;
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.4);
    background: rgba(2, 6, 23, 0.76);
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.4);
    z-index: 21;
    overflow: hidden;
    cursor: pointer;
  }
  .nt-minimap-svg {
    width: 100%;
    height: 100%;
    display: block;
  }
  .nt-minimap-viewport {
    position: absolute;
    border: 1px solid rgba(125, 211, 252, 0.92);
    box-shadow: inset 0 0 0 1px rgba(125, 211, 252, 0.3);
    background: rgba(56, 189, 248, 0.08);
    pointer-events: none;
  }
  .nt-empty { 
    color: rgba(176, 176, 176, 0.6); 
    text-align: center; 
    padding: 40px 12px; 
    font-size: 14px; 
    pointer-events: none; 
  }
  .nt-add-btn { 
    position: absolute; 
    top: 50%; 
    left: 50%; 
    transform: translate(-50%, -50%); 
    width: 50px; 
    height: 50px; 
    border-radius: 50%; 
    border: 2px solid rgba(106, 125, 227, 0.5); 
    background: linear-gradient(135deg, rgba(106, 125, 227, 0.3) 0%, rgba(143, 157, 240, 0.3) 100%); 
    color: #ffffff; 
    font-size: 24px; 
    font-weight: 300;
    cursor: pointer; 
    box-shadow: 
      0 4px 15px rgba(106, 125, 227, 0.4),
      0 0 30px rgba(106, 125, 227, 0.2);
    display: flex; 
    align-items: center; 
    justify-content: center; 
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); 
    z-index: 5; 
  }
  .nt-add-btn:hover { 
    transform: translate(-50%, -50%) translateY(-3px) scale(1.1); 
    box-shadow: 
      0 6px 25px rgba(106, 125, 227, 0.5),
      0 0 40px rgba(106, 125, 227, 0.3);
    border-color: rgba(106, 125, 227, 0.8);
  }
  .nt-node-add { 
    border: 1px solid rgba(106, 125, 227, 0.4); 
    background: rgba(106, 125, 227, 0.2); 
    color: #ffffff; 
    border-radius: 6px; 
    width: 28px; 
    height: 28px; 
    font-size: 18px; 
    font-weight: 300;
    display: inline-flex; 
    align-items: center; 
    justify-content: center; 
    cursor: pointer; 
    transition: all 0.2s ease; 
  }
  .nt-node-add:hover { 
    background: rgba(106, 125, 227, 0.4); 
    border-color: rgba(106, 125, 227, 0.8);
    transform: scale(1.1);
    box-shadow: 
      0 2px 10px rgba(106, 125, 227, 0.4),
      0 0 15px rgba(106, 125, 227, 0.2);
  }
  .nt-menu { 
    position: absolute; 
    background: rgba(26, 26, 46, 0.95);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(106, 125, 227, 0.3); 
    border-radius: 10px; 
    box-shadow: 
      0 8px 32px rgba(0, 0, 0, 0.4),
      0 0 40px rgba(106, 125, 227, 0.2);
    min-width: 140px; 
    padding: 8px 0; 
    z-index: 20; 
    display: none; 
  }
  .nt-menu button { 
    width: 100%; 
    padding: 10px 16px; 
    text-align: left; 
    background: none; 
    border: none; 
    font-size: 13px; 
    cursor: pointer; 
    color: #e0e0e0; 
    transition: all 0.2s ease;
  }
  .nt-menu button:hover { 
    background: rgba(106, 125, 227, 0.2); 
    color: #ffffff;
  }
  .nt-label[contenteditable="true"] { 
    outline: 2px solid rgba(106, 125, 227, 0.6); 
    outline-offset: 2px;
    border-radius: 6px; 
    background: rgba(106, 125, 227, 0.1);
  }
  /* 保存按钮样式已移到外部，这里保留样式以防其他地方使用 */
  .nt-btn-save { 
    border: 1px solid rgba(106, 125, 227, 0.4); 
    background: linear-gradient(135deg, rgba(106, 125, 227, 0.4) 0%, rgba(143, 157, 240, 0.4) 100%); 
    color: #ffffff; 
    border-radius: 8px; 
    padding: 10px 20px; 
    cursor: pointer; 
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    font-weight: 500;
    font-size: 14px;
    /* 统一阴影效果 */
    box-shadow: 
      0 2px 8px rgba(106, 125, 227, 0.3), 
      0 0 20px rgba(106, 125, 227, 0.1),
      inset 0 1px 0 rgba(255, 255, 255, 0.1);
  }
  .nt-btn-save:hover { 
    background: linear-gradient(135deg, rgba(106, 125, 227, 0.6) 0%, rgba(143, 157, 240, 0.6) 100%);
    border-color: rgba(106, 125, 227, 0.8);
    transform: translateY(-2px);
    box-shadow: 
      0 4px 15px rgba(106, 125, 227, 0.4),
      0 0 30px rgba(106, 125, 227, 0.2),
      inset 0 1px 0 rgba(255, 255, 255, 0.15);
  }
  .nt-btn-save:active {
    transform: translateY(0);
    box-shadow: 
      0 2px 6px rgba(106, 125, 227, 0.3),
      inset 0 2px 4px rgba(0, 0, 0, 0.2);
  }
  .nt-btn-save:disabled { 
    opacity: 0.5; 
    cursor: not-allowed; 
    transform: none;
  }
  .nt-save-msg { 
    font-size: 12px; 
    color: rgba(176, 176, 176, 0.8);
    font-weight: 500;
  }
  .nt-save-overlay {
    position: absolute;
    inset: 0;
    z-index: 1300;
    display: none;
    align-items: center;
    justify-content: center;
    background: rgba(2, 6, 23, 0.48);
    backdrop-filter: blur(4px);
    pointer-events: all;
  }
  .nt-save-overlay.show {
    display: flex;
  }
  .nt-save-overlay-panel {
    min-width: 220px;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    background: rgba(15, 23, 42, 0.88);
    color: #e2e8f0;
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.35);
  }
  .nt-save-spinner {
    width: 18px;
    height: 18px;
    border-radius: 999px;
    border: 2px solid rgba(148, 163, 184, 0.35);
    border-top-color: rgba(96, 165, 250, 0.95);
    animation: nt-spin 0.8s linear infinite;
    flex: 0 0 auto;
  }
  .nt-save-overlay-text {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.02em;
  }
  @keyframes nt-spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
  /* 统一隐藏滚动条 */
  .nt-tree-box,
  .nt-tree-box * {
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
  }
  .nt-tree-box::-webkit-scrollbar,
  .nt-tree-box *::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    display: none !important;
  }
</style>

<div id="nt-root" class="nt-container">
  <button id="nt-add" class="nt-add-btn" title="Add node">+</button>
  <div class="nt-history" id="nt-history">
    <div class="nt-history-head">
      <span class="nt-history-title">Action Timeline</span>
      <div class="nt-history-actions">
        <button id="nt-undo" class="nt-history-btn" title="Undo (Ctrl/Cmd+Z)">Undo</button>
        <button id="nt-redo" class="nt-history-btn" title="Redo (Ctrl/Cmd+Y)">Redo</button>
      </div>
    </div>
    <ul id="nt-history-list" class="nt-history-list"></ul>
  </div>
  <div class="nt-tree-box" id="nt-tree-box">
    <div class="nt-tools">
      <button id="nt-batch-delete" class="nt-tool-btn" type="button">Batch Delete</button>
      <button id="nt-batch-collapse" class="nt-tool-btn" type="button">Batch Collapse</button>
      <button id="nt-clear-selection" class="nt-tool-btn" type="button">Clear Selection</button>
    </div>
    <div class="nt-search-bar">
      <input id="nt-search-input" class="nt-search-input" type="text" placeholder="Search node name..." />
      <button id="nt-search-prev" class="nt-search-btn" type="button" title="Previous">↑</button>
      <button id="nt-search-next" class="nt-search-btn" type="button" title="Next">↓</button>
      <span id="nt-search-meta" class="nt-search-meta">0/0</span>
    </div>
    <div id="nt-tip" class="nt-hover-tip"></div>
    <div class="nt-tree" id="nt-tree"></div>
  </div>
  <div id="nt-minimap" class="nt-minimap" title="Click to locate">
    <svg id="nt-minimap-svg" class="nt-minimap-svg" viewBox="0 0 190 128" preserveAspectRatio="none"></svg>
    <div id="nt-minimap-viewport" class="nt-minimap-viewport"></div>
  </div>
  <div id="nt-image-preview-mask" class="nt-image-preview-mask">
    <div class="nt-image-preview-panel">
      <button id="nt-image-preview-close" class="nt-image-preview-close" type="button" title="Close">×</button>
      <img id="nt-image-preview-img" class="nt-image-preview-img" src="" alt="Preview" />
      <div id="nt-image-preview-cap" class="nt-image-preview-cap"></div>
    </div>
  </div>
  <div id="nt-save-overlay" class="nt-save-overlay">
    <div class="nt-save-overlay-panel">
      <div class="nt-save-spinner"></div>
      <div class="nt-save-overlay-text">Saving changes...</div>
    </div>
  </div>
  <!-- 保存按钮已移到外部tree-header，这里不再显示 -->
  <div id="nt-menu" class="nt-menu">
    <button data-action="add">ADD</button>
    <button data-action="rename">RENAME</button>
    <button data-action="delete">DELETE</button>
  </div>
</div>

<script>
(function() {
  const app = document.getElementById("nt-root");
  if (!app) return;

  const treeEl = app.querySelector("#nt-tree");
  const treeBox = app.querySelector("#nt-tree-box");
  const addBtn = app.querySelector("#nt-add");
  const menuEl = app.querySelector("#nt-menu");
  const tipEl = app.querySelector("#nt-tip");
  const historyListEl = app.querySelector("#nt-history-list");
  const undoBtn = app.querySelector("#nt-undo");
  const redoBtn = app.querySelector("#nt-redo");
  const saveBtn = app.querySelector("#nt-save");
  const saveMsg = app.querySelector("#nt-save-msg");
  const batchDeleteBtn = app.querySelector("#nt-batch-delete");
  const batchCollapseBtn = app.querySelector("#nt-batch-collapse");
  const clearSelectionBtn = app.querySelector("#nt-clear-selection");
  const searchInputEl = app.querySelector("#nt-search-input");
  const searchPrevBtn = app.querySelector("#nt-search-prev");
  const searchNextBtn = app.querySelector("#nt-search-next");
  const searchMetaEl = app.querySelector("#nt-search-meta");
  const minimapEl = app.querySelector("#nt-minimap");
  const minimapSvg = app.querySelector("#nt-minimap-svg");
  const minimapViewportEl = app.querySelector("#nt-minimap-viewport");
  const imagePreviewMask = app.querySelector("#nt-image-preview-mask");
  const imagePreviewImg = app.querySelector("#nt-image-preview-img");
  const imagePreviewCap = app.querySelector("#nt-image-preview-cap");
  const imagePreviewCloseBtn = app.querySelector("#nt-image-preview-close");
  const saveOverlay = app.querySelector("#nt-save-overlay");

  const rawData = __INIT_DATA__;
  const state = {
    nodes: [],
    selectedId: null,
    selectedIds: new Set(),
    lastSelectedId: null,
    menuTarget: null,
    collapsed: new Set(),
    zoom: 1,
    nodeOffsets: {},
    nodeDrag: null,
    imageItems: [],
    imageDrag: null,
    imageResize: null,
    editorConversationId: "",
    suppressClickUntil: 0,
    traceNodeIds: new Set(),
    traceEdgeKeys: new Set(),
    traceAnswerNodeId: null,
    tracePathNodeOrder: [],
    tracePathEdgeOrder: [],
    traceVisitCounts: new Map(),
    tracePlaybackTimer: null,
    tracePlaybackRunning: false,
    traceSequenceTimers: [],
    tracePlaybackSpeed: 1,
    traceCanonicalProjectionMap: null,
    searchQuery: "",
    searchResultIds: [],
    searchCursor: -1,
    minimapMeta: null,
    history: [],
    historyIndex: -1
  };
  const panState = { dragging: false, startX: 0, startY: 0, startScrollLeft: 0, startScrollTop: 0 };
  let tipHideTimer = null;

  const uid = () => "n-" + Math.random().toString(16).slice(2);

  const createNode = (name = "New Node") => ({ id: uid(), name, children: [] });
  const createImageId = () => "img-" + Math.random().toString(16).slice(2);

  function normalizeImageItem(item = {}) {
    const width = Math.max(80, Math.min(640, Number(item.width || 240) || 240));
    const height = Math.max(60, Math.min(480, Number(item.height || 150) || 150));
    return {
      id: String(item.id || createImageId()),
      name: String(item.name || "Image"),
      url: String(item.url || ""),
      x: Number(item.x || 220) || 220,
      y: Number(item.y || 160) || 160,
      width,
      height
    };
  }

  async function persistImageLayout() {
    const convId = String(state.editorConversationId || "").trim();
    if (!convId) return;
    try {
      await fetch(`/api/history/${encodeURIComponent(convId)}/tree-images`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images: state.imageItems })
      });
    } catch (e) {
      // ignore
    }
  }

  const UNIT_X = 150; // 控制水平间距
  const UNIT_Y = 110; // 控制行高

  const escapeHtml = (str = "") => str.replace(/[&<>"']/g, s => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[s] || s));

  const HISTORY_MAX = 80;

  function formatTime(ts) {
    const d = new Date(ts);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  }

  function setSaveOverlayVisible(show) {
    if (!saveOverlay) return;
    saveOverlay.classList.toggle("show", !!show);
  }

  function snapshotState(label = "Edit") {
    return {
      label,
      ts: Date.now(),
      nodes: JSON.parse(JSON.stringify(state.nodes)),
      imageItems: JSON.parse(JSON.stringify(state.imageItems || [])),
      nodeOffsets: JSON.parse(JSON.stringify(state.nodeOffsets || {})),
      collapsed: Array.from(state.collapsed || []),
      selectedId: state.selectedId || null,
      selectedIds: Array.from(state.selectedIds || []),
      lastSelectedId: state.lastSelectedId || null
    };
  }

  function applySnapshot(snapshot) {
    if (!snapshot) return;
    state.nodes = JSON.parse(JSON.stringify(snapshot.nodes || []));
    state.imageItems = JSON.parse(JSON.stringify(snapshot.imageItems || []));
    state.nodeOffsets = JSON.parse(JSON.stringify(snapshot.nodeOffsets || {}));
    state.collapsed = new Set(snapshot.collapsed || []);
    state.selectedId = snapshot.selectedId || null;
    state.selectedIds = new Set(snapshot.selectedIds || (state.selectedId ? [state.selectedId] : []));
    state.lastSelectedId = snapshot.lastSelectedId || state.selectedId || null;
    hideMenu();
    renderTree();
    updateHistoryView();
  }

  function updateHistoryView() {
    if (!historyListEl) return;
    historyListEl.innerHTML = state.history.map((item, index) => {
      const active = index === state.historyIndex ? "active" : "";
      return `
        <li class="nt-history-item ${active}">
          <button data-history-index="${index}">
            ${escapeHtml(item.label || "Edit")}
            <span class="nt-history-time">${formatTime(item.ts)}</span>
          </button>
        </li>
      `;
    }).join("");
    if (undoBtn) undoBtn.disabled = state.historyIndex <= 0;
    if (redoBtn) redoBtn.disabled = state.historyIndex >= state.history.length - 1;
  }

  function recordHistory(label = "Edit") {
    const snap = snapshotState(label);
    if (state.historyIndex < state.history.length - 1) {
      state.history = state.history.slice(0, state.historyIndex + 1);
    }
    state.history.push(snap);
    if (state.history.length > HISTORY_MAX) {
      const removeCount = state.history.length - HISTORY_MAX;
      state.history.splice(0, removeCount);
    }
    state.historyIndex = state.history.length - 1;
    updateHistoryView();
  }

  function jumpToHistory(index) {
    if (!Number.isInteger(index)) return;
    if (index < 0 || index >= state.history.length) return;
    state.historyIndex = index;
    applySnapshot(state.history[index]);
  }

  function undoHistory() {
    if (state.historyIndex <= 0) return;
    state.historyIndex -= 1;
    applySnapshot(state.history[state.historyIndex]);
  }

  function redoHistory() {
    if (state.historyIndex >= state.history.length - 1) return;
    state.historyIndex += 1;
    applySnapshot(state.history[state.historyIndex]);
  }

  function getNodeNameById(id) {
    const target = findNode(id);
    return target?.node?.name || "Node";
  }

  function buildNodeId(pathParts) {
    const raw = Array.isArray(pathParts) && pathParts.length ? pathParts.join("|") : "root";
    const safe = String(raw)
      .replace(/[^a-zA-Z0-9_-]+/g, "_")
      .replace(/_+/g, "_")
      .replace(/^_+|_+$/g, "");
    return `n_${safe || "root"}`;
  }

  function normalizeNode(obj, fallbackName = "Untitled Node", pathParts = []) {
    if (obj === null || obj === undefined) return null;

    // Canonical tree node shape: { name, children }
    if (typeof obj === "object" && ("name" in obj || "children" in obj)) {
      const name = String(obj.name ?? fallbackName);
      const childrenRaw = Array.isArray(obj.children) ? obj.children : [];
      const backendId = String(obj.id || "").trim();
      const canonicalId = String(obj.canonicalId || obj.canonicalTraceId || "").trim();
      const groupCanonicalId = String(obj.groupCanonicalId || obj.traceGroupCanonicalId || "").trim();
      const node = {
        id: backendId || buildNodeId(pathParts),
        canonicalId,
        groupCanonicalId,
        canonicalTraceId: canonicalId,
        traceGroupCanonicalId: groupCanonicalId,
        name: name || "Untitled Node",
        nodeType: String(obj.nodeType || ""),
        sourceKind: String(obj.sourceKind || ""),
        traceAliases: Array.isArray(obj.traceAliases) ? obj.traceAliases.map((item) => String(item || "").trim()).filter(Boolean) : [],
        children: []
      };
      node.children = childrenRaw
        .map((child, idx) => normalizeNode(child, `[${idx}]`, pathParts.concat([`c_${idx}`])))
        .filter(Boolean);
      return node;
    }

    // Plain JSON object (e.g. table dict): expand keys into child nodes.
    if (typeof obj === "object" && !Array.isArray(obj)) {
      const node = {
        id: buildNodeId(pathParts),
        canonicalId: "",
        groupCanonicalId: "",
        canonicalTraceId: "",
        traceGroupCanonicalId: "",
        name: String(fallbackName || "Untitled Node"),
        nodeType: "",
        sourceKind: "",
        traceAliases: [],
        children: []
      };
      const entries = Object.entries(obj);
      node.children = entries
        .map(([key, value]) => normalizeNode(value, String(key), pathParts.concat([`k_${String(key)}`])))
        .filter(Boolean);
      return node;
    }

    // Array without canonical node shape: convert each item to [idx] child.
    if (Array.isArray(obj)) {
      const node = {
        id: buildNodeId(pathParts),
        canonicalId: "",
        groupCanonicalId: "",
        canonicalTraceId: "",
        traceGroupCanonicalId: "",
        name: String(fallbackName || "Untitled Node"),
        nodeType: "",
        sourceKind: "",
        traceAliases: [],
        children: []
      };
      node.children = obj
        .map((item, idx) => normalizeNode(item, `[${idx}]`, pathParts.concat([`i_${idx}`])))
        .filter(Boolean);
      return node;
    }

    // Primitive leaf:
    // - If this primitive came from an object/array key (fallbackName), preserve that key
    //   as a node and place the primitive as its child, e.g. "April -> 58621".
    // - Otherwise, use primitive text directly.
    const hasNamedWrapper = fallbackName && fallbackName !== "Untitled Node";
    if (hasNamedWrapper) {
      return {
        id: buildNodeId(pathParts),
        canonicalId: "",
        groupCanonicalId: "",
        canonicalTraceId: "",
        traceGroupCanonicalId: "",
        name: String(fallbackName),
        nodeType: "",
        sourceKind: "value_leaf",
        traceAliases: [],
        children: [{ id: buildNodeId(pathParts.concat(["v"])), canonicalId: "", groupCanonicalId: "", canonicalTraceId: "", traceGroupCanonicalId: "", name: String(obj), nodeType: "", sourceKind: "value_leaf", traceAliases: [], children: [] }]
      };
    }
    return { id: buildNodeId(pathParts), canonicalId: "", groupCanonicalId: "", canonicalTraceId: "", traceGroupCanonicalId: "", name: String(obj), nodeType: "", sourceKind: "value_leaf", traceAliases: [], children: [] };
  }

  function normalizeForest(raw) {
    if (raw && typeof raw === "object" && raw.version === "v2" && Array.isArray(raw.roots)) {
      return raw.roots
        .map((item, idx) => normalizeNode(item, "Untitled Node", ["root", `typed_${idx}`]))
        .filter(Boolean);
    }
    if (Array.isArray(raw)) {
      const canonical = raw.every(item => item && typeof item === "object" && ("name" in item || "children" in item));
      if (canonical) return raw.map((item, idx) => normalizeNode(item, "Untitled Node", ["root", `r_${idx}`])).filter(Boolean);
      return raw.map((item, idx) => normalizeNode(item, `[${idx}]`, ["root", `r_${idx}`])).filter(Boolean);
    }
    if (raw && typeof raw === "object") {
      const canonical = ("name" in raw) || ("children" in raw);
      if (canonical) return [normalizeNode(raw, "Untitled Node", ["root"])].filter(Boolean);
      return Object.entries(raw)
        .map(([key, value]) => normalizeNode(value, String(key), ["root", `k_${String(key)}`]))
        .filter(Boolean);
    }
    return [];
  }

  // Export column-mode in-memory tree to canonical dict/list JSON.
  // Rule:
  // 1) Drop outer "flat column view".
  // 2) For each index_node:
  //    - if its body children are leaf B_NODEs -> value is list (always list, comma-split each leaf text)
  //    - else -> value is nested dict recursively built from descendants.
  function exportCanonicalFromStateNodes(nodes = []) {
    function splitCommaValues(text) {
      return String(text ?? "")
        .split(",")
        .map((x) => x.trim())
        .filter((x) => x !== "");
    }

    function isIndexNode(node) {
      if (!node || typeof node !== "object") return false;
      const t = String(node.nodeType || "").toUpperCase();
      const s = String(node.sourceKind || "").toLowerCase();
      return t === "M_NODE" || s === "index_node";
    }

    function isLeafBNode(node) {
      if (!node || typeof node !== "object") return false;
      const t = String(node.nodeType || "").toUpperCase();
      const children = Array.isArray(node.children) ? node.children : [];
      return t === "B_NODE" && children.length === 0;
    }

    function collectIndexNodes(children = []) {
      const result = [];
      const queue = Array.isArray(children) ? [...children] : [];
      while (queue.length) {
        const cur = queue.shift();
        if (!cur || typeof cur !== "object") continue;
        if (isIndexNode(cur)) {
          result.push(cur);
          continue;
        }
        const ch = Array.isArray(cur.children) ? cur.children : [];
        for (const item of ch) queue.push(item);
      }
      return result;
    }

    function parseIndexNode(node) {
      if (!node || typeof node !== "object") return undefined;
      const name = String(node.name || "");
      const children = Array.isArray(node.children) ? node.children : [];
      if (!name) return undefined;
      if (!children.length) return { [name]: [] };

      // 判断“叶子值模式”采用第一个子节点；后续由用户约束同层一致。
      const firstChild = children[0];
      const firstIsLeaf = isLeafBNode(firstChild);
      if (firstIsLeaf) {
        const values = [];
        for (const child of children) {
          if (!child || typeof child !== "object") continue;
          const leafText = String(child.name ?? "");
          const parts = splitCommaValues(leafText);
          if (parts.length) values.push(...parts);
          else values.push("");
        }
        return { [name]: values };
      }

      const nestedDict = {};
      const nestedIndexNodes = collectIndexNodes(children);
      for (const idxNode of nestedIndexNodes) {
        const parsed = parseIndexNode(idxNode);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) continue;
        for (const [k, v] of Object.entries(parsed)) nestedDict[k] = v;
      }
      return { [name]: nestedDict };
    }

    const roots = Array.isArray(nodes) ? nodes : [];
    if (!roots.length) return {};
    const firstRootName = String((roots[0] && roots[0].name) || "").trim().toLowerCase();
    // 严格要求：仅列模式内存树可导出 canonical
    if (firstRootName !== "flat column view") {
      return null;
    }

    const root = roots[0];
    const topChildren = Array.isArray(root.children) ? root.children : [];
    const topIndexNodes = collectIndexNodes(topChildren);
    const result = {};
    for (const idxNode of topIndexNodes) {
      const parsed = parseIndexNode(idxNode);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) continue;
      for (const [k, v] of Object.entries(parsed)) result[k] = v;
    }
    return result;
  }

  function collapseAll(nodes) {
    nodes.forEach(n => {
      state.collapsed.add(n.id);
      if (n.children?.length) collapseAll(n.children);
    });
  }

  state.nodes = normalizeForest(rawData);
  collapseAll(state.nodes);
  recordHistory("Initialize tree");

  const UNIT_X_PIX = UNIT_X;
  const UNIT_Y_PIX = UNIT_Y;

  // 查找节点
  function findNode(id, nodes = state.nodes, parent = null) {
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      if (node.id === id) return { node, parent, index: i, siblings: nodes };
      if (node.children?.length) {
        const found = findNode(id, node.children, node);
        if (found) return found;
      }
    }
    return null;
  }

  const UNIT_X_STEP = UNIT_X_PIX;
  const UNIT_Y_STEP = UNIT_Y_PIX;
  const MIN_ZOOM = 0.55;
  const MAX_ZOOM = 2.2;

  // 过滤出可见树（折叠节点不展开）
  function filterVisible(list, collapsedSet) {
    return list.map(n => {
      const children = collapsedSet.has(n.id) ? [] : filterVisible(n.children || [], collapsedSet);
      return { ...n, children };
    });
  }

  function getNodeOffset(id) {
    return state.nodeOffsets[id] || { x: 0, y: 0 };
  }

  function getVisibleNodeOrder() {
    return Array.from(treeEl.querySelectorAll(".nt-abs-node[data-id]"))
      .map((el) => String(el.dataset.id || ""))
      .filter(Boolean);
  }

  function syncSelectionToSingle(id = null) {
    state.selectedId = id || null;
    state.selectedIds = new Set(id ? [id] : []);
    state.lastSelectedId = id || null;
  }

  function toggleSelectionId(id) {
    if (!id) return;
    if (state.selectedIds.has(id)) {
      state.selectedIds.delete(id);
    } else {
      state.selectedIds.add(id);
    }
    const selectedArray = Array.from(state.selectedIds);
    state.selectedId = selectedArray.length ? selectedArray[selectedArray.length - 1] : null;
    state.lastSelectedId = id;
  }

  function selectRangeTo(id) {
    if (!id) return;
    const order = getVisibleNodeOrder();
    if (!order.length) return;
    const anchor = state.lastSelectedId && order.includes(state.lastSelectedId) ? state.lastSelectedId : (state.selectedId || id);
    const a = order.indexOf(anchor);
    const b = order.indexOf(id);
    if (a < 0 || b < 0) {
      syncSelectionToSingle(id);
      return;
    }
    const [start, end] = a <= b ? [a, b] : [b, a];
    const set = new Set(state.selectedIds || []);
    for (let i = start; i <= end; i += 1) set.add(order[i]);
    state.selectedIds = set;
    state.selectedId = id;
    state.lastSelectedId = id;
  }

  function getEffectiveSelectionIds() {
    if (state.selectedIds && state.selectedIds.size) return Array.from(state.selectedIds);
    return state.selectedId ? [state.selectedId] : [];
  }

  function pruneSelectionByExistingNodes() {
    const exists = new Set();
    walkNodes(state.nodes, (node) => exists.add(node.id));
    const next = new Set();
    Array.from(state.selectedIds || []).forEach((id) => {
      if (exists.has(id)) next.add(id);
    });
    state.selectedIds = next;
    if (state.selectedId && !exists.has(state.selectedId)) {
      state.selectedId = null;
    }
    if (!state.selectedId && next.size) {
      state.selectedId = Array.from(next).slice(-1)[0];
    }
    if (state.lastSelectedId && !exists.has(state.lastSelectedId)) {
      state.lastSelectedId = state.selectedId || null;
    }
  }

  function buildParentMap() {
    const parentMap = new Map();
    walkNodes(state.nodes, (node, parentId) => parentMap.set(node.id, parentId || null));
    return parentMap;
  }

  function isDescendantNodeId(ancestorId, targetId) {
    const parentMap = buildParentMap();
    let cur = targetId;
    while (cur) {
      const p = parentMap.get(cur);
      if (!p) break;
      if (p === ancestorId) return true;
      cur = p;
    }
    return false;
  }

  function searchNodes(query = "") {
    const q = normalizeText(query);
    if (!q) return [];
    const hits = [];
    walkNodes(state.nodes, (node) => {
      const nameNorm = normalizeText(node?.name || "");
      if (!nameNorm) return;
      if (nameNorm.includes(q) || (q.length >= 3 && q.includes(nameNorm))) {
        hits.push(node.id);
      }
    });
    return hits;
  }

  function updateSearchMeta() {
    if (!searchMetaEl) return;
    const total = state.searchResultIds.length;
    if (!total) {
      searchMetaEl.textContent = "0/0";
      return;
    }
    const cur = Math.max(0, state.searchCursor) + 1;
    searchMetaEl.textContent = `${cur}/${total}`;
  }

  function getSearchState() {
    const total = Number(state.searchResultIds?.length || 0);
    const current = total > 0 ? Math.max(1, Math.min(total, Number(state.searchCursor || 0) + 1)) : 0;
    return {
      query: String(state.searchQuery || ""),
      total,
      current
    };
  }

  function updateSearchResults(query = "", { autoFocus = true } = {}) {
    state.searchQuery = String(query || "");
    state.searchResultIds = searchNodes(state.searchQuery);
    state.searchCursor = state.searchResultIds.length ? 0 : -1;
    if (state.searchResultIds.length) {
      expandAncestorsByIds(state.searchResultIds);
    }
    renderTree();
    updateSearchMeta();
    if (autoFocus && state.searchResultIds.length) {
      requestAnimationFrame(() => focusNodeById(state.searchResultIds[state.searchCursor]));
    }
  }

  function goToSearchResult(step = 1) {
    const total = state.searchResultIds.length;
    if (!total) return;
    state.searchCursor = (state.searchCursor + step + total) % total;
    const id = state.searchResultIds[state.searchCursor];
    if (id) {
      expandAncestorsByIds([id]);
      renderTree();
      requestAnimationFrame(() => focusNodeById(id));
    }
    updateSearchMeta();
  }

  // 渲染树
  function renderTree() {
    hideTip();
    pruneSelectionByExistingNodes();
    if (state.searchQuery) {
      const nextResults = searchNodes(state.searchQuery);
      state.searchResultIds = nextResults;
      if (!nextResults.length) {
        state.searchCursor = -1;
      } else if (state.searchCursor < 0 || state.searchCursor >= nextResults.length) {
        state.searchCursor = 0;
      }
    } else {
      state.searchResultIds = [];
      state.searchCursor = -1;
    }
    addBtn.style.display = state.nodes.length ? "none" : "flex";

    if (!state.nodes.length) {
      treeEl.innerHTML = '<div class="nt-empty">Click + to start adding nodes</div>';
      treeEl.style.minHeight = "320px";
      treeEl.style.minWidth = "100%";
      return;
    }

    const visibleForest = filterVisible(state.nodes, state.collapsed);
    const layout = buildLayout(visibleForest);
    const { nodes: positioned, maxDepth, totalWidthUnits } = layout;

    const zoom = state.zoom;
    const baseWidth = Math.max(treeEl.clientWidth || 0, app.clientWidth || 600, totalWidthUnits * UNIT_X_STEP + 120);
    const leftAnchor = baseWidth / 2 - (totalWidthUnits * UNIT_X_STEP) / 2;
    const minHeight = (maxDepth + 1) * UNIT_Y_STEP + 120;
    treeEl.style.minHeight = `${Math.max(minHeight * zoom, treeBox.clientHeight || 0)}px`;
    treeEl.style.minWidth = `${Math.max(baseWidth * zoom, treeBox.clientWidth || 0)}px`;

    const searchSet = new Set(state.searchResultIds || []);
    const activeSearchId = state.searchCursor >= 0 ? state.searchResultIds[state.searchCursor] : null;
    const nodesHtml = positioned.map(p => {
      const offset = getNodeOffset(p.id);
      const xRaw = leftAnchor + p.unitX * UNIT_X_STEP + offset.x;
      const yRaw = p.y + offset.y;
      const x = xRaw * zoom;
      const y = yRaw * zoom;
      const selected = p.id === state.selectedId ? "selected" : "";
      const multiSelected = state.selectedIds.has(p.id) ? "multi-selected" : "";
      const searchHit = searchSet.has(p.id) ? "search-hit" : "";
      const searchActive = activeSearchId === p.id ? "search-active" : "";
      const typeClass = p.nodeType === "M_NODE" ? "m-node" : (p.nodeType === "B_NODE" ? "b-node" : "");
      const traceClass = [
        state.traceNodeIds.has(p.id) ? "trace-hit" : "",
        state.traceAnswerNodeId === p.id ? "trace-answer" : ""
      ].filter(Boolean).join(" ");
      const visitCount = Number(state.traceVisitCounts.get(p.id) || 0);
      const visitBadge = visitCount > 1 ? `<span class="nt-visit-badge">×${visitCount}</span>` : "";
      const fullLabel = p.name || "Untitled Node";
      const displayLabel = fullLabel.length > 5 ? `${fullLabel.slice(0,5)}...` : fullLabel;
      return `
        <div class="nt-abs-node" data-id="${p.id}" style="left:${x}px; top:${y}px;" data-unitx="${p.unitX}">
          <div class="nt-node ${typeClass} ${selected} ${multiSelected} ${searchHit} ${searchActive} ${traceClass}" data-node-type="${escapeHtml(String(p.nodeType || ""))}">
            ${visitBadge}
            <span class="nt-label" data-full="${escapeHtml(fullLabel)}" contenteditable="false">${escapeHtml(displayLabel)}</span>
          </div>
        </div>
      `;
    }).join("");

    const imagesHtml = (state.imageItems || []).map((img) => {
      const x = (Number(img.x || 0) * zoom);
      const y = (Number(img.y || 0) * zoom);
      const w = Math.max(40, Number(img.width || 240) * zoom);
      const h = Math.max(30, Number(img.height || 150) * zoom);
      return `
        <div class="nt-image-item" data-img-id="${escapeHtml(String(img.id || ""))}" style="left:${x}px; top:${y}px; width:${w}px; height:${h}px;">
          <button class="nt-image-delete" data-img-del="${escapeHtml(String(img.id || ""))}" title="Delete image" type="button">×</button>
          <img src="${escapeHtml(String(img.url || ""))}" alt="${escapeHtml(String(img.name || "Image"))}" draggable="false" />
          <div class="nt-image-cap">${escapeHtml(String(img.name || "Image"))}</div>
          <div class="nt-image-resize-handle" data-img-resize="${escapeHtml(String(img.id || ""))}" title="Drag to resize"></div>
        </div>
      `;
    }).join("");

    treeEl.innerHTML = nodesHtml + imagesHtml;
    const linesHtml = buildLinesSvgFromDom(visibleForest);
    if (linesHtml) treeEl.insertAdjacentHTML("afterbegin", linesHtml);

    bindNodeEvents();
    bindImageEvents();
    renderMinimap();
    updateSearchMeta();
  }

  function updateMinimapViewport() {
    if (!minimapEl || !minimapViewportEl || !state.minimapMeta) return;
    const meta = state.minimapMeta;
    const viewLeft = meta.offsetX + treeBox.scrollLeft * meta.scale;
    const viewTop = meta.offsetY + treeBox.scrollTop * meta.scale;
    const viewW = Math.max(10, treeBox.clientWidth * meta.scale);
    const viewH = Math.max(8, treeBox.clientHeight * meta.scale);
    minimapViewportEl.style.left = `${viewLeft}px`;
    minimapViewportEl.style.top = `${viewTop}px`;
    minimapViewportEl.style.width = `${Math.min(meta.drawW, viewW)}px`;
    minimapViewportEl.style.height = `${Math.min(meta.drawH, viewH)}px`;
  }

  function renderMinimap() {
    if (!minimapEl || !minimapSvg) return;
    const miniW = minimapEl.clientWidth || 190;
    const miniH = minimapEl.clientHeight || 128;
    const worldW = Math.max(treeEl.scrollWidth, treeBox.clientWidth || 1);
    const worldH = Math.max(treeEl.scrollHeight, treeBox.clientHeight || 1);
    const scale = Math.max(0.001, Math.min((miniW - 8) / worldW, (miniH - 8) / worldH));
    const drawW = worldW * scale;
    const drawH = worldH * scale;
    const offsetX = (miniW - drawW) / 2;
    const offsetY = (miniH - drawH) / 2;
    const searchSet = new Set(state.searchResultIds || []);
    const activeSearchId = state.searchCursor >= 0 ? state.searchResultIds[state.searchCursor] : null;

    let nodeDots = "";
    treeEl.querySelectorAll(".nt-abs-node[data-id]").forEach((holder) => {
      const id = String(holder.dataset.id || "");
      const x = Number(parseFloat(holder.style.left || "0")) * scale + offsetX;
      const y = Number(parseFloat(holder.style.top || "0")) * scale + offsetY;
      const isSel = state.selectedIds.has(id);
      const isActive = activeSearchId === id;
      const isHit = searchSet.has(id);
      const fill = isActive ? "rgba(250, 204, 21, 0.95)" : (isSel ? "rgba(125, 211, 252, 0.9)" : (isHit ? "rgba(251, 191, 36, 0.85)" : "rgba(148, 163, 184, 0.75)"));
      nodeDots += `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="1.8" fill="${fill}" />`;
    });

    let imageRects = "";
    treeEl.querySelectorAll(".nt-image-item[data-img-id]").forEach((el) => {
      const x = Number(parseFloat(el.style.left || "0"));
      const y = Number(parseFloat(el.style.top || "0"));
      const w = Number(parseFloat(el.style.width || "0"));
      const h = Number(parseFloat(el.style.height || "0"));
      imageRects += `<rect x="${(x - w / 2) * scale + offsetX}" y="${(y - h / 2) * scale + offsetY}" width="${Math.max(2, w * scale)}" height="${Math.max(2, h * scale)}" fill="rgba(94, 234, 212, 0.25)" stroke="rgba(94, 234, 212, 0.65)" stroke-width="0.8" />`;
    });

    minimapSvg.setAttribute("viewBox", `0 0 ${miniW} ${miniH}`);
    minimapSvg.innerHTML = `
      <rect x="${offsetX}" y="${offsetY}" width="${drawW}" height="${drawH}" fill="rgba(15,23,42,0.36)" stroke="rgba(100,116,139,0.6)" stroke-width="0.8" />
      ${imageRects}
      ${nodeDots}
    `;
    state.minimapMeta = { scale, offsetX, offsetY, drawW, drawH, worldW, worldH };
    updateMinimapViewport();
  }

  // 构建布局：通过子树宽度计算对称位置
  function buildLayout(forest) {
    function measure(node) {
      if (!node.children || !node.children.length) {
        node._w = 1;
        return 1;
      }
      let sum = 0;
      node.children.forEach(ch => { sum += measure(ch); });
      node._w = Math.max(1, sum);
      return node._w;
    }

    const virtualRoot = { id: "root", children: forest };
    const totalWidthUnits = measure(virtualRoot);

    const positioned = [];
    let maxDepth = 0;

    function assign(node, startUnit, depth) {
      if (!node.children) node.children = [];
      const centerUnit = startUnit + node._w / 2;
      if (node.id !== "root") {
        positioned.push({
          id: node.id,
          name: node.name,
          nodeType: node.nodeType || "",
          sourceKind: node.sourceKind || "",
          unitX: centerUnit,
          y: depth * UNIT_Y_STEP + 40,
          depth
        });
        if (depth > maxDepth) maxDepth = depth;
      }
      let cursor = startUnit;
      node.children.forEach(ch => {
        assign(ch, cursor, depth + 1);
        cursor += ch._w;
      });
    }

    assign(virtualRoot, 0, 0);
    return { nodes: positioned, maxDepth, totalWidthUnits };
  }

  // 构建连接线 SVG
  function buildLinesSvgFromDom(visibleForest) {
    const lines = [];
    function walk(list, parentId = null) {
      list.forEach(n => {
        if (parentId) lines.push([parentId, n.id]);
        if (n.children?.length) walk(n.children, n.id);
      });
    }
    walk(visibleForest);
    const treeRect = treeEl.getBoundingClientRect();
    const nodePos = new Map();
    treeEl.querySelectorAll(".nt-abs-node").forEach((holder) => {
      const id = holder.dataset.id;
      if (!id) return;
      const nodeEl = holder.querySelector(".nt-node") || holder;
      const rect = nodeEl.getBoundingClientRect();
      nodePos.set(id, {
        cx: rect.left - treeRect.left + rect.width / 2,
        top: rect.top - treeRect.top,
        bottom: rect.bottom - treeRect.top,
      });
    });

    const svgLines = lines.map(([pId, cId]) => {
      const p = nodePos.get(pId);
      const c = nodePos.get(cId);
      if (!p || !c) return "";
      const x1 = p.cx;
      const y1 = p.bottom;
      const x2 = c.cx;
      const y2 = c.top;
      const bend = Math.max(28, (y2 - y1) * 0.45);
      const c1x = x1;
      const c1y = y1 + bend;
      const c2x = x2;
      const c2y = y2 - bend;
      const edgeKey = `${pId}->${cId}`;
      const lineClass = [
        "nt-line",
        state.traceEdgeKeys.has(edgeKey) ? "trace-hit" : "",
        (state.traceAnswerNodeId === cId || state.traceAnswerNodeId === pId) && state.traceEdgeKeys.has(edgeKey) ? "trace-answer" : ""
      ].filter(Boolean).join(" ");
      return `<path class="${lineClass}" data-from="${pId}" data-to="${cId}" d="M ${x1} ${y1} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${x2} ${y2}" stroke="#b7b8ff" fill="none" marker-end="url(#nt-arrow)" />`;
    }).join("");
    const arrowDef = `
      <marker id="nt-arrow" markerWidth="7" markerHeight="6" refX="6.2" refY="3" orient="auto">
        <polygon points="0 0, 7 3, 0 6" fill="#b7b8ff"></polygon>
      </marker>
    `;
    const svgWidth = Math.max(treeEl.scrollWidth, treeBox.clientWidth || 0);
    const svgHeight = Math.max(treeEl.scrollHeight, treeBox.clientHeight || 0);
    return `<svg class="nt-lines" width="${svgWidth}" height="${svgHeight}" viewBox="0 0 ${Math.max(svgWidth, 1)} ${Math.max(svgHeight, 1)}" preserveAspectRatio="none"><defs>${arrowDef}</defs>${svgLines}</svg>`;
  }

  // 绑定事件
  function bindNodeEvents() {
    treeEl.querySelectorAll(".nt-node").forEach(el => {
      el.onmousedown = (e) => {
        if (e.button !== 0) return;
        if (e.ctrlKey || e.metaKey || e.shiftKey) return;
        const holder = el.closest(".nt-abs-node");
        const id = holder?.dataset.id;
        if (!id) return;
        e.stopPropagation();
        const offset = getNodeOffset(id);
        state.nodeDrag = {
          id,
          startX: e.clientX,
          startY: e.clientY,
          startOffsetX: offset.x,
          startOffsetY: offset.y,
          moved: false
        };
      };
      el.onclick = (e) => {
        if (Date.now() < state.suppressClickUntil) return;
        const holder = el.closest(".nt-abs-node");
        const id = holder?.dataset.id || null;
        if (!id) return;
        if (e.ctrlKey || e.metaKey) {
          toggleSelectionId(id);
          hideMenu();
          renderTree();
          return;
        }
        if (e.shiftKey) {
          selectRangeTo(id);
          hideMenu();
          renderTree();
          return;
        }
        syncSelectionToSingle(id);
        toggleCollapse(id);
        hideMenu();
        renderTree();
      };
      el.oncontextmenu = (e) => {
        e.preventDefault();
        if (Date.now() < state.suppressClickUntil) return;
        const holder = el.closest(".nt-abs-node");
        const id = holder?.dataset.id || null;
        if (!id) return;
        if (!state.selectedIds.has(id)) {
          syncSelectionToSingle(id);
        } else {
          state.selectedId = id;
          state.lastSelectedId = id;
        }
        state.menuTarget = holder?.dataset.id || null;
        showMenu(e.clientX, e.clientY);
        renderTree();
      };
      el.onmouseenter = () => {
        if (tipHideTimer) {
          clearTimeout(tipHideTimer);
          tipHideTimer = null;
        }
        const holder = el.closest(".nt-abs-node");
        const id = holder?.dataset.id || "";
        showTip(el, id);
      };
      el.onmouseleave = () => {
        if (tipHideTimer) clearTimeout(tipHideTimer);
        tipHideTimer = setTimeout(() => hideTip(), 120);
      };
    });
  }

  function bindImageEvents() {
    treeEl.querySelectorAll(".nt-image-item").forEach((el) => {
      el.onmousedown = (e) => {
        if (e.button !== 0) return;
        const imgId = String(el.dataset.imgId || "");
        if (!imgId) return;
        e.stopPropagation();
        const item = (state.imageItems || []).find((x) => String(x.id) === imgId);
        if (!item) return;
        state.imageDrag = {
          id: imgId,
          startX: e.clientX,
          startY: e.clientY,
          startPosX: Number(item.x || 0),
          startPosY: Number(item.y || 0),
          moved: false
        };
      };
      el.onclick = () => {
        if (Date.now() < state.suppressClickUntil) return;
        const imgId = String(el.dataset.imgId || "");
        if (!imgId) return;
        const item = (state.imageItems || []).find((x) => String(x.id) === imgId);
        if (!item || !item.url) return;
        openImagePreview(item);
      };
      el.ondblclick = () => {
        const imgId = String(el.dataset.imgId || "");
        if (!imgId) return;
        removeOverlayImageById(imgId);
      };
    });
    treeEl.querySelectorAll(".nt-image-delete").forEach((btn) => {
      btn.onclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        const imgId = String(btn.dataset.imgDel || "");
        if (!imgId) return;
        removeOverlayImageById(imgId);
      };
    });
    treeEl.querySelectorAll(".nt-image-resize-handle").forEach((handle) => {
      handle.onmousedown = (e) => {
        if (e.button !== 0) return;
        const imgId = String(handle.dataset.imgResize || "");
        if (!imgId) return;
        e.preventDefault();
        e.stopPropagation();
        const item = (state.imageItems || []).find((x) => String(x.id) === imgId);
        if (!item) return;
        state.imageResize = {
          id: imgId,
          startX: e.clientX,
          startY: e.clientY,
          startW: Number(item.width || 240),
          startH: Number(item.height || 150),
          moved: false
        };
      };
    });
  }

  async function requestDeleteOverlayImageAsset(item) {
    const convId = String(state.editorConversationId || "").trim();
    if (!convId || !item) return;
    const imgId = encodeURIComponent(String(item.id || ""));
    if (!imgId) return;
    try {
      await fetch(`/api/history/${encodeURIComponent(convId)}/tree-images/${imgId}`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: String(item.id || ""),
          url: String(item.url || ""),
          name: String(item.name || "")
        })
      });
    } catch (e) {
      // ignore
    }
  }

  function removeOverlayImageById(imgId) {
    const idx = (state.imageItems || []).findIndex((x) => String(x.id) === String(imgId));
    if (idx < 0) return;
    const removed = state.imageItems[idx];
    state.imageItems.splice(idx, 1);
    renderTree();
    recordHistory("Delete image");
    persistImageLayout();
    requestDeleteOverlayImageAsset(removed);
  }

  function showMenu(x, y) {
    menuEl.style.display = "block";
    const rect = app.getBoundingClientRect();
    menuEl.style.left = (x - rect.left + app.scrollLeft) + "px";
    menuEl.style.top = (y - rect.top + app.scrollTop) + "px";
  }

  function hideMenu() {
    menuEl.style.display = "none";
    state.menuTarget = null;
  }

  function openImagePreview(item) {
    if (!imagePreviewMask || !imagePreviewImg || !imagePreviewCap) return;
    imagePreviewImg.src = String(item.url || "");
    imagePreviewImg.alt = String(item.name || "Image");
    imagePreviewCap.textContent = String(item.name || "Image");
    imagePreviewMask.classList.add("show");
    document.body.style.overflow = "hidden";
  }

  function closeImagePreview() {
    if (!imagePreviewMask || !imagePreviewImg || !imagePreviewCap) return;
    imagePreviewMask.classList.remove("show");
    imagePreviewImg.src = "";
    imagePreviewCap.textContent = "";
    document.body.style.overflow = "";
  }

  function showTip(targetEl, nodeId) {
    if (!tipEl) return;
    const label = targetEl?.querySelector(".nt-label");
    const fullName = (label?.dataset.full || "").trim();
    if (!fullName) {
      tipEl.style.display = "none";
      return;
    }

    const found = nodeId ? findNode(nodeId) : null;
    const childCount = found?.node?.children?.length || 0;
    const safeNodeId = String(nodeId || "").replace(/"/g, "&quot;");
    tipEl.innerHTML =
      `<div class="nt-tip-title">Node Details</div>` +
      `<div class="nt-tip-body">${escapeHtml(fullName)}</div>` +
      `<div class="nt-tip-meta">Children: ${childCount}</div>` +
      `<button class="nt-tip-copy" data-copy="${escapeHtml(fullName)}" data-id="${safeNodeId}">Copy Node Name</button>`;
    tipEl.style.display = "block";

    const treeBoxRect = treeBox.getBoundingClientRect();
    const nodeRect = targetEl.getBoundingClientRect();
    const pad = 8;

    let left = nodeRect.left - treeBoxRect.left + treeBox.scrollLeft;
    let top = nodeRect.bottom - treeBoxRect.top + treeBox.scrollTop + 8;

    const minLeft = treeBox.scrollLeft + pad;
    const maxLeft = treeBox.scrollLeft + treeBox.clientWidth - tipEl.offsetWidth - pad;
    left = Math.min(Math.max(left, minLeft), Math.max(minLeft, maxLeft));

    const maxTop = treeBox.scrollTop + treeBox.clientHeight - tipEl.offsetHeight - pad;
    if (top > maxTop) {
      top = nodeRect.top - treeBoxRect.top + treeBox.scrollTop - tipEl.offsetHeight - 8;
    }
    const minTop = treeBox.scrollTop + pad;
    top = Math.max(top, minTop);

    tipEl.style.left = `${left}px`;
    tipEl.style.top = `${top}px`;

    const copyBtn = tipEl.querySelector(".nt-tip-copy");
    if (copyBtn) {
      copyBtn.onclick = async (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        const text = copyBtn.dataset.copy || fullName;
        const copied = await copyText(text);
        copyBtn.textContent = copied ? "Copied" : "Copy failed";
        setTimeout(() => {
          copyBtn.textContent = "Copy Node Name";
        }, 900);
      };
    }
  }

  function hideTip() {
    if (!tipEl) return;
    tipEl.style.display = "none";
  }

  tipEl.addEventListener("mouseenter", () => {
    if (tipHideTimer) {
      clearTimeout(tipHideTimer);
      tipHideTimer = null;
    }
  });
  tipEl.addEventListener("mouseleave", () => hideTip());

  async function copyText(text) {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        return true;
      }
    } catch (e) {}
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return !!ok;
    } catch (e) {
      return false;
    }
  }

  function normalizeText(text) {
    return String(text || "").toLowerCase().replace(/\s+/g, "").trim();
  }

  function getNodeCanonicalId(node) {
    return String(node?.canonicalId || node?.canonicalTraceId || "").trim();
  }

  function getNodeGroupCanonicalId(node) {
    return String(node?.groupCanonicalId || node?.traceGroupCanonicalId || "").trim();
  }

  function getEventCanonicalId(ev) {
    return String(ev?.canonical_id || ev?.canonical_trace_id || "").trim();
  }

  function hashTraceText(text) {
    let hash = 2166136261;
    const raw = String(text || "");
    for (let i = 0; i < raw.length; i += 1) {
      hash ^= raw.charCodeAt(i);
      hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
    }
    return (hash >>> 0).toString(16).padStart(8, "0");
  }

  function buildTraceTreeFingerprint(catalogData = null) {
    const data = catalogData || buildTraceNodeCatalog();
    const catalog = Array.isArray(data?.catalog) ? data.catalog : [];
    if (!catalog.length) return "";
    const idToCanonical = new Map();
    catalog.forEach((item) => {
      const id = String(item?.id || "").trim();
      const canonical = String(item?.canonicalId || item?.groupCanonicalId || "").trim();
      if (id) idToCanonical.set(id, canonical);
    });
    const rows = catalog.map((item) => {
      const canonicalId = String(item?.canonicalId || "").trim();
      const groupCanonicalId = String(item?.groupCanonicalId || "").trim();
      if (!canonicalId && !groupCanonicalId) return "";
      const parentCanonicalId = String(idToCanonical.get(String(item?.parentId || "").trim()) || "").trim();
      return `${canonicalId}#${groupCanonicalId}#${parentCanonicalId}`;
    }).filter(Boolean).sort();
    if (!rows.length) return "";
    return `tf_${hashTraceText(rows.join("|"))}`;
  }

  function buildTraceNodeCatalog() {
    const catalog = [];
    const idSet = new Set();
    const parentMap = new Map();
    const canonicalMap = new Map();
    const groupMap = new Map();
    walkNodes(state.nodes, (node, parentId) => {
      const id = String(node?.id || "").trim();
      if (!id) return;
      idSet.add(id);
      parentMap.set(id, parentId || null);
      const canonicalId = getNodeCanonicalId(node);
      if (canonicalId) {
        const hits = canonicalMap.get(canonicalId) || [];
        hits.push(id);
        canonicalMap.set(canonicalId, hits);
      }
      const groupCanonicalId = getNodeGroupCanonicalId(node);
      if (groupCanonicalId) {
        const hits = groupMap.get(groupCanonicalId) || [];
        hits.push(id);
        groupMap.set(groupCanonicalId, hits);
      }
      const traceAliases = Array.isArray(node?.traceAliases) ? node.traceAliases : [];
      catalog.push({
        id,
        name: String(node?.name || ""),
        normName: normalizeText(node?.name || ""),
        canonicalId,
        groupCanonicalId,
        traceAliases: traceAliases.map((alias) => String(alias || "").trim()).filter(Boolean),
        parentId: parentId || null
      });
    });
    return { catalog, idSet, parentMap, canonicalMap, groupMap };
  }

  function extractTraceIdHints(rawId) {
    const text = String(rawId || "").trim();
    if (!text) return [];
    const hints = [];
    const seen = new Set();
    const pushHint = (value) => {
      const label = String(value || "").trim();
      const norm = normalizeText(label);
      if (!label || !norm || seen.has(norm)) return;
      seen.add(norm);
      hints.push({ text: label, norm });
    };
    const decodePart = (part) => {
      let value = String(part || "").trim();
      if (!value) return;
      if (value.includes("/") || value.includes("|")) {
        value.split(/[\/|]+/).forEach((inner) => decodePart(inner));
        return;
      }
      value = value.replace(/^ft:/i, "");
      const itemMatch = value.match(/^i_(\d+)$/i);
      if (itemMatch) {
        pushHint(`item[${itemMatch[1]}]`);
        return;
      }
      value = value.replace(/^[mb]_\d+_/i, "");
      value = value.replace(/^k_/i, "");
      value = value.replace(/^idx_\d+$/i, "");
      value = value.replace(/_subtree$/i, "");
      value = value.replace(/_group$/i, "");
      value = value.replace(/^root_/i, "");
      value = value.replace(/^ho_tree$/i, "");
      value = value.replace(/^body$/i, "");
      value = value.replace(/^group$/i, "");
      value = value.replace(/_+/g, " ").trim();
      if (!value) return;
      pushHint(value);
    };
    text.split("/").forEach(decodePart);
    return hints;
  }

  function inferTraceViewMode(catalogData = null) {
    const data = catalogData || buildTraceNodeCatalog();
    const canonicalMap = data && data.canonicalMap;
    if (canonicalMap && typeof canonicalMap.keys === "function") {
      for (const canonicalId of canonicalMap.keys()) {
        const text = String(canonicalId || "").trim();
        if (!text) continue;
        if (text.includes("_flat_column_")) return "column";
        if (text.includes("_flat_row_")) return "row";
      }
    }
    return "row";
  }

  function expandProjectedTraceIds(targetId, options = {}) {
    const expanded = [];
    const seen = new Set();
    const push = (value) => {
      const text = String(value || "").trim();
      if (!text || seen.has(text)) return;
      seen.add(text);
      expanded.push(text);
    };
    const sourceId = String(targetId || "").trim();
    if (!sourceId) return expanded;
    push(sourceId);

    const projectionMap = options.projectionMap || state.traceCanonicalProjectionMap || null;
    if (!projectionMap || typeof projectionMap !== "object") return expanded;

    const semanticToViews = projectionMap.semanticToViews || projectionMap.semantic_to_views || {};
    const semanticCompactToViews = projectionMap.semanticCompactToViews || projectionMap.semantic_compact_to_views || {};
    const semanticLegacyToCompact = projectionMap.semanticLegacyToCompact || projectionMap.semantic_legacy_to_compact || {};
    const semanticCompactToLegacy = projectionMap.semanticCompactToLegacy || projectionMap.semantic_compact_to_legacy || {};
    const rowToSemantic = projectionMap.rowCanonicalToSemantic || projectionMap.row_canonical_to_semantic || {};
    const columnToSemantic = projectionMap.columnCanonicalToSemantic || projectionMap.column_canonical_to_semantic || {};
    const rowToSemanticCompact = projectionMap.rowCanonicalToSemanticCompact || projectionMap.row_canonical_to_semantic_compact || {};
    const columnToSemanticCompact = projectionMap.columnCanonicalToSemanticCompact || projectionMap.column_canonical_to_semantic_compact || {};
    const semanticCompactToGroupIds = projectionMap.semanticCompactToGroupIds || projectionMap.semantic_compact_to_group_ids || {};
    const groupToSemanticCompacts = projectionMap.groupToSemanticCompacts || projectionMap.group_to_semantic_compacts || {};
    let semanticId = "";
    if (semanticCompactToViews && Object.prototype.hasOwnProperty.call(semanticCompactToViews, sourceId)) {
      semanticId = sourceId;
    }
    if (!semanticId && semanticLegacyToCompact && Object.prototype.hasOwnProperty.call(semanticLegacyToCompact, sourceId)) {
      semanticId = String(semanticLegacyToCompact[sourceId] || "").trim();
    }
    if (!semanticId) {
      semanticId = String(
        (rowToSemanticCompact && rowToSemanticCompact[sourceId])
        || (columnToSemanticCompact && columnToSemanticCompact[sourceId])
        || (rowToSemantic && rowToSemantic[sourceId])
        || (columnToSemantic && columnToSemantic[sourceId])
        || ""
      ).trim();
      if (semanticId && semanticLegacyToCompact && Object.prototype.hasOwnProperty.call(semanticLegacyToCompact, semanticId)) {
        semanticId = String(semanticLegacyToCompact[semanticId] || "").trim();
      }
    }
    // Playback often carries ct_semantic_* ids; canonicalMap is ct_tree_* / ft only.
    if (!semanticId && /^ct_semantic_(node|group)_/.test(sourceId)
        && semanticToViews && typeof semanticToViews[sourceId] === "object") {
      semanticId = sourceId;
    }
    if (!semanticId) return expanded;
    push(semanticId);
    const legacySemanticId = String((semanticCompactToLegacy && semanticCompactToLegacy[semanticId]) || "").trim();
    if (legacySemanticId) push(legacySemanticId);

    const viewMode = String(options.viewMode || inferTraceViewMode(options.catalogData)).trim() === "column" ? "column" : "row";
    const viewEntry = (semanticCompactToViews && typeof semanticCompactToViews[semanticId] === "object")
      ? semanticCompactToViews[semanticId]
      : (semanticToViews && typeof semanticToViews[semanticId] === "object")
          ? semanticToViews[semanticId]
          : ((semanticToViews && typeof semanticToViews[legacySemanticId] === "object")
              ? semanticToViews[legacySemanticId]
              : {});
    const primaryIds = Array.isArray(viewEntry[viewMode]) ? viewEntry[viewMode] : [];
    const secondaryIds = Array.isArray(viewEntry[viewMode === "column" ? "row" : "column"])
      ? viewEntry[viewMode === "column" ? "row" : "column"]
      : [];
    primaryIds.forEach(push);
    secondaryIds.forEach(push);

    const groupIds = Array.isArray(semanticCompactToGroupIds && semanticCompactToGroupIds[semanticId])
      ? semanticCompactToGroupIds[semanticId]
      : [];
    groupIds.forEach(push);
    const compactSet = new Set([semanticId]);
    groupIds.forEach((gid) => {
      const semList = Array.isArray(groupToSemanticCompacts && groupToSemanticCompacts[gid])
        ? groupToSemanticCompacts[gid]
        : [];
      semList.forEach((sid) => {
        const s = String(sid || "").trim();
        if (!s || compactSet.has(s)) return;
        compactSet.add(s);
        push(s);
      });
    });
    compactSet.forEach((sid) => {
      const leg = String((semanticCompactToLegacy && semanticCompactToLegacy[sid]) || "").trim();
      const entry = (semanticCompactToViews && typeof semanticCompactToViews[sid] === "object")
        ? semanticCompactToViews[sid]
        : (semanticToViews && typeof semanticToViews[sid] === "object")
            ? semanticToViews[sid]
            : (leg && semanticToViews && typeof semanticToViews[leg] === "object")
                ? semanticToViews[leg]
                : {};
      const p = Array.isArray(entry[viewMode]) ? entry[viewMode] : [];
      const s = Array.isArray(entry[viewMode === "column" ? "row" : "column"]) ? entry[viewMode === "column" ? "row" : "column"] : [];
      p.forEach(push);
      s.forEach(push);
    });
    return expanded;
  }

  function resolveTraceNodeIds(rawId, options = {}) {
    const targetId = String(rawId || "").trim();
    if (!targetId) return [];
    const catalogData = options.catalogData || buildTraceNodeCatalog();
    const { idSet, parentMap, canonicalMap, groupMap } = catalogData;
    if (idSet.has(targetId)) return [targetId];
    const preferredAncestorId = String(options.preferredAncestorId || "").trim() || null;
    const scoreCandidate = (candidateId) => {
      let score = 0;
      if (preferredAncestorId) {
        let cur = candidateId;
        let hops = 0;
        while (cur && hops < 20) {
          if (cur === preferredAncestorId) {
            score += 80 - Math.min(hops, 24);
            break;
          }
          cur = parentMap.get(cur);
          hops += 1;
        }
      }
      return score;
    };
    const chooseCandidate = (candidateIds) => {
      const uniqueIds = Array.from(new Set((candidateIds || []).map((item) => String(item || "").trim()).filter(Boolean)));
      if (!uniqueIds.length) return [];
      if (uniqueIds.length === 1) return uniqueIds;
      uniqueIds.sort((a, b) => scoreCandidate(b) - scoreCandidate(a));
      return uniqueIds;
    };

    const projectedIds = expandProjectedTraceIds(targetId, {
      ...options,
      catalogData
    });
    for (const projectedId of projectedIds) {
      const canonicalCandidates = canonicalMap.get(projectedId) || [];
      const canonicalResolved = chooseCandidate(canonicalCandidates);
      if (canonicalResolved.length) {
        return canonicalResolved;
      }
      const groupResolved = chooseCandidate(groupMap.get(projectedId) || []);
      if (groupResolved.length) {
        return groupResolved;
      }
    }
    return [];
  }

  function resolveTraceNodeId(rawId, options = {}) {
    const candidates = resolveTraceNodeIds(rawId, options);
    return candidates[0] || null;
  }

  function buildTraceEdgesFromResolvedNodes(nodeIds, parentMap) {
    const edgeIds = [];
    const seen = new Set();
    (Array.isArray(nodeIds) ? nodeIds : []).forEach((nodeId) => {
      let cur = String(nodeId || "").trim();
      while (cur) {
        const parentId = parentMap.get(cur);
        if (!parentId) break;
        const edgeKey = `${parentId}->${cur}`;
        if (!seen.has(edgeKey)) {
          seen.add(edgeKey);
          edgeIds.push(edgeKey);
        }
        cur = parentId;
      }
    });
    return edgeIds;
  }

  function walkNodes(nodes, visit, parentId = null) {
    (nodes || []).forEach((node) => {
      visit(node, parentId);
      if (node.children?.length) walkNodes(node.children, visit, node.id);
    });
  }

  function extractBracketTokens(value) {
    const tokens = [];
    const text = String(value || "");
    const matches = text.match(/\[(.*?)\]/g) || [];
    matches.forEach((m) => {
      const token = m.slice(1, -1).trim();
      if (!token) return;
      const upper = token.toUpperCase();
      const stop = new Set(["CHL","FAT","EXT","COND","FOREACH","CMP","END","N","COND","FOREACH","EQ","LT","GT","LTE","GTE"]);
      if (stop.has(upper)) return;
      if (token.length <= 1) return;
      tokens.push(token);
    });
    return tokens;
  }

  function flattenStrings(value, collector) {
    if (value === null || value === undefined) return;
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      const s = String(value).trim();
      if (s) collector.push(s);
      return;
    }
    if (Array.isArray(value)) {
      value.forEach((v) => flattenStrings(v, collector));
      return;
    }
    if (typeof value === "object") {
      Object.values(value).forEach((v) => flattenStrings(v, collector));
    }
  }

  function getAncestorIds(targetId) {
    const parentMap = new Map();
    walkNodes(state.nodes, (n, parentId) => parentMap.set(n.id, parentId));
    const res = [];
    let cur = targetId;
    while (cur) {
      const p = parentMap.get(cur);
      if (!p) break;
      res.push(p);
      cur = p;
    }
    return res;
  }

  function expandAncestorsByIds(ids) {
    (ids || []).forEach((id) => {
      getAncestorIds(id).forEach((ancestorId) => state.collapsed.delete(ancestorId));
    });
  }

  function focusNodeById(nodeId, options = {}) {
    if (!nodeId) return;
    const holder = treeEl.querySelector(`.nt-abs-node[data-id="${nodeId}"]`);
    if (!holder) return;
    const rect = holder.getBoundingClientRect();
    const boxRect = treeBox.getBoundingClientRect();
    const centerX = rect.left - boxRect.left + treeBox.scrollLeft + rect.width / 2;
    const centerY = rect.top - boxRect.top + treeBox.scrollTop + rect.height / 2;
    treeBox.scrollTo({
      left: Math.max(0, centerX - treeBox.clientWidth / 2),
      top: Math.max(0, centerY - treeBox.clientHeight / 2),
      behavior: options.behavior === "auto" ? "auto" : "smooth"
    });
    const nodeEl = holder.querySelector(".nt-node");
    if (nodeEl && options.pulse !== false) {
      nodeEl.classList.remove("trace-pulse");
      void nodeEl.offsetWidth;
      nodeEl.classList.add("trace-pulse");
      setTimeout(() => nodeEl.classList.remove("trace-pulse"), 1050);
    }
  }

  function applyQaTrace(payload) {
    const executionTrace = Array.isArray(payload?.executionTrace) ? payload.executionTrace : [];
    const catalogData = buildTraceNodeCatalog();
    const projectionMap = payload?.canonicalProjectionMap || payload?.canonical_projection_map || null;
    state.traceCanonicalProjectionMap = projectionMap && typeof projectionMap === "object" ? projectionMap : null;
    const payloadTraceTreeFingerprint = String(payload?.traceTreeFingerprint || payload?.trace_tree_fingerprint || "").trim();
    const currentTraceTreeFingerprint = buildTraceTreeFingerprint(catalogData);
    const traceTreeFingerprintMatched = !payloadTraceTreeFingerprint
      || !currentTraceTreeFingerprint
      || payloadTraceTreeFingerprint === currentTraceTreeFingerprint;
    const unresolvedCanonicalIds = new Set();
    const markUnresolvedCanonical = (rawId) => {
      const text = String(rawId || "").trim();
      if (!text) return;
      unresolvedCanonicalIds.add(text);
    };
    const withTraceContext = (result = {}) => ({
      ...result,
      traceTreeFingerprintMatched,
      payloadTraceTreeFingerprint: payloadTraceTreeFingerprint || null,
      currentTraceTreeFingerprint: currentTraceTreeFingerprint || null,
      unresolvedCanonicalIds: Array.from(unresolvedCanonicalIds)
    });
    if (executionTrace.length) {
      const nodeCatalog = Array.isArray(catalogData?.catalog) ? catalogData.catalog : [];
      const resolveVisitNodeId = (ev) => {
        const canonicalId = getEventCanonicalId(ev);
        if (canonicalId) {
          const resolvedCanonicalId = resolveTraceNodeId(canonicalId, { catalogData });
          if (resolvedCanonicalId) return resolvedCanonicalId;
          markUnresolvedCanonical(canonicalId);
          return null;
        }
        const directId = String(ev?.frontend_node_id || "").trim();
        if (directId && nodeCatalog.some((n) => n.id === directId)) return directId;
        const labelNorm = normalizeText(ev?.node_value || "");
        if (!labelNorm) return null;
        for (const n of nodeCatalog) {
          if (!n.normName) continue;
          if (n.normName.includes(labelNorm) || (labelNorm.length >= 4 && labelNorm.includes(n.normName))) {
            return n.id;
          }
        }
        return null;
      };

      const visitEvents = executionTrace.filter((ev) => String(ev?.event_type || "") === "visit");
      const visitNodeOrder = visitEvents
        .map((ev) => resolveVisitNodeId(ev))
        .filter(Boolean);

      if (visitNodeOrder.length) {
        const visitNodeSet = new Set(visitNodeOrder);
        const visitEdgeOrder = [];
        for (let i = 1; i < visitNodeOrder.length; i += 1) {
          visitEdgeOrder.push(`${visitNodeOrder[i - 1]}->${visitNodeOrder[i]}`);
        }
        const strictAnswerRawId = String(payload?.trace?.answer_node_id || "").trim();
        let answerNodeId = strictAnswerRawId
          ? resolveTraceNodeId(strictAnswerRawId, { catalogData })
          : null;
        if (strictAnswerRawId && !answerNodeId) markUnresolvedCanonical(strictAnswerRawId);
        const answerNorm = normalizeText(payload?.answerText || "");
        if (!answerNodeId && answerNorm) {
          for (const n of nodeCatalog) {
            if (!n.normName) continue;
            if (answerNorm.includes(n.normName) || n.normName.includes(answerNorm)) {
              answerNodeId = n.id;
              break;
            }
          }
        }
        if (!answerNodeId) answerNodeId = visitNodeOrder[visitNodeOrder.length - 1] || null;
        expandAncestorsByIds(Array.from(visitNodeSet));
        // 回放前不预置橙色轨迹；橙色仅在真实回放经过后留下
        state.traceNodeIds = new Set();
        state.traceEdgeKeys = new Set();
        state.traceVisitCounts = new Map();
        state.traceAnswerNodeId = null;
        state.tracePathNodeOrder = visitNodeOrder.slice();
        state.tracePathEdgeOrder = visitEdgeOrder.slice();

        renderTree();
        requestAnimationFrame(() => focusNodeById(state.traceAnswerNodeId || visitNodeOrder[0] || null, { pulse: false }));

        return withTraceContext({
          matchedNodeCount: visitNodeSet.size,
          edgeCount: visitEdgeOrder.length,
          answerNodeId: answerNodeId,
          answerNodeName: answerNodeId ? getNodeNameById(answerNodeId) : null,
          canPlayback: visitNodeOrder.length > 0,
          traceMode: "strict_visit",
          matchedNodeIds: Array.from(visitNodeSet),
          pathNodeOrder: visitNodeOrder.slice(),
          pathEdgeOrder: visitEdgeOrder.slice()
        });
      }
    }

    const strictTrace = payload?.trace && typeof payload.trace === "object" ? payload.trace : null;
    if (strictTrace) {
      const strictMatchedRaw = Array.isArray(strictTrace.matched_node_ids) ? strictTrace.matched_node_ids : [];
      const strictNodeOrderRaw = Array.isArray(strictTrace.path_node_order) ? strictTrace.path_node_order : [];
      const strictMatched = strictMatchedRaw
        .map((rawId) => {
          const resolvedIds = resolveTraceNodeIds(rawId, { catalogData });
          if (!resolvedIds.length) markUnresolvedCanonical(rawId);
          return resolvedIds;
        })
        .flat()
        .filter(Boolean);
      const strictNodeOrder = strictNodeOrderRaw
        .map((rawId) => {
          const resolvedId = resolveTraceNodeId(rawId, { catalogData });
          if (!resolvedId) markUnresolvedCanonical(rawId);
          return resolvedId;
        })
        .filter(Boolean);
      const strictEdgeOrder = (Array.isArray(strictTrace.path_edge_order) ? strictTrace.path_edge_order : []).map((edgeId) => {
        const [rawFromId, rawToId] = String(edgeId || "").split("->");
        const fromId = resolveTraceNodeId(rawFromId, { catalogData });
        const toId = resolveTraceNodeId(rawToId, { catalogData, preferredAncestorId: fromId || null });
        if (!fromId) markUnresolvedCanonical(rawFromId);
        if (!toId) markUnresolvedCanonical(rawToId);
        return fromId && toId ? `${fromId}->${toId}` : "";
      }).filter(Boolean);
      if (!strictEdgeOrder.length && strictNodeOrder.length > 1) {
        for (let i = 1; i < strictNodeOrder.length; i += 1) {
          strictEdgeOrder.push(`${strictNodeOrder[i - 1]}->${strictNodeOrder[i]}`);
        }
      }
      const strictAnswerNodeId = resolveTraceNodeId(strictTrace.answer_node_id || null, { catalogData });
      if (strictTrace.answer_node_id && !strictAnswerNodeId) markUnresolvedCanonical(strictTrace.answer_node_id);
      const strictNodeSet = new Set(strictMatched.length ? strictMatched : strictNodeOrder);

      if (strictAnswerNodeId) strictNodeSet.add(strictAnswerNodeId);
      expandAncestorsByIds(Array.from(strictNodeSet));

      state.traceNodeIds = new Set();
      state.traceEdgeKeys = new Set();
      state.traceVisitCounts = new Map();
      state.traceAnswerNodeId = null;
      state.tracePathNodeOrder = strictNodeOrder.slice();
      state.tracePathEdgeOrder = strictEdgeOrder.slice();

      renderTree();
      requestAnimationFrame(() => focusNodeById(strictAnswerNodeId || strictNodeOrder[0] || null, { pulse: false }));

      return withTraceContext({
        matchedNodeCount: strictNodeSet.size,
        edgeCount: strictEdgeOrder.length,
        answerNodeId: strictAnswerNodeId,
        answerNodeName: strictAnswerNodeId ? getNodeNameById(strictAnswerNodeId) : null,
        canPlayback: strictEdgeOrder.length > 0,
        traceMode: "strict",
        matchedNodeIds: Array.from(strictNodeSet),
        pathNodeOrder: strictNodeOrder.slice(),
        pathEdgeOrder: strictEdgeOrder.slice()
      });
    }

    if (!traceTreeFingerprintMatched) {
      return withTraceContext({
        matchedNodeCount: 0,
        edgeCount: 0,
        answerNodeId: null,
        answerNodeName: null,
        canPlayback: false,
        traceMode: "strict_mismatch",
        matchedNodeIds: [],
        pathNodeOrder: [],
        pathEdgeOrder: []
      });
    }

    const chain = payload?.chain || {};
    const answerText = String(payload?.answerText || chain?.question_answering?.final_answer || "");
    const tokenCollector = [];
    const orderedTokens = [];

    const subqueries = chain?.question_answering?.subqueries || [];
    subqueries.forEach((sq) => {
      const rp = sq?.reasoning_path;
      if (Array.isArray(rp)) {
        rp.forEach((x) => {
          const tks = extractBracketTokens(x);
          tokenCollector.push(...tks);
          orderedTokens.push(...tks);
        });
      } else {
        const tks = extractBracketTokens(rp);
        tokenCollector.push(...tks);
        orderedTokens.push(...tks);
      }
      flattenStrings(sq?.retrieved_data, tokenCollector);
    });
    tokenCollector.push(...extractBracketTokens(answerText));

    const cleanedTokens = Array.from(new Set(
      tokenCollector
        .map((x) => String(x || "").trim())
        .filter((x) => x.length >= 2 && x.length <= 80)
    ));

    const matchedIds = [];
    const tokenNorms = cleanedTokens.map((t) => normalizeText(t)).filter(Boolean);
    walkNodes(state.nodes, (node) => {
      const name = String(node.name || "");
      const normName = normalizeText(name);
      if (!normName) return;
      const matched = tokenNorms.some((tk) => normName.includes(tk) || (tk.length >= 4 && tk.includes(normName)));
      if (matched) matchedIds.push(node.id);
    });

    // 答案节点优先从答案文本匹配，否则取最后一个命中的链路节点
    let answerNodeId = null;
    if (answerText) {
      const answerNorm = normalizeText(answerText);
      if (answerNorm) {
        walkNodes(state.nodes, (node) => {
          if (answerNodeId) return;
          const normName = normalizeText(node.name || "");
          if (normName && answerNorm.includes(normName)) answerNodeId = node.id;
        });
      }
    }
    if (!answerNodeId && matchedIds.length) answerNodeId = matchedIds[matchedIds.length - 1];

    const finalNodeSet = new Set(matchedIds);
    if (answerNodeId) finalNodeSet.add(answerNodeId);

    expandAncestorsByIds(Array.from(finalNodeSet));

    // 根据命中节点回溯祖先，生成路径高亮
    const edgeSet = new Set();
    const parentMap = new Map();
    walkNodes(state.nodes, (n, parentId) => parentMap.set(n.id, parentId));
    finalNodeSet.forEach((id) => {
      let cur = id;
      while (cur) {
        const p = parentMap.get(cur);
        if (!p) break;
        edgeSet.add(`${p}->${cur}`);
        cur = p;
      }
    });

    state.traceNodeIds = new Set();
    state.traceEdgeKeys = new Set();
    state.traceVisitCounts = new Map();
    state.traceAnswerNodeId = null;

    // 用于路径回放：优先使用思维链条顺序（reasoning_path token顺序）
    const pathNodeOrder = [];
    const pathEdgeOrder = [];
    const findBestNodeByToken = (token) => {
      const tk = normalizeText(token);
      if (!tk) return null;
      let best = null;
      walkNodes(state.nodes, (node) => {
        if (best) return;
        const normName = normalizeText(node.name || "");
        if (!normName) return;
        if (normName.includes(tk) || (tk.length >= 4 && tk.includes(normName))) {
          best = node.id;
        }
      });
      return best;
    };
    const uniqueOrderedNodes = [];
    const seenOrdered = new Set();
    orderedTokens.forEach((tk) => {
      const id = findBestNodeByToken(tk);
      if (!id || seenOrdered.has(id)) return;
      seenOrdered.add(id);
      uniqueOrderedNodes.push(id);
    });
    if (answerNodeId && !seenOrdered.has(answerNodeId)) {
      uniqueOrderedNodes.push(answerNodeId);
    }

    const addPathRootTo = (targetId) => {
      const rev = [targetId];
      let cur = targetId;
      while (cur) {
        const p = parentMap.get(cur);
        if (!p) break;
        rev.push(p);
        cur = p;
      }
      const seq = rev.reverse();
      for (let i = 0; i < seq.length; i += 1) {
        const nid = seq[i];
        if (pathNodeOrder[pathNodeOrder.length - 1] !== nid) {
          pathNodeOrder.push(nid);
        }
        if (i > 0) {
          const ek = `${seq[i - 1]}->${seq[i]}`;
          if (!pathEdgeOrder.includes(ek)) pathEdgeOrder.push(ek);
        }
      }
    };

    if (uniqueOrderedNodes.length) {
      uniqueOrderedNodes.forEach((nid) => addPathRootTo(nid));
    } else if (answerNodeId) {
      addPathRootTo(answerNodeId);
    }
    state.tracePathNodeOrder = pathNodeOrder;
    state.tracePathEdgeOrder = pathEdgeOrder;

    renderTree();
    requestAnimationFrame(() => focusNodeById(answerNodeId || matchedIds[0] || null, { pulse: false }));

    return withTraceContext({
      matchedNodeCount: finalNodeSet.size,
      edgeCount: edgeSet.size,
      answerNodeId: answerNodeId || null,
      answerNodeName: answerNodeId ? getNodeNameById(answerNodeId) : null,
      canPlayback: pathEdgeOrder.length > 0,
      traceMode: "inferred",
      matchedNodeIds: Array.from(finalNodeSet),
      pathNodeOrder: pathNodeOrder.slice(),
      pathEdgeOrder: pathEdgeOrder.slice()
    });
  }

  function setTraceCanonicalProjectionMap(projectionMap = null) {
    state.traceCanonicalProjectionMap = projectionMap && typeof projectionMap === "object"
      ? projectionMap
      : null;
    return { ok: true, hasProjectionMap: !!state.traceCanonicalProjectionMap };
  }

  function clearQaTrace() {
    stopTracePlayback();
    clearTraceFlashDom();
    state.traceNodeIds = new Set();
    state.traceEdgeKeys = new Set();
    state.traceVisitCounts = new Map();
    state.traceAnswerNodeId = null;
    state.tracePathNodeOrder = [];
    state.tracePathEdgeOrder = [];
    renderTree();
  }

  function normalizeTraceSegment(segment = {}) {
    const rawNodeIds = Array.isArray(segment?.nodeIds)
      ? segment.nodeIds.map((id) => String(id || "").trim()).filter(Boolean)
      : [];
    const rawEdgeIds = Array.isArray(segment?.edgeIds)
      ? segment.edgeIds.map((id) => String(id || "").trim()).filter(Boolean)
      : [];
    const catalogData = buildTraceNodeCatalog();
    const segmentViewMode = String(segment?.viewMode || "").trim().toLowerCase() === "row" ? "row" : "column";
    const nodeAliasMap = new Map();
    const nodeIds = [];
    let previousResolvedId = null;
    rawNodeIds.forEach((rawId) => {
      const resolvedIds = resolveTraceNodeIds(rawId, {
        catalogData,
        viewMode: segmentViewMode,
        preferredAncestorId: previousResolvedId
      });
      if (!resolvedIds.length) return;
      nodeAliasMap.set(rawId, resolvedIds.slice());
      resolvedIds.forEach((resolvedId) => {
        if (!nodeIds.includes(resolvedId)) {
          nodeIds.push(resolvedId);
        }
      });
      previousResolvedId = resolvedIds[resolvedIds.length - 1] || previousResolvedId;
    });
    let edgeIds = rawEdgeIds.map((edgeId) => {
      const [rawFromId, rawToId] = String(edgeId || "").split("->");
      const fromIds = nodeAliasMap.get(rawFromId) || resolveTraceNodeIds(rawFromId, {
        catalogData,
        viewMode: segmentViewMode
      });
      const fromId = Array.isArray(fromIds) ? (fromIds[0] || null) : null;
      const toIds = nodeAliasMap.get(rawToId) || resolveTraceNodeIds(rawToId, {
        catalogData,
        viewMode: segmentViewMode,
        preferredAncestorId: fromId
      });
      const toId = Array.isArray(toIds) ? (toIds[0] || null) : null;
      if (!fromId || !toId) return "";
      return `${fromId}->${toId}`;
    }).filter(Boolean);
    const deriveEdges = segment?.deriveEdges !== false;
    if (deriveEdges && !edgeIds.length && nodeIds.length > 0) {
      edgeIds = buildTraceEdgesFromResolvedNodes(nodeIds, catalogData.parentMap);
    }
    if (deriveEdges && !edgeIds.length && nodeIds.length > 1) {
      for (let i = 1; i < nodeIds.length; i += 1) {
        edgeIds.push(`${nodeIds[i - 1]}->${nodeIds[i]}`);
      }
    }
    const resolvedAnswerNodeId = resolveTraceNodeId(segment?.answerNodeId, {
      catalogData,
      viewMode: segmentViewMode,
      preferredAncestorId: nodeIds[nodeIds.length - 1] || null
    });
    const resolvedFocusNodeId = resolveTraceNodeId(segment?.focusNodeId, {
      catalogData,
      viewMode: segmentViewMode,
      preferredAncestorId: resolvedAnswerNodeId || nodeIds[nodeIds.length - 1] || null
    });
    const normalized = {
      nodeIds,
      edgeIds,
      answerNodeId: resolvedAnswerNodeId || (nodeIds[nodeIds.length - 1] || null),
      resetTraceHits: segment?.resetTraceHits !== false,
      focusNodeId: resolvedFocusNodeId || (nodeIds[nodeIds.length - 1] || nodeIds[0] || null)
    };
    const debugReason = !rawNodeIds.length
      ? "no-raw-nodeIds"
      : (!normalized.nodeIds.length ? "no-resolved-nodeIds" : "ok");
    try {
      window.__traceResolveDebug = {
        ts: Date.now(),
        rawNodeIds,
        rawEdgeIds,
        reason: debugReason,
        normalized,
      };
      if (!normalized.nodeIds.length) {
        console.warn("[trace-debug] normalizeTraceSegment no-path", window.__traceResolveDebug);
      } else {
        console.info("[trace-debug] normalizeTraceSegment", window.__traceResolveDebug);
      }
    } catch (e) {
      // ignore
    }
    return normalized;
  }

  function applyTraceSegment(segment = {}) {
    const normalized = normalizeTraceSegment(segment);
    if (!normalized.nodeIds.length) {
      return { applied: false, reason: "no-path" };
    }
    stopTracePlayback();
    expandAncestorsByIds(normalized.nodeIds);
    if (normalized.resetTraceHits) {
      state.traceNodeIds = new Set();
      state.traceEdgeKeys = new Set();
      state.traceVisitCounts = new Map();
    }
    state.traceNodeIds = new Set(normalized.nodeIds);
    state.traceEdgeKeys = new Set(normalized.edgeIds);
    state.traceAnswerNodeId = normalized.answerNodeId || null;
    state.tracePathNodeOrder = normalized.nodeIds.slice();
    state.tracePathEdgeOrder = normalized.edgeIds.slice();
    renderTree();
    requestAnimationFrame(() => focusNodeById(normalized.focusNodeId || normalized.answerNodeId, { pulse: false }));
    return {
      applied: true,
      canPlayback: normalized.nodeIds.length > 0,
      pathNodeOrder: normalized.nodeIds.slice(),
      pathEdgeOrder: normalized.edgeIds.slice(),
      answerNodeId: normalized.answerNodeId
    };
  }

  function previewTraceSegment(segment = {}) {
    return applyTraceSegment({ ...segment, resetTraceHits: true });
  }

  function playTraceSegment(segment = {}) {
    const applied = applyTraceSegment(segment);
    if (!applied?.applied) return { started: false, reason: applied?.reason || "no-path" };
    return startTracePlayback();
  }

  function clearPlaybackDom() {
    treeEl.querySelectorAll(".trace-playing").forEach((el) => el.classList.remove("trace-playing"));
  }

  function clearTraceFlashDom(options = {}) {
    const opts = options || {};
    if (!opts.preserveSequence && Array.isArray(state.traceSequenceTimers) && state.traceSequenceTimers.length) {
      state.traceSequenceTimers.forEach((timerId) => clearTimeout(timerId));
      state.traceSequenceTimers = [];
    }
    if (state.traceFlashTimer) {
      clearTimeout(state.traceFlashTimer);
      state.traceFlashTimer = null;
    }
    treeEl.querySelectorAll(".trace-flash").forEach((el) => el.classList.remove("trace-flash"));
  }

  function applyFlashToNormalizedSegment(normalized, durationMs, options = {}) {
    if (!normalized || !normalized.nodeIds.length) return;
    clearTraceFlashDom({ preserveSequence: options.preserveSequence === true });
    const focusTargetId = normalized.focusNodeId || normalized.answerNodeId || normalized.nodeIds[normalized.nodeIds.length - 1] || normalized.nodeIds[0] || null;
    focusNodeById(focusTargetId, { pulse: false, behavior: "auto" });
    requestAnimationFrame(() => {
      normalized.nodeIds.forEach((nodeId) => {
        const holder = treeEl.querySelector(`.nt-abs-node[data-id="${nodeId}"]`);
        const nodeEl = holder?.querySelector(".nt-node");
        if (nodeEl) nodeEl.classList.add("trace-flash");
      });
      normalized.edgeIds.forEach((edgeKey) => {
        const [fromId, toId] = String(edgeKey || "").split("->");
        if (!fromId || !toId) return;
        const edgeEl = treeEl.querySelector(`.nt-line[data-from="${fromId}"][data-to="${toId}"]`);
        if (edgeEl) edgeEl.classList.add("trace-flash");
      });
    });
    state.traceFlashTimer = setTimeout(() => {
      clearTraceFlashDom();
    }, durationMs);
  }

  function flashTraceSegment(segment = {}, options = {}) {
    const normalized = normalizeTraceSegment(segment);
    if (!normalized.nodeIds.length) {
      return { flashed: false, reason: "no-path" };
    }
    stopTracePlayback();
    const durationMs = Math.max(1200, Math.round(Number(options.duration_ms || 2200)));
    applyFlashToNormalizedSegment(normalized, durationMs, { preserveSequence: false });
    return {
      flashed: true,
      duration_ms: durationMs,
      node_count: normalized.nodeIds.length,
      edge_count: normalized.edgeIds.length
    };
  }

  function flashTraceSequence(segments = [], options = {}) {
    const normalizedSegments = (Array.isArray(segments) ? segments : [])
      .map((segment) => normalizeTraceSegment(segment))
      .filter((segment) => Array.isArray(segment.nodeIds) && segment.nodeIds.length);
    if (!normalizedSegments.length) {
      return { flashed: false, reason: "no-path" };
    }
    stopTracePlayback();
    clearTraceFlashDom();
    const gapMs = Math.max(120, Number(options.gap_ms || 180));
    const stepDurationMs = Math.max(700, Number(options.step_duration_ms || 980));
    const finalDurationMs = Math.max(stepDurationMs, Number(options.final_duration_ms || 1650));
    let offsetMs = 0;
    normalizedSegments.forEach((segment, index) => {
      const durationMs = index === normalizedSegments.length - 1 ? finalDurationMs : stepDurationMs;
      const timerId = setTimeout(() => {
        applyFlashToNormalizedSegment(segment, durationMs, { preserveSequence: true });
      }, offsetMs);
      state.traceSequenceTimers.push(timerId);
      offsetMs += durationMs + gapMs;
    });
    return {
      flashed: true,
      duration_ms: offsetMs,
      node_count: normalizedSegments.reduce((sum, segment) => sum + segment.nodeIds.length, 0),
      edge_count: normalizedSegments.reduce((sum, segment) => sum + segment.edgeIds.length, 0)
    };
  }

  function stopTracePlayback(options = {}) {
    const preserveTraceHits = !!(options && options.preserveTraceHits);
    if (state.tracePlaybackTimer) {
      clearInterval(state.tracePlaybackTimer);
      state.tracePlaybackTimer = null;
    }
    state.tracePlaybackRunning = false;
    clearPlaybackDom();
    if (!preserveTraceHits) {
      state.traceNodeIds = new Set();
      state.traceEdgeKeys = new Set();
      state.traceVisitCounts = new Map();
      state.traceAnswerNodeId = null;
    }
    renderTree();
    return { running: false };
  }

  function startTracePlayback() {
    if (!state.tracePathNodeOrder.length) return { started: false, reason: "no-path" };
    // 重新播放前仅停止旧计时器，不清空已解析出的 trace 命中集合。
    stopTracePlayback({ preserveTraceHits: true });
    state.tracePlaybackRunning = true;
    let step = 0;
    const maxSteps = state.tracePathNodeOrder.length;
    const speed = Number(state.tracePlaybackSpeed || 1);
    const intervalMs = Math.max(80, Math.round(420 / (speed > 0 ? speed : 1)));
    let prevNodeEl = null;
    let prevEdgeEl = null;
    state.tracePlaybackTimer = setInterval(() => {
      if (!state.tracePlaybackRunning) return;
      if (prevNodeEl) prevNodeEl.classList.remove("trace-playing");
      if (prevEdgeEl) prevEdgeEl.classList.remove("trace-playing");
      const nodeId = state.tracePathNodeOrder[step];
      const edgeKey = step > 0 ? `${state.tracePathNodeOrder[step - 1]}->${nodeId}` : null;
      const holder = treeEl.querySelector(`.nt-abs-node[data-id="${nodeId}"]`);
      const nodeEl = holder?.querySelector(".nt-node");
      if (nodeEl) {
        nodeEl.classList.add("trace-playing");
        prevNodeEl = nodeEl;
      } else {
        prevNodeEl = null;
      }
      if (edgeKey) {
        const [fromId, toId] = edgeKey.split("->");
        const edgeEl = treeEl.querySelector(`.nt-line[data-from="${fromId}"][data-to="${toId}"]`);
        if (edgeEl) {
          edgeEl.classList.add("trace-playing");
          prevEdgeEl = edgeEl;
        } else {
          prevEdgeEl = null;
        }
      } else {
        prevEdgeEl = null;
      }
      focusNodeById(nodeId, { pulse: false });
      step += 1;
      if (step >= maxSteps) {
        // 播放结束后保留命中高亮，便于用户停留查看。
        stopTracePlayback({ preserveTraceHits: true });
      }
    }, intervalMs);
    return { started: true, steps: maxSteps, interval_ms: intervalMs, speed };
  }

  async function saveToServer() {
    const hasSaveUi = !!(saveBtn && saveMsg);
    if (hasSaveUi) {
      saveBtn.disabled = true;
      saveMsg.textContent = "Saving...";
    }
    setSaveOverlayVisible(true);
    try {
      const origin =
        (window.location.origin === "null" || window.location.origin === "file://")
          ? (window.parent && window.parent.location ? window.parent.location.origin : "")
          : window.location.origin;
      // 使用新的API路径
      const url = (origin || "") + "/api/save_tree";
      const convId = String(state.editorConversationId || "").trim();
      const canonicalTree = exportCanonicalFromStateNodes(state.nodes);
      const currentRootName = String((state.nodes && state.nodes[0] && state.nodes[0].name) || "").trim();
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: convId,
          tree: state.nodes,
          canonical_tree: canonicalTree,
          canonical_tree_mode: canonicalTree ? "flat_column" : "none",
          ui_root_name: currentRootName
        }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      if (data.success) {
        persistImageLayout();
        if (hasSaveUi) {
          saveMsg.textContent = "Saved!";
          saveMsg.style.color = "rgba(74, 222, 128, 0.9)";
        }
        // 保存后自动刷新树视图
        setTimeout(() => {
      if (window.parent && window.parent !== window) {
            // 尝试触发父窗口的重新加载
        try {
              // 方法1: 尝试调用父窗口的reloadTree函数
              if (window.parent.reloadTree && typeof window.parent.reloadTree === 'function') {
                window.parent.reloadTree();
              } else {
                // 方法2: 尝试点击刷新按钮
                const reloadBtn = window.parent.document.querySelector('[onclick*="reloadTree"]');
                if (reloadBtn) {
                  reloadBtn.click();
                }
          }
        } catch (e) {
              console.warn("Auto refresh failed, please refresh manually", e);
            }
        }
        }, 500);
      } else {
        throw new Error(data.error || "Save failed");
      }
      return data;
    } catch (e) {
      console.error(e);
      if (hasSaveUi) {
        saveMsg.textContent = "Save failed: " + e.message;
        saveMsg.style.color = "rgba(248, 113, 113, 0.9)";
      }
      throw e;
    } finally {
      if (hasSaveUi) saveBtn.disabled = false;
      setSaveOverlayVisible(false);
    }
  }

  // 供父窗口调用
  window.saveToServer = saveToServer;
  window.getTreeData = () => state.nodes;
  window.getSelectedNodeIds = () => {
    const ids = getEffectiveSelectionIds();
    return Array.isArray(ids) ? ids.slice() : [];
  };
  window.replaceTreeData = (raw = []) => {
    const next = normalizeForest(raw);
    state.nodes = next;
    state.collapsed = new Set();
    collapseAll(state.nodes);
    syncSelectionToSingle(null);
    state.searchQuery = "";
    state.searchResultIds = [];
    state.searchCursor = -1;
    renderTree();
    return { ok: true, nodeCount: state.nodes.length };
  };
  window.zoomTree = (direction = "in") => {
    const factor = direction === "out" ? 0.9 : 1.1;
    state.zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, state.zoom * factor));
    renderTree();
  };
  window.setTreeZoom = (value = 1) => {
    const zoom = Number(value);
    if (!Number.isFinite(zoom)) return;
    state.zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom));
    renderTree();
  };
  window.applyQaTrace = applyQaTrace;
  window.clearQaTrace = clearQaTrace;
  window.focusNodeById = focusNodeById;
  window.previewTraceSegment = previewTraceSegment;
  window.playTraceSegment = playTraceSegment;
  window.flashTraceSegment = flashTraceSegment;
  window.flashTraceSequence = flashTraceSequence;
  window.startTracePlayback = startTracePlayback;
  window.stopTracePlayback = stopTracePlayback;
  window.setTraceCanonicalProjectionMap = setTraceCanonicalProjectionMap;
  window.getLastTraceResolveDebug = () => window.__traceResolveDebug || null;
  window.setEditorConversationId = (conversationId = "") => {
    state.editorConversationId = String(conversationId || "").trim();
  };
  window.setOverlayImages = (images = []) => {
    state.imageItems = Array.isArray(images) ? images.map((it) => normalizeImageItem(it)) : [];
    // 同步到时间线快照，避免回到“初始化树”时图片丢失
    const imageSnap = JSON.parse(JSON.stringify(state.imageItems || []));
    if (state.history.length) {
      if (state.history[0]) {
        state.history[0].imageItems = JSON.parse(JSON.stringify(imageSnap));
      }
      if (state.historyIndex >= 0 && state.history[state.historyIndex]) {
        state.history[state.historyIndex].imageItems = JSON.parse(JSON.stringify(imageSnap));
      }
      updateHistoryView();
    }
    renderTree();
  };
  window.addOverlayImage = (image = null) => {
    if (!image || !image.url) return;
    const item = normalizeImageItem(image);
    state.imageItems.push(item);
    renderTree();
    recordHistory("Add image");
    persistImageLayout();
  };
  window.getOverlayImages = () => (state.imageItems || []).map((it) => ({ ...it }));
  window.setTracePlaybackSpeed = (speed = 1) => {
    const s = Number(speed);
    if (!Number.isFinite(s) || s <= 0) return;
    state.tracePlaybackSpeed = s;
  };
  window.batchDeleteSelection = () => {
    const ids = getEffectiveSelectionIds();
    if (!ids.length) return { deleted: 0 };
    deleteNodeIds(ids);
    return { deleted: ids.length };
  };
  window.batchCollapseSelection = () => {
    const ids = getEffectiveSelectionIds();
    if (!ids.length) return { collapsed: 0 };
    collapseNodeIds(ids);
    return { collapsed: ids.length };
  };
  window.clearTreeSelection = () => {
    syncSelectionToSingle(null);
    renderTree();
    return { cleared: true };
  };
  window.setTreeSearchQuery = (query = "", options = {}) => {
    updateSearchResults(String(query || ""), options || {});
    return getSearchState();
  };
  window.nextTreeSearchResult = () => {
    goToSearchResult(1);
    return getSearchState();
  };
  window.prevTreeSearchResult = () => {
    goToSearchResult(-1);
    return getSearchState();
  };
  window.getTreeSearchState = () => getSearchState();

  // 添加节点：仅用于空白时添加根节点（中心 + 号）
  function addNode() {
    const node = createNode();
    state.nodes.push(node);
    syncSelectionToSingle(node.id);
    hideMenu();
    renderTree();
    recordHistory("Add root node");
    startRename(node.id);
  }

  // 给指定父节点添加子节点
  function addChildTo(parentId) {
    const target = findNode(parentId);
    if (!target) return;
    target.node.children = target.node.children || [];
    const child = createNode();
    target.node.children.push(child);
    state.collapsed.delete(parentId);
    syncSelectionToSingle(child.id);
    hideMenu();
    renderTree();
    recordHistory(`Add child node (${target.node.name || "Node"})`);
    startRename(child.id);
  }

  // 启动重命名
  function startRename(id) {
    const label = treeEl.querySelector(`.nt-abs-node[data-id="${id}"] .nt-label`);
    if (!label) return;
    label.textContent = label.dataset.full || label.textContent;
    label.setAttribute("contenteditable", "true");
    label.focus();
    const range = document.createRange();
    range.selectNodeContents(label);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);

    const stop = (commit) => {
      if (commit) commitRename(id, label.innerText.trim());
      label.setAttribute("contenteditable", "false");
      label.onkeydown = null;
      label.onblur = null;
    };

    label.onkeydown = (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        stop(true);
      } else if (e.key === "Escape") {
        e.preventDefault();
        renderTree();
      }
    };
    label.onblur = () => stop(true);
  }

  // 确认重命名
  function commitRename(id, name) {
    const target = findNode(id);
    if (!target) return;
    const oldName = target.node.name || "Untitled Node";
    const nextName = name || "Untitled Node";
    target.node.name = nextName;
    renderTree();
    if (oldName !== nextName) {
      recordHistory(`Rename: ${oldName} -> ${nextName}`);
    }
  }

  // 删除节点
  function deleteNode(id) {
    const target = findNode(id);
    if (!target) return;
    const removedName = target.node.name || "Node";
    target.siblings.splice(target.index, 1);
    state.collapsed.delete(id);
    state.selectedIds.delete(id);
    state.selectedId = state.selectedIds.size ? Array.from(state.selectedIds).slice(-1)[0] : null;
    state.lastSelectedId = state.selectedId;
    hideMenu();
    renderTree();
    recordHistory(`Delete node: ${removedName}`);
  }

  function deleteNodeIds(ids = []) {
    const list = Array.from(new Set((ids || []).filter(Boolean)));
    if (!list.length) return;
    const selectedSet = new Set(list);
    const parentMap = buildParentMap();
    const roots = list.filter((id) => {
      let cur = id;
      while (cur) {
        const p = parentMap.get(cur);
        if (!p) break;
        if (selectedSet.has(p)) return false;
        cur = p;
      }
      return true;
    });
    let removedCount = 0;
    roots.forEach((id) => {
      const target = findNode(id);
      if (!target) return;
      target.siblings.splice(target.index, 1);
      state.collapsed.delete(id);
      state.selectedIds.delete(id);
      removedCount += 1;
    });
    if (!removedCount) return;
    state.selectedId = state.selectedIds.size ? Array.from(state.selectedIds).slice(-1)[0] : null;
    state.lastSelectedId = state.selectedId;
    hideMenu();
    renderTree();
    recordHistory(`Batch delete nodes: ${removedCount}`);
  }

  function collapseNodeIds(ids = []) {
    const list = Array.from(new Set((ids || []).filter(Boolean)));
    if (!list.length) return;
    let collapsedCount = 0;
    list.forEach((id) => {
      const target = findNode(id);
      if (!target || !(target.node.children && target.node.children.length)) return;
      if (!state.collapsed.has(id)) {
        state.collapsed.add(id);
        collapsedCount += 1;
      }
    });
    if (!collapsedCount) return;
    renderTree();
    recordHistory(`Batch collapse nodes: ${collapsedCount}`);
  }

  function tryReparentNode(dragId, dropId) {
    if (!dragId || !dropId || dragId === dropId) return false;
    if (isDescendantNodeId(dragId, dropId)) return false;
    const dragTarget = findNode(dragId);
    const dropTarget = findNode(dropId);
    if (!dragTarget || !dropTarget) return false;
    if (dragTarget.parent && dragTarget.parent.id === dropId) return false;
    const movedNode = dragTarget.node;
    dragTarget.siblings.splice(dragTarget.index, 1);
    dropTarget.node.children = dropTarget.node.children || [];
    dropTarget.node.children.push(movedNode);
    state.collapsed.delete(dropId);
    delete state.nodeOffsets[dragId];
    syncSelectionToSingle(dragId);
    hideMenu();
    renderTree();
    recordHistory(`Reparent subtree: ${getNodeNameById(dragId)} -> ${getNodeNameById(dropId)}`);
    return true;
  }

  // 折叠/展开切换
  function toggleCollapse(id) {
    const target = findNode(id);
    if (!target || !(target.node.children && target.node.children.length)) return;
    if (state.collapsed.has(id)) {
      state.collapsed.delete(id);
    } else {
      state.collapsed.add(id);
    }
  }

  // 菜单点击
  menuEl.onclick = (e) => {
    const action = e.target.dataset.action;
    if (!action || !state.menuTarget) return;
    if (action === "add") addChildTo(state.menuTarget);
    if (action === "rename") startRename(state.menuTarget);
    if (action === "delete") deleteNode(state.menuTarget);
    hideMenu();
  };

  // 点击空白隐藏菜单
  document.addEventListener("click", (e) => {
    if (!menuEl.contains(e.target)) hideMenu();
  });
  if (imagePreviewMask) {
    imagePreviewMask.addEventListener("click", (e) => {
      if (e.target === imagePreviewMask) closeImagePreview();
    });
  }
  if (imagePreviewCloseBtn) {
    imagePreviewCloseBtn.onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      closeImagePreview();
    };
  }
  if (batchDeleteBtn) {
    batchDeleteBtn.onclick = () => {
      const ids = getEffectiveSelectionIds();
      if (!ids.length) return;
      deleteNodeIds(ids);
    };
  }
  if (batchCollapseBtn) {
    batchCollapseBtn.onclick = () => {
      const ids = getEffectiveSelectionIds();
      if (!ids.length) return;
      collapseNodeIds(ids);
    };
  }
  if (clearSelectionBtn) {
    clearSelectionBtn.onclick = () => {
      syncSelectionToSingle(null);
      renderTree();
    };
  }
  if (searchInputEl) {
    searchInputEl.addEventListener("input", () => {
      updateSearchResults(searchInputEl.value || "", { autoFocus: false });
    });
    searchInputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        if (e.shiftKey) goToSearchResult(-1);
        else goToSearchResult(1);
      } else if (e.key === "Escape") {
        searchInputEl.value = "";
        updateSearchResults("", { autoFocus: false });
      }
    });
  }
  if (searchPrevBtn) searchPrevBtn.onclick = () => goToSearchResult(-1);
  if (searchNextBtn) searchNextBtn.onclick = () => goToSearchResult(1);
  if (minimapEl) {
    minimapEl.addEventListener("mousedown", (e) => {
      if (!state.minimapMeta) return;
      const rect = minimapEl.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const meta = state.minimapMeta;
      const worldX = (x - meta.offsetX) / meta.scale;
      const worldY = (y - meta.offsetY) / meta.scale;
      if (!Number.isFinite(worldX) || !Number.isFinite(worldY)) return;
      treeBox.scrollTo({
        left: Math.max(0, worldX - treeBox.clientWidth / 2),
        top: Math.max(0, worldY - treeBox.clientHeight / 2),
        behavior: "smooth"
      });
    });
  }
  treeBox.addEventListener("scroll", () => updateMinimapViewport(), { passive: true });
  window.addEventListener("resize", () => renderMinimap());

  // 拖拽平移：按住空白区域可拖动画布
  treeBox.addEventListener("mousedown", (e) => {
    const interactive = e.target.closest(".nt-node, .nt-image-item, .nt-menu, .nt-node-add, .nt-label, button, input, .nt-search-bar, .nt-tools, .nt-minimap");
    if (interactive || e.button !== 0) return;
    panState.dragging = true;
    panState.startX = e.clientX;
    panState.startY = e.clientY;
    panState.startScrollLeft = treeBox.scrollLeft;
    panState.startScrollTop = treeBox.scrollTop;
    treeBox.classList.add("dragging");
  });

  document.addEventListener("mousemove", (e) => {
    if (state.nodeDrag) {
      const drag = state.nodeDrag;
      const dx = (e.clientX - drag.startX) / state.zoom;
      const dy = (e.clientY - drag.startY) / state.zoom;
      if (Math.abs(dx) > 1 || Math.abs(dy) > 1) drag.moved = true;
      state.nodeOffsets[drag.id] = {
        x: drag.startOffsetX + dx,
        y: drag.startOffsetY + dy
      };
      renderTree();
      return;
    }
    if (state.imageDrag) {
      const drag = state.imageDrag;
      const dx = (e.clientX - drag.startX) / state.zoom;
      const dy = (e.clientY - drag.startY) / state.zoom;
      if (Math.abs(dx) > 1 || Math.abs(dy) > 1) drag.moved = true;
      const idx = (state.imageItems || []).findIndex((it) => String(it.id) === String(drag.id));
      if (idx >= 0) {
        state.imageItems[idx].x = drag.startPosX + dx;
        state.imageItems[idx].y = drag.startPosY + dy;
        renderTree();
      }
      return;
    }
    if (state.imageResize) {
      const resize = state.imageResize;
      const dx = (e.clientX - resize.startX) / state.zoom;
      const dy = (e.clientY - resize.startY) / state.zoom;
      if (Math.abs(dx) > 1 || Math.abs(dy) > 1) resize.moved = true;
      const idx = (state.imageItems || []).findIndex((it) => String(it.id) === String(resize.id));
      if (idx >= 0) {
        const nextW = Math.max(80, Math.min(900, resize.startW + dx));
        const nextH = Math.max(60, Math.min(700, resize.startH + dy));
        state.imageItems[idx].width = nextW;
        state.imageItems[idx].height = nextH;
        renderTree();
      }
      return;
    }
    if (!panState.dragging) return;
    const dx = e.clientX - panState.startX;
    const dy = e.clientY - panState.startY;
    treeBox.scrollLeft = panState.startScrollLeft - dx;
    treeBox.scrollTop = panState.startScrollTop - dy;
  });

  document.addEventListener("mouseup", (e) => {
    if (state.nodeDrag) {
      if (state.nodeDrag.moved) {
        const dragId = state.nodeDrag.id;
        const dragHolderEl = treeEl.querySelector(`.nt-abs-node[data-id="${dragId}"]`);
        const prevPe = dragHolderEl ? dragHolderEl.style.pointerEvents : "";
        if (dragHolderEl) dragHolderEl.style.pointerEvents = "none";
        const dropEl = document.elementFromPoint(e.clientX, e.clientY);
        if (dragHolderEl) dragHolderEl.style.pointerEvents = prevPe || "";
        const dropHolder = dropEl?.closest(".nt-abs-node[data-id]");
        const dropId = String(dropHolder?.dataset?.id || "");
        const reparented = dropId ? tryReparentNode(dragId, dropId) : false;
        if (!reparented) {
          state.suppressClickUntil = Date.now() + 180;
          const movedName = getNodeNameById(dragId);
          recordHistory(`Move node: ${movedName}`);
        }
      }
      state.nodeDrag = null;
      return;
    }
    if (state.imageDrag) {
      if (state.imageDrag.moved) {
        state.suppressClickUntil = Date.now() + 180;
        recordHistory("Move image");
        persistImageLayout();
      }
      state.imageDrag = null;
      return;
    }
    if (state.imageResize) {
      if (state.imageResize.moved) {
        state.suppressClickUntil = Date.now() + 180;
        recordHistory("Resize image");
        persistImageLayout();
      }
      state.imageResize = null;
      return;
    }
    if (!panState.dragging) return;
    panState.dragging = false;
    treeBox.classList.remove("dragging");
  });

  treeBox.addEventListener("wheel", (e) => {
    e.preventDefault();
    const prevZoom = state.zoom;
    const scaleFactor = e.deltaY < 0 ? 1.08 : 0.92;
    const nextZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, prevZoom * scaleFactor));
    if (Math.abs(nextZoom - prevZoom) < 0.001) return;

    const rect = treeBox.getBoundingClientRect();
    const pointerX = e.clientX - rect.left;
    const pointerY = e.clientY - rect.top;
    const worldX = (treeBox.scrollLeft + pointerX) / prevZoom;
    const worldY = (treeBox.scrollTop + pointerY) / prevZoom;

    state.zoom = nextZoom;
    renderTree();

    requestAnimationFrame(() => {
      treeBox.scrollLeft = worldX * nextZoom - pointerX;
      treeBox.scrollTop = worldY * nextZoom - pointerY;
    });
  }, { passive: false });

  // 加号按钮
  addBtn.onclick = () => addNode();
  if (saveBtn) saveBtn.onclick = () => saveToServer();
  if (undoBtn) undoBtn.onclick = () => undoHistory();
  if (redoBtn) redoBtn.onclick = () => redoHistory();
  if (historyListEl) {
    historyListEl.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-history-index]");
      if (!btn) return;
      const idx = Number(btn.dataset.historyIndex);
      jumpToHistory(idx);
    });
  }
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && imagePreviewMask?.classList.contains("show")) {
      e.preventDefault();
      closeImagePreview();
      return;
    }
    if (e.key === "Delete" || e.key === "Backspace") {
      const active = document.activeElement;
      const isInput = active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA");
      if (!isInput) {
        const ids = getEffectiveSelectionIds();
        if (ids.length) {
          e.preventDefault();
          deleteNodeIds(ids);
          return;
        }
      }
    }
    const activeEl = document.activeElement;
    const isEditing = activeEl && activeEl.getAttribute && activeEl.getAttribute("contenteditable") === "true";
    if (isEditing) return;
    const isMod = e.ctrlKey || e.metaKey;
    if (isMod && String(e.key || "").toLowerCase() === "f") {
      if (searchInputEl) {
        e.preventDefault();
        searchInputEl.focus();
        searchInputEl.select();
      }
      return;
    }
    if (!isMod) return;
    const key = String(e.key || "").toLowerCase();
    if (key === "z" && !e.shiftKey) {
      e.preventDefault();
      undoHistory();
    } else if (key === "y" || (key === "z" && e.shiftKey)) {
      e.preventDefault();
      redoHistory();
    }
  });

  renderTree();
  updateHistoryView();
})();
</script>
"""


def build_new_tree_html(initial_data_path="cache/temp.ui.tree.json", initial_data=None):
    init_data = initial_data if initial_data is not None else load_initial_tree_data(path=initial_data_path)
    init_json = json.dumps(init_data, ensure_ascii=False)
    return NEW_TREE_HTML_TEMPLATE.replace("__INIT_DATA__", init_json)


def build_new_tree_iframe_html(initial_data_path="cache/temp.ui.tree.json", initial_data=None):
    """生成通过 iframe srcdoc 渲染的 HTML，保持同源以便 fetch /save_tree。"""
    raw = build_new_tree_html(initial_data_path=initial_data_path, initial_data=initial_data)
    escaped = html.escape(raw, quote=True)
    return (
        "<iframe id='new-tree-iframe' style='width:100%;min-height:720px;border:0;background:transparent;' "
        f"srcdoc=\"{escaped}\"></iframe>"
    )


import json
import os
import base64
import html


def load_initial_tree_data(path="cache/temp.json"):
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
  body { margin: 0; padding: 0; background: #0a0a0f; }
  .nt-container { 
    background: #3A3939; 
    border: none; 
    border-radius: 12px; 
    padding: 20px; 
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; 
    position: relative; 
    min-height: 100vh; 
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
    padding: 20px; 
    background: #3A3939; 
    min-height: calc(100vh - 100px); 
    overflow: auto; 
    position: relative;
    box-shadow: none;
  }
  .nt-tree { position: relative; min-height: 320px; }
  .nt-abs-node { position: absolute; transform: translate(-50%, -50%); }
  .nt-lines { position: absolute; inset: 0; pointer-events: none; }
  .nt-line { 
    stroke-width: 3 !important;
    filter: drop-shadow(0 0 4px rgba(255, 255, 255, 0.4));
  }
  .nt-hover-tip { 
    position: fixed; 
    top: 12px; 
    right: 12px; 
    min-width: 160px; 
    max-width: 260px; 
    padding: 10px 12px; 
    border: 1px solid rgba(106, 125, 227, 0.3); 
    border-radius: 10px; 
    background: rgba(26, 26, 46, 0.95);
    backdrop-filter: blur(20px);
    color: #e0e0e0; 
    font-size: 12px; 
    box-shadow: 
      0 4px 20px rgba(0, 0, 0, 0.4),
      0 0 30px rgba(106, 125, 227, 0.2);
    pointer-events: none; 
    display: none; 
    z-index: 999; 
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
  /* 滚动条样式 - 统一美化样式 */
  /* Webkit浏览器（Chrome, Safari, Edge） */
  .nt-tree-box::-webkit-scrollbar {
    width: 10px;
    height: 10px;
  }
  .nt-tree-box::-webkit-scrollbar-track {
    background: rgba(10, 10, 15, 0.5);
    border-radius: 5px;
  }
  .nt-tree-box::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, rgba(106, 125, 227, 0.6) 0%, rgba(143, 157, 240, 0.6) 100%);
    border-radius: 5px;
    box-shadow: 0 0 10px rgba(106, 125, 227, 0.5);
  }
  .nt-tree-box::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, rgba(106, 125, 227, 0.8) 0%, rgba(143, 157, 240, 0.8) 100%);
  }
  
  /* Firefox浏览器支持 */
  .nt-tree-box {
    scrollbar-width: thin;
    scrollbar-color: rgba(106, 125, 227, 0.6) rgba(10, 10, 15, 0.5);
  }
  
  /* 确保所有滚动容器都应用样式 */
  *::-webkit-scrollbar {
    width: 10px;
    height: 10px;
  }
  
  *::-webkit-scrollbar-track {
    background: rgba(10, 10, 15, 0.5);
    border-radius: 5px;
  }
  
  *::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, rgba(106, 125, 227, 0.6) 0%, rgba(143, 157, 240, 0.6) 100%);
    border-radius: 5px;
    box-shadow: 0 0 10px rgba(106, 125, 227, 0.5);
  }
  
  *::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, rgba(106, 125, 227, 0.8) 0%, rgba(143, 157, 240, 0.8) 100%);
  }
  
  * {
    scrollbar-width: thin;
    scrollbar-color: rgba(106, 125, 227, 0.6) rgba(10, 10, 15, 0.5);
  }
</style>

<div id="nt-root" class="nt-container">
  <button id="nt-add" class="nt-add-btn" title="添加节点">+</button>
  <div class="nt-tree-box">
    <div id="nt-tip" class="nt-hover-tip"></div>
    <div class="nt-tree" id="nt-tree"></div>
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
  const addBtn = app.querySelector("#nt-add");
  const menuEl = app.querySelector("#nt-menu");
  const tipEl = app.querySelector("#nt-tip");
  const saveBtn = app.querySelector("#nt-save");
  const saveMsg = app.querySelector("#nt-save-msg");

  const rawData = __INIT_DATA__;
  const state = { nodes: [], selectedId: null, menuTarget: null, collapsed: new Set() };

  const uid = () => "n-" + Math.random().toString(16).slice(2);

  const createNode = (name = "新节点") => ({ id: uid(), name, children: [] });

  const UNIT_X = 150; // 控制水平间距
  const UNIT_Y = 110; // 控制行高

  const escapeHtml = (str = "") => str.replace(/[&<>"']/g, s => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[s] || s));

  function normalizeNode(obj) {
    if (obj === null || obj === undefined) return null;
    const name = (typeof obj === "object" && "name" in obj) ? obj.name : String(obj);
    const childrenRaw = (obj && typeof obj === "object" && Array.isArray(obj.children)) ? obj.children : [];
    const node = { id: uid(), name: name || "未命名节点", children: [] };
    node.children = childrenRaw.map(normalizeNode).filter(Boolean);
    return node;
  }

  function normalizeForest(raw) {
    if (Array.isArray(raw)) return raw.map(normalizeNode).filter(Boolean);
    if (raw && typeof raw === "object") return [normalizeNode(raw)].filter(Boolean);
    return [];
  }

  function collapseAll(nodes) {
    nodes.forEach(n => {
      state.collapsed.add(n.id);
      if (n.children?.length) collapseAll(n.children);
    });
  }

  state.nodes = normalizeForest(rawData);
  collapseAll(state.nodes);

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

  // 过滤出可见树（折叠节点不展开）
  function filterVisible(list, collapsedSet) {
    return list.map(n => {
      const children = collapsedSet.has(n.id) ? [] : filterVisible(n.children || [], collapsedSet);
      return { ...n, children };
    });
  }

  // 渲染树
  function renderTree() {
    addBtn.style.display = state.nodes.length ? "none" : "flex";

    if (!state.nodes.length) {
      treeEl.innerHTML = '<div class="nt-empty">点击 + 号开始添加节点</div>';
      treeEl.style.minHeight = "320px";
      treeEl.style.minWidth = "100%";
      return;
    }

    const visibleForest = filterVisible(state.nodes, state.collapsed);
    const layout = buildLayout(visibleForest);
    const { nodes: positioned, maxDepth, totalWidthUnits } = layout;

    const baseWidth = Math.max(treeEl.clientWidth || 0, app.clientWidth || 600, totalWidthUnits * UNIT_X_STEP + 120);
    const leftAnchor = baseWidth / 2 - (totalWidthUnits * UNIT_X_STEP) / 2;
    const minHeight = (maxDepth + 1) * UNIT_Y_STEP + 120;
    treeEl.style.minHeight = `${minHeight}px`;
    treeEl.style.minWidth = `${baseWidth}px`;

    const nodesHtml = positioned.map(p => {
      const x = leftAnchor + p.unitX * UNIT_X_STEP;
      const selected = p.id === state.selectedId ? "selected" : "";
      const fullLabel = p.name || "未命名节点";
      const displayLabel = fullLabel.length > 5 ? `${fullLabel.slice(0,5)}...` : fullLabel;
      return `
        <div class="nt-abs-node" data-id="${p.id}" style="left:${x}px; top:${p.y}px;" data-unitx="${p.unitX}">
          <div class="nt-node ${selected}">
            <span class="nt-label" data-full="${escapeHtml(fullLabel)}" contenteditable="false">${escapeHtml(displayLabel)}</span>
          </div>
        </div>
      `;
    }).join("");

    const linesHtml = buildLinesSvg(visibleForest, positioned, leftAnchor, baseWidth, minHeight);
    treeEl.innerHTML = linesHtml + nodesHtml;

    bindNodeEvents();
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
  function buildLinesSvg(visibleForest, positionedNodes, leftAnchor, width, height) {
    const map = new Map(positionedNodes.map(n => [n.id, n]));
    const lines = [];
    function walk(list, parentId = null) {
      list.forEach(n => {
        if (parentId) lines.push([parentId, n.id]);
        if (n.children?.length) walk(n.children, n.id);
      });
    }
    walk(visibleForest);

    // 彩虹色数组 - 整条线不断变幻颜色
    const rainbowColors = [
      '#FF6B9D',  // 粉色
      '#FFA07A',  // 橙色
      '#FFD700',  // 金色
      '#32CD32',  // 绿色
      '#00CED1',  // 青色
      '#4169E1',  // 蓝色
      '#9370DB',  // 紫色
      '#FF69B4',  // 热粉色
    ];

    // 生成所有渐变定义和线条
    const defs = [];
    const svgLines = lines.map(([pId, cId], index) => {
      const p = map.get(pId);
      const c = map.get(cId);
      if (!p || !c) return "";
      const x1 = leftAnchor + p.unitX * UNIT_X_STEP;
      const y1 = p.y;
      const x2 = leftAnchor + c.unitX * UNIT_X_STEP;
      const y2 = c.y;
      
      // 为每条线创建多色渐变，让整条线不断变幻颜色
      const gradientId = `lineGradient-${index}`;
      
      // 创建多个颜色停止点，每段颜色短一点，形成连续变幻效果
      const numStops = 8; // 8个颜色段
      const stops = rainbowColors.map((color, i) => {
        const offset = (i / (numStops - 1)) * 100;
        return `<stop offset="${offset}%" style="stop-color:${color};stop-opacity:1" />`;
      }).join("");
      
      // 使用userSpaceOnUse让渐变沿着线条方向
      defs.push(`<linearGradient id="${gradientId}" gradientUnits="userSpaceOnUse" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}">
        ${stops}
      </linearGradient>`);
      
      // 使用渐变作为stroke
      return `<line class="nt-line" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="url(#${gradientId})" style="stroke-width: 3;" />`;
    }).join("");
    return `<svg class="nt-lines" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none"><defs>${defs.join("")}</defs>${svgLines}</svg>`;
  }

  // 绑定事件
  function bindNodeEvents() {
    treeEl.querySelectorAll(".nt-node").forEach(el => {
      el.onclick = () => {
        const holder = el.closest(".nt-abs-node");
        const id = holder?.dataset.id || null;
        state.selectedId = id;
        toggleCollapse(id);
        hideMenu();
        renderTree();
      };
      el.oncontextmenu = (e) => {
        e.preventDefault();
        const holder = el.closest(".nt-abs-node");
        state.selectedId = holder?.dataset.id || null;
        state.menuTarget = holder?.dataset.id || null;
        showMenu(e.clientX, e.clientY);
        renderTree();
      };
      el.onmouseenter = () => {
        const label = el.querySelector(".nt-label");
        showTip(label?.dataset.full || "");
      };
      el.onmouseleave = () => hideTip();
    });
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

  function showTip(text) {
    if (!tipEl) return;
    if (!text) {
      tipEl.style.display = "none";
      return;
    }
    tipEl.textContent = text;
    tipEl.style.display = "block";
  }

  function hideTip() {
    if (!tipEl) return;
    tipEl.style.display = "none";
  }

  async function saveToServer() {
    const hasSaveUi = !!(saveBtn && saveMsg);
    if (hasSaveUi) {
      saveBtn.disabled = true;
      saveMsg.textContent = "保存中...";
    }
    try {
      const origin =
        (window.location.origin === "null" || window.location.origin === "file://")
          ? (window.parent && window.parent.location ? window.parent.location.origin : "")
          : window.location.origin;
      // 使用新的API路径
      const url = (origin || "") + "/api/save_tree";
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state.nodes),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      if (data.success) {
        if (hasSaveUi) {
          saveMsg.textContent = "保存成功！";
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
              console.warn("自动刷新失败，请手动刷新", e);
            }
        }
        }, 500);
      } else {
        throw new Error(data.error || "保存失败");
      }
      if (hasSaveUi) saveBtn.disabled = false;
      return data;
    } catch (e) {
      console.error(e);
      if (hasSaveUi) {
        saveMsg.textContent = "保存失败: " + e.message;
        saveMsg.style.color = "rgba(248, 113, 113, 0.9)";
        saveBtn.disabled = false;
      }
      throw e;
    }
  }

  // 供父窗口调用
  window.saveToServer = saveToServer;
  window.getTreeData = () => state.nodes;

  // 添加节点：仅用于空白时添加根节点（中心 + 号）
  function addNode() {
    const node = createNode();
    state.nodes.push(node);
    state.selectedId = node.id;
    hideMenu();
    renderTree();
    startRename(state.selectedId);
  }

  // 给指定父节点添加子节点
  function addChildTo(parentId) {
    const target = findNode(parentId);
    if (!target) return;
    target.node.children = target.node.children || [];
    const child = createNode();
    target.node.children.push(child);
    state.collapsed.delete(parentId);
    state.selectedId = child.id;
    hideMenu();
    renderTree();
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
    target.node.name = name || "未命名节点";
    renderTree();
  }

  // 删除节点
  function deleteNode(id) {
    const target = findNode(id);
    if (!target) return;
    target.siblings.splice(target.index, 1);
    state.collapsed.delete(id);
    state.selectedId = null;
    hideMenu();
    renderTree();
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

  // 加号按钮
  addBtn.onclick = () => addNode();
  if (saveBtn) saveBtn.onclick = () => saveToServer();

  renderTree();
})();
</script>
"""


def build_new_tree_html():
    init_data = load_initial_tree_data()
    init_json = json.dumps(init_data, ensure_ascii=False)
    return NEW_TREE_HTML_TEMPLATE.replace("__INIT_DATA__", init_json)


def build_new_tree_iframe_html():
    """生成通过 iframe srcdoc 渲染的 HTML，保持同源以便 fetch /save_tree。"""
    raw = build_new_tree_html()
    escaped = html.escape(raw, quote=True)
    return (
        "<iframe id='new-tree-iframe' style='width:100%;min-height:720px;border:0;background:transparent;' "
        f"srcdoc=\"{escaped}\"></iframe>"
    )


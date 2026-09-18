/**
 * Interactive SVG Knowledge Graph Visualizer with Force-Directed Physics.
 */

class GraphVisualizer {
  constructor(svgElementId, tooltipId) {
    this.svg = document.getElementById(svgElementId);
    this.tooltip = document.getElementById(tooltipId);
    this.nodes = [];
    this.edges = [];
    this.simulation = null;
    this.transform = { x: 0, y: 0, scale: 1 };
    this.isDragging = false;
    this.dragNode = null;
    this.panStart = { x: 0, y: 0 };

    this.typeColors = {
      ALGORITHM: "#06b6d4",
      PROTOCOL: "#8b5cf6",
      SYSTEM: "#3b82f6",
      VULNERABILITY: "#f43f5e",
      ROLE: "#f59e0b",
      CONCEPT: "#10b981",
    };

    this.initEvents();
  }

  initEvents() {
    if (!this.svg) return;

    this.svg.addEventListener("mousedown", (e) => {
      if (e.target === this.svg || e.target.tagName === "g") {
        this.isDragging = true;
        this.panStart = { x: e.clientX - this.transform.x, y: e.clientY - this.transform.y };
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (this.isDragging) {
        this.transform.x = e.clientX - this.panStart.x;
        this.transform.y = e.clientY - this.panStart.y;
        this.updateTransform();
      } else if (this.dragNode) {
        const pt = this.screenToSvg(e.clientX, e.clientY);
        this.dragNode.x = pt.x;
        this.dragNode.y = pt.y;
        this.render();
      }
    });

    window.addEventListener("mouseup", () => {
      this.isDragging = false;
      this.dragNode = null;
    });

    this.svg.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.12 : 0.88;
      this.transform.scale = Math.max(0.2, Math.min(4.0, this.transform.scale * zoomFactor));
      this.updateTransform();
    });
  }

  screenToSvg(screenX, screenY) {
    const rect = this.svg.getBoundingClientRect();
    return {
      x: (screenX - rect.left - this.transform.x) / this.transform.scale,
      y: (screenY - rect.top - this.transform.y) / this.transform.scale,
    };
  }

  updateTransform() {
    const g = this.svg.querySelector(".graph-root");
    if (g) {
      g.setAttribute(
        "transform",
        `translate(${this.transform.x}, ${this.transform.y}) scale(${this.transform.scale})`
      );
    }
  }

  resetView() {
    this.transform = { x: 0, y: 0, scale: 1 };
    this.updateTransform();
  }

  setData(subgraph) {
    if (!subgraph || !subgraph.nodes || subgraph.nodes.length === 0) {
      this.svg.innerHTML = '<text x="50%" y="50%" fill="#64748b" text-anchor="middle">No graph data available</text>';
      return;
    }

    const width = this.svg.clientWidth || 800;
    const height = this.svg.clientHeight || 500;

    // Map nodes
    const nodeMap = {};
    this.nodes = subgraph.nodes.map((n, i) => {
      const angle = (i / subgraph.nodes.length) * 2 * Math.PI;
      const radius = 120 + Math.random() * 100;
      const node = {
        id: n.id,
        name: n.name || n.id,
        type: n.type || "CONCEPT",
        description: n.description || "",
        isSeed: n.is_seed || false,
        degree: n.degree || 1,
        x: width / 2 + Math.cos(angle) * radius,
        y: height / 2 + Math.sin(angle) * radius,
        vx: 0,
        vy: 0,
      };
      nodeMap[n.id] = node;
      return node;
    });

    // Map edges
    this.edges = [];
    (subgraph.edges || []).forEach((e) => {
      const src = nodeMap[e.source];
      const tgt = nodeMap[e.target];
      if (src && tgt) {
        this.edges.push({
          source: src,
          target: tgt,
          relation: e.relation || "RELATES_TO",
          weight: e.weight || 1.0,
        });
      }
    });

    // Run simple force simulation layout for 60 iterations
    this.runForceLayout(width, height, 75);
    this.render();
  }

  runForceLayout(width, height, iterations = 60) {
    const k = Math.sqrt((width * height) / (this.nodes.length + 1)) * 0.75;

    for (let iter = 0; iter < iterations; iter++) {
      // Repulsion between all node pairs
      for (let i = 0; i < this.nodes.length; i++) {
        for (let j = i + 1; j < this.nodes.length; j++) {
          const na = this.nodes[i];
          const nb = this.nodes[j];
          let dx = na.x - nb.x;
          let dy = na.y - nb.y;
          let dist = Math.sqrt(dx * dx + dy * dy) || 1;
          let force = (k * k) / dist;
          let fx = (dx / dist) * force;
          let fy = (dy / dist) * force;

          na.vx += fx * 0.05;
          na.vy += fy * 0.05;
          nb.vx -= fx * 0.05;
          nb.vy -= fy * 0.05;
        }
      }

      // Attraction along edges
      for (const e of this.edges) {
        let dx = e.target.x - e.source.x;
        let dy = e.target.y - e.source.y;
        let dist = Math.sqrt(dx * dx + dy * dy) || 1;
        let force = (dist * dist) / k;
        let fx = (dx / dist) * force * 0.06;
        let fy = (dy / dist) * force * 0.06;

        e.source.vx += fx;
        e.source.vy += fy;
        e.target.vx -= fx;
        e.target.vy -= fy;
      }

      // Gravity towards center
      const cx = width / 2;
      const cy = height / 2;
      for (const n of this.nodes) {
        n.vx += (cx - n.x) * 0.015;
        n.vy += (cy - n.y) * 0.015;

        // Apply velocity with damping
        n.x += Math.max(-15, Math.min(15, n.vx * 0.4));
        n.y += Math.max(-15, Math.min(15, n.vy * 0.4));
        n.vx *= 0.85;
        n.vy *= 0.85;
      }
    }
  }

  render() {
    if (!this.svg) return;

    let html = `
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569" />
        </marker>
        <marker id="arrow-active" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#06b6d4" />
        </marker>
      </defs>
      <g class="graph-root" transform="translate(${this.transform.x}, ${this.transform.y}) scale(${this.transform.scale})">
    `;

    // Edges
    html += '<g class="edges-layer">';
    for (const e of this.edges) {
      const midX = (e.source.x + e.target.x) / 2;
      const midY = (e.source.y + e.target.y) / 2;

      html += `
        <line x1="${e.source.x}" y1="${e.source.y}" x2="${e.target.x}" y2="${e.target.y}"
              stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
        <text x="${midX}" y="${midY}" fill="#64748b" font-size="9" font-family="'JetBrains Mono', monospace"
              text-anchor="middle" dy="-3">${e.relation}</text>
      `;
    }
    html += "</g>";

    // Nodes
    html += '<g class="nodes-layer">';
    for (const n of this.nodes) {
      const color = this.typeColors[n.type] || "#10b981";
      const radius = n.isSeed ? 16 : 12;
      const stroke = n.isSeed ? "#38bdf8" : "rgba(255,255,255,0.3)";
      const strokeWidth = n.isSeed ? 3 : 1.5;

      html += `
        <g class="node-group" data-id="${n.id}" transform="translate(${n.x}, ${n.y})" style="cursor: pointer;">
          <circle r="${radius}" fill="${color}" stroke="${stroke}" stroke-width="${strokeWidth}" />
          <text y="${radius + 14}" fill="#f8fafc" font-size="11" font-weight="600"
                text-anchor="middle" font-family="'Plus Jakarta Sans', sans-serif">${n.name}</text>
        </g>
      `;
    }
    html += "</g></g>";

    this.svg.innerHTML = html;

    // Bind node drag & hover events
    this.svg.querySelectorAll(".node-group").forEach((el) => {
      const id = el.getAttribute("data-id");
      const nodeObj = this.nodes.find((n) => n.id === id);

      el.addEventListener("mousedown", (e) => {
        e.stopPropagation();
        this.dragNode = nodeObj;
      });

      el.addEventListener("mouseenter", (e) => {
        if (!this.tooltip || !nodeObj) return;
        this.tooltip.innerHTML = `
          <strong>${nodeObj.name}</strong> (${nodeObj.type})<br/>
          <span style="color: #94a3b8; font-size: 0.75rem;">Connected to ${nodeObj.degree} entities</span>
        `;
        this.tooltip.classList.remove("hidden");
        const rect = this.svg.getBoundingClientRect();
        this.tooltip.style.left = `${e.clientX - rect.left + 15}px`;
        this.tooltip.style.top = `${e.clientY - rect.top + 15}px`;
      });

      el.addEventListener("mouseleave", () => {
        if (this.tooltip) this.tooltip.classList.add("hidden");
      });
    });
  }
}

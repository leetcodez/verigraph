/**
 * VeriGraph Main Frontend Application Controller.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Global State & Visualizers
  let mainGraphViz = null;
  let currentActiveTab = "workbench";

  // Initialize UI Visualizer
  mainGraphViz = new GraphVisualizer("mainGraphSvg", "graphTooltip");

  // Initial Data Fetch
  fetchClusterStats();
  fetchDocumentsList();
  fetchGraphData();

  // Tab Navigation
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      switchTab(targetTab);
    });
  });

  function switchTab(tabId) {
    currentActiveTab = tabId;
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));

    const activeBtn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
    const activePane = document.getElementById(`tab-${tabId}`);
    if (activeBtn) activeBtn.classList.add("active");
    if (activePane) activePane.classList.add("active");

    if (tabId === "graph") {
      fetchGraphData();
    } else if (tabId === "documents") {
      fetchDocumentsList();
    }
  }

  // Sample prompt pills
  document.querySelectorAll(".pill-btn").forEach((pill) => {
    pill.addEventListener("click", () => {
      const q = pill.getAttribute("data-query");
      document.getElementById("queryInput").value = q;
    });
  });

  // Query Execution Handler
  const queryForm = document.getElementById("queryForm");
  const queryInput = document.getElementById("queryInput");
  const responseOutput = document.getElementById("responseOutput");
  const strategyBadge = document.getElementById("strategyBadge");
  const btnSubmitQuery = document.getElementById("btnSubmitQuery");
  const faithfulnessBar = document.getElementById("faithfulnessBar");
  const faithfulnessIndicator = document.getElementById("faithfulnessIndicator");
  const claimsSummary = document.getElementById("claimsSummary");
  const sourcesList = document.getElementById("sourcesList");
  const sourcesCount = document.getElementById("sourcesCount");
  const subgraphCount = document.getElementById("subgraphCount");
  const miniGraphCanvas = document.getElementById("miniGraphCanvas");

  queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    const optGraph = document.getElementById("optGraph").checked;
    const optStream = document.getElementById("optStream").checked;

    btnSubmitQuery.disabled = true;
    btnSubmitQuery.innerHTML = "<span>Processing...</span>";
    responseOutput.innerHTML = '<span style="color: #64748b;">Decomposing query and traversing graph...</span>';
    strategyBadge.innerText = "Strategy: Planning";
    claimsSummary.innerHTML = "";
    sourcesList.innerHTML = '<p class="placeholder-text">Retrieving candidates...</p>';

    if (optStream) {
      // SSE Streaming Mode
      try {
        const response = await fetch("/api/v1/query/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: query, enable_graph: optGraph, top_k: 5 }),
        });

        if (!response.ok) throw new Error(`HTTP error ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let streamedText = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n\n");
          buffer = lines.pop(); // keep last incomplete chunk

          for (const block of lines) {
            if (!block.trim()) continue;
            let eventType = "message";
            let dataStr = "";

            for (const line of block.split("\n")) {
              if (line.startsWith("event: ")) eventType = line.substring(7).trim();
              if (line.startsWith("data: ")) dataStr = line.substring(6).trim();
            }

            if (eventType && dataStr) {
              handleSseEvent(eventType, JSON.parse(dataStr));
            }
          }
        }
      } catch (err) {
        responseOutput.innerHTML = `<span style="color: #ef4444;">Query failed: ${err.message}</span>`;
      } finally {
        btnSubmitQuery.disabled = false;
        btnSubmitQuery.innerHTML = `<span>Execute Verified Query</span>`;
      }
    } else {
      // Synchronous REST Mode
      try {
        const resp = await fetch("/api/v1/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: query, enable_graph: optGraph, top_k: 5 }),
        });
        const data = await resp.json();
        renderCompleteQueryResponse(data);
      } catch (err) {
        responseOutput.innerHTML = `<span style="color: #ef4444;">Query failed: ${err.message}</span>`;
      } finally {
        btnSubmitQuery.disabled = false;
        btnSubmitQuery.innerHTML = `<span>Execute Verified Query</span>`;
      }
    }
  });

  function handleSseEvent(event, data) {
    if (event === "routing") {
      strategyBadge.innerText = `Strategy: ${data.strategy.toUpperCase()}`;
    } else if (event === "subgraph") {
      subgraphCount.innerText = `${data.nodes.length} nodes`;
      renderMiniGraph(data);
    } else if (event === "sources") {
      sourcesCount.innerText = `${data.length} passages`;
      renderSources(data);
    } else if (event === "token") {
      if (responseOutput.querySelector("span")) {
        responseOutput.innerHTML = "";
      }
      responseOutput.innerText += data.token;
    } else if (event === "verification") {
      const faithPct = (data.faithfulness_score * 100).toFixed(1);
      faithfulnessIndicator.innerText = `${faithPct}%`;
      faithfulnessBar.style.width = `${faithPct}%`;

      let html = "";
      data.claims.forEach((c) => {
        html += `
          <div class="claim-item ${c.status}">
            <strong>${c.status}</strong> (${(c.confidence * 100).toFixed(0)}% conf): ${c.claim_text}
          </div>
        `;
      });
      claimsSummary.innerHTML = html;
    }
  }

  function renderCompleteQueryResponse(data) {
    strategyBadge.innerText = `Strategy: ${data.strategy.toUpperCase()}`;
    responseOutput.innerText = data.response;

    const faithPct = (data.faithfulness_score * 100).toFixed(1);
    faithfulnessIndicator.innerText = `${faithPct}%`;
    faithfulnessBar.style.width = `${faithPct}%`;

    // Render claims
    let claimHtml = "";
    (data.claims || []).forEach((c) => {
      claimHtml += `
        <div class="claim-item ${c.status}">
          <strong>${c.status}</strong> (${(c.confidence * 100).toFixed(0)}% conf): ${c.claim_text}
        </div>
      `;
    });
    claimsSummary.innerHTML = claimHtml;

    // Render sources
    sourcesCount.innerText = `${(data.retrieved_chunks || []).length} passages`;
    renderSources(data.retrieved_chunks || []);

    // Render mini subgraph
    if (data.subgraph && data.subgraph.nodes) {
      subgraphCount.innerText = `${data.subgraph.nodes.length} nodes`;
      renderMiniGraph(data.subgraph);
    }
  }

  function renderSources(chunks) {
    if (!chunks || chunks.length === 0) {
      sourcesList.innerHTML = '<p class="placeholder-text">No relevant passages found.</p>';
      return;
    }

    let html = "";
    chunks.forEach((c, idx) => {
      const bread = (c.heading_breadcrumbs || []).join(" > ") || "Document Root";
      html += `
        <div class="source-item">
          <div class="s-title">[#${idx + 1}] ${c.document_title}</div>
          <div class="s-bread">${bread} (Score: ${(c.score || 0).toFixed(3)})</div>
          <div class="s-excerpt">${(c.content || "").substring(0, 180)}...</div>
        </div>
      `;
    });
    sourcesList.innerHTML = html;
  }

  function renderMiniGraph(subgraph) {
    if (!subgraph.nodes || subgraph.nodes.length === 0) {
      miniGraphCanvas.innerHTML = '<div class="empty-graph-state">No graph relations detected.</div>';
      return;
    }

    let nodesHtml = subgraph.nodes
      .slice(0, 8)
      .map(
        (n) => `
        <span style="display:inline-block; margin: 3px; padding: 4px 8px; border-radius: 6px; font-size: 0.75rem; background: rgba(6,182,212,0.15); border: 1px solid #06b6d4; color: #38bdf8;">
          ${n.name}
        </span>
      `
      )
      .join("");

    let edgesHtml = (subgraph.edges || [])
      .slice(0, 4)
      .map(
        (e) => `
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">
          • ${e.source} ➔ <span style="color:#06b6d4;">[${e.relation}]</span> ➔ ${e.target}
        </div>
      `
      )
      .join("");

    miniGraphCanvas.innerHTML = `
      <div style="padding: 10px; width: 100%;">
        <div style="margin-bottom: 6px;">${nodesHtml}</div>
        <div style="border-top: 1px solid rgba(255,255,255,0.06); padding-top: 6px;">${edgesHtml}</div>
      </div>
    `;
  }

  // Fetch Cluster & Graph Stats
  async function fetchClusterStats() {
    try {
      const resp = await fetch("/health");
      const data = await resp.json();
      document.getElementById("statDocs").innerText = data.documents_indexed || 0;
      document.getElementById("statNodes").innerText = data.graph_nodes || 0;
      document.getElementById("statEdges").innerText = data.graph_edges || 0;
    } catch (e) {
      console.warn("Failed to fetch cluster stats", e);
    }
  }

  // Knowledge Graph Tab Handling
  async function fetchGraphData() {
    try {
      const [subgraphResp, statsResp] = await Promise.all([
        fetch("/api/v1/graph/subgraph?max_nodes=40"),
        fetch("/api/v1/graph/stats"),
      ]);

      const subgraph = await subgraphResp.json();
      const stats = await statsResp.json();

      if (mainGraphViz) {
        mainGraphViz.setData(subgraph);
      }

      // Populate Central Entities Table
      const tbody = document.querySelector("#centralEntitiesTable tbody");
      if (stats.central_entities && stats.central_entities.length > 0) {
        tbody.innerHTML = stats.central_entities
          .map(
            (e) => `
            <tr>
              <td><strong>${e.name}</strong></td>
              <td><span class="badge-subtle">${e.type}</span></td>
              <td><code>${e.score}</code></td>
            </tr>
          `
          )
          .join("");
      } else {
        tbody.innerHTML = '<tr><td colspan="3" class="text-center">No entities indexed yet.</td></tr>';
      }
    } catch (err) {
      console.warn("Failed to load graph data", err);
    }
  }

  document.getElementById("btnRefreshGraph").addEventListener("click", fetchGraphData);
  document.getElementById("btnZoomFit").addEventListener("click", () => {
    if (mainGraphViz) mainGraphViz.resetView();
  });

  // Document Ingestion Tab Handling
  async function fetchDocumentsList() {
    try {
      const resp = await fetch("/api/v1/documents");
      const data = await resp.json();
      const docs = data.documents || [];
      document.getElementById("docCountBadge").innerText = `${docs.length} documents`;

      const tbody = document.querySelector("#documentsTable tbody");
      if (docs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center">No documents ingested.</td></tr>';
        return;
      }

      tbody.innerHTML = docs
        .map(
          (d) => `
          <tr>
            <td><strong>${d.title}</strong></td>
            <td><span class="badge-subtle">${d.mime_type.split("/")[1] || "txt"}</span></td>
            <td>${d.chunk_count} chunks</td>
            <td>${(d.file_size_bytes / 1024).toFixed(1)} KB</td>
            <td>
              <button class="action-btn" onclick="inspectDocumentChunks('${d.id}', '${escapeHtml(d.title)}')">View Chunks</button>
              <button class="action-btn delete" onclick="deleteDocument('${d.id}')">Delete</button>
            </td>
          </tr>
        `
        )
        .join("");
    } catch (e) {
      console.warn("Failed to fetch documents", e);
    }
  }

  // Drag & drop file upload
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      uploadFile(fileInput.files[0]);
    }
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  async function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const resp = await fetch("/api/v1/documents/upload", {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) throw new Error("Upload failed");
      fetchDocumentsList();
      fetchClusterStats();
      alert(`Document "${file.name}" successfully indexed!`);
    } catch (e) {
      alert(`Upload error: ${e.message}`);
    }
  }

  // Raw text ingest form
  const textIngestForm = document.getElementById("textIngestForm");
  textIngestForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const title = document.getElementById("textTitle").value.trim();
    const content = document.getElementById("textContent").value.trim();
    if (!title || !content) return;

    try {
      const resp = await fetch("/api/v1/documents/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, content }),
      });
      if (!resp.ok) throw new Error("Ingestion failed");
      textIngestForm.reset();
      fetchDocumentsList();
      fetchClusterStats();
      alert(`Text document "${title}" indexed!`);
    } catch (e) {
      alert(`Ingestion error: ${e.message}`);
    }
  });

  // Global window functions for document actions
  window.deleteDocument = async function (docId) {
    if (!confirm("Are you sure you want to delete this document?")) return;
    try {
      await fetch(`/api/v1/documents/${docId}`, { method: "DELETE" });
      fetchDocumentsList();
      fetchClusterStats();
    } catch (e) {
      alert(`Delete error: ${e.message}`);
    }
  };

  window.inspectDocumentChunks = async function (docId, docTitle) {
    const inspector = document.getElementById("chunkInspector");
    const titleEl = document.getElementById("inspectorTitle");
    const listEl = document.getElementById("inspectorChunksList");

    inspector.style.display = "block";
    titleEl.innerText = `Chunks for: ${docTitle}`;
    listEl.innerHTML = "<p>Loading chunks...</p>";

    try {
      const resp = await fetch(`/api/v1/documents/${docId}/chunks`);
      const chunks = await resp.json();

      listEl.innerHTML = chunks
        .map(
          (c) => `
          <div class="chunk-card">
            <div class="c-bread">Chunk #${c.chunk_index} | ${c.heading_breadcrumbs.join(" > ") || "Root"}</div>
            <div class="c-body">${escapeHtml(c.content)}</div>
          </div>
        `
        )
        .join("");
    } catch (e) {
      listEl.innerHTML = `<p style="color: #ef4444;">Failed to load chunks: ${e.message}</p>`;
    }
  };

  document.getElementById("btnCloseInspector").addEventListener("click", () => {
    document.getElementById("chunkInspector").style.display = "none";
  });

  // Benchmark Tab Handling
  const btnRunBenchmark = document.getElementById("btnRunBenchmark");
  btnRunBenchmark.addEventListener("click", async () => {
    btnRunBenchmark.disabled = true;
    btnRunBenchmark.innerHTML = "Executing Benchmarks...";

    try {
      const resp = await fetch("/api/v1/benchmark/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ num_samples: 5 }),
      });
      const data = await resp.json();

      // Update UI with empirical results
      document.getElementById("bmNaiveFaith").innerText = `${(data.naive_rag.faithfulness * 100).toFixed(1)}%`;
      document.getElementById("bmNaiveRel").innerText = `${(data.naive_rag.answer_relevance * 100).toFixed(1)}%`;
      document.getElementById("bmNaiveRecall").innerText = `${(data.naive_rag.multi_hop_recall * 100).toFixed(1)}%`;
      document.getElementById("bmNaiveLat").innerText = `${data.naive_rag.avg_latency_ms} ms`;

      document.getElementById("bmHybridFaith").innerText = `${(data.hybrid_rag.faithfulness * 100).toFixed(1)}%`;
      document.getElementById("bmHybridRel").innerText = `${(data.hybrid_rag.answer_relevance * 100).toFixed(1)}%`;
      document.getElementById("bmHybridRecall").innerText = `${(data.hybrid_rag.multi_hop_recall * 100).toFixed(1)}%`;
      document.getElementById("bmHybridLat").innerText = `${data.hybrid_rag.avg_latency_ms} ms`;

      document.getElementById("bmVG_Faith").innerText = `${(data.verigraph.faithfulness * 100).toFixed(1)}%`;
      document.getElementById("bmVG_Rel").innerText = `${(data.verigraph.answer_relevance * 100).toFixed(1)}%`;
      document.getElementById("bmVG_Recall").innerText = `${(data.verigraph.multi_hop_recall * 100).toFixed(1)}%`;
      document.getElementById("bmVG_Lat").innerText = `${data.verigraph.avg_latency_ms} ms`;

      document.getElementById("impFaith").innerText = `+${data.improvements.faithfulness_gain_percent}%`;
      document.getElementById("impRecall").innerText = `+${data.improvements.multi_hop_recall_gain_percent}%`;

      alert("Benchmark suite completed successfully!");
    } catch (e) {
      alert(`Benchmark error: ${e.message}`);
    } finally {
      btnRunBenchmark.disabled = false;
      btnRunBenchmark.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
        Execute Benchmark Suite
      `;
    }
  });

  function escapeHtml(str) {
    return (str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }
});

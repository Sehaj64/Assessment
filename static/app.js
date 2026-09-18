document.addEventListener("DOMContentLoaded", () => {
  // Global state
  let currentAnomalies = [];

  // DOM Elements
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  const queryForm = document.getElementById("query-form");
  const queryInput = document.getElementById("query-input");
  const queryLoading = document.getElementById("query-loading");
  const queryResultCard = document.getElementById("query-result-card");
  const chips = document.querySelectorAll(".chip");

  // KPI elements
  const kpiTotal = document.getElementById("kpi-total");
  const kpiUnresolved = document.getElementById("kpi-unresolved");
  const kpiResTime = document.getElementById("kpi-res-time");
  const kpiRating = document.getElementById("kpi-rating");
  const kpiAnomalies = document.getElementById("kpi-anomalies");
  const providerLabel = document.getElementById("provider-label");

  // Result elements
  const resAnswer = document.getElementById("res-answer");
  const resSql = document.getElementById("res-sql");
  const resBadgeProvider = document.getElementById("res-badge-provider");
  const resBadgeTime = document.getElementById("res-badge-time");
  const resBadgeRows = document.getElementById("res-badge-rows");
  const resTableWrap = document.getElementById("res-table-wrap");
  const resTable = document.getElementById("res-table");
  const btnCopySql = document.getElementById("btn-copy-sql");

  // Anomaly elements
  const anomalyList = document.getElementById("anomaly-list");
  const anomalyFilterChips = document.querySelectorAll(".filter-chip");
  const anomalyFilterCount = document.getElementById("anomaly-filter-count");

  // Ticket Explorer elements
  const ticketsTable = document.getElementById("tickets-table");
  const filterCategory = document.getElementById("filter-category");
  const filterPriority = document.getElementById("filter-priority");
  const filterStatus = document.getElementById("filter-status");
  const btnRefreshTickets = document.getElementById("btn-refresh-tickets");

  // TAB SWITCHING
  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-tab");
      tabButtons.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      const targetElem = document.getElementById(target);
      if (targetElem) targetElem.classList.add("active");

      if (target === "tab-tickets" && ticketsTable.querySelector("tbody").children.length === 0) {
        fetchTickets();
      }
    });
  });

  // INITIAL LOAD
  fetchHealth();
  fetchStats();
  fetchAnomalies();

  // HEALTH CHECK
  async function fetchHealth() {
    try {
      const res = await fetch("/health");
      const data = await res.json();
      if (data.llm_configuration) {
        const prov = data.llm_configuration.provider.toUpperCase();
        const model = data.llm_configuration.model;
        providerLabel.textContent = `${prov} (${model})`;
      }
    } catch (e) {
      providerLabel.textContent = "Offline Mode";
    }
  }

  // FETCH STATS FOR KPIS
  async function fetchStats() {
    try {
      const res = await fetch("/stats");
      const data = await res.json();
      kpiTotal.textContent = data.total_tickets.toLocaleString();
      kpiUnresolved.textContent = data.unresolved_count.toLocaleString();
      kpiResTime.textContent = `${data.avg_resolution_time_hrs} hrs`;
      kpiRating.textContent = `${data.avg_customer_rating} / 5`;
    } catch (e) {
      console.error("Failed to load stats", e);
    }
  }

  // FETCH ANOMALIES
  async function fetchAnomalies() {
    try {
      const res = await fetch("/anomalies?limit=200");
      const data = await res.json();
      currentAnomalies = data.anomalies || [];
      kpiAnomalies.textContent = data.total_anomalies_flagged;
      renderAnomalies("ALL");
    } catch (e) {
      anomalyList.innerHTML = `<div class="error-msg">Failed to load anomalies: ${e.message}</div>`;
    }
  }

  // RENDER ANOMALIES WITH FILTER
  function renderAnomalies(filterSev) {
    let filtered = currentAnomalies;
    if (filterSev !== "ALL") {
      filtered = currentAnomalies.filter(a => a.severity === filterSev);
    }

    anomalyFilterCount.textContent = `Showing ${filtered.length} of ${currentAnomalies.length} flagged anomalies`;

    if (filtered.length === 0) {
      anomalyList.innerHTML = `<div class="empty-state">No anomalies found for severity ${filterSev}.</div>`;
      return;
    }

    anomalyList.innerHTML = filtered.map(a => `
      <div class="anomaly-card">
        <div>
          <div class="anomaly-card-header">
            <span class="anomaly-tid">${a.ticket_id}</span>
            <span class="sev-badge sev-${a.severity}">${a.severity}</span>
          </div>
          <div class="anomaly-meta">
            <span><strong>Cat:</strong> ${a.category}</span>
            <span><strong>Priority:</strong> ${a.priority}</span>
            <span><strong>Agent:</strong> ${a.agent_id}</span>
            <span><strong>Type:</strong> ${a.anomaly_type}</span>
          </div>
          <p class="anomaly-reason">${a.reason}</p>
        </div>
        <div class="anomaly-action">
          <span class="action-tag">Action:</span> ${a.recommended_action}
        </div>
      </div>
    `).join("");
  }

  // ANOMALY FILTER BUTTONS
  anomalyFilterChips.forEach(chip => {
    chip.addEventListener("click", () => {
      anomalyFilterChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      const sev = chip.getAttribute("data-sev");
      renderAnomalies(sev);
    });
  });

  // PRESET QUERY CHIPS
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-q");
      queryInput.value = q;
      queryForm.dispatchEvent(new Event("submit"));
    });
  });

  // SUBMIT QUERY
  queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    queryLoading.classList.remove("hidden");
    queryResultCard.classList.add("hidden");

    try {
      const res = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query })
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Query failed");
      }

      const data = await res.json();
      displayQueryResult(data);
    } catch (err) {
      alert(`Query Error: ${err.message}`);
    } finally {
      queryLoading.classList.add("hidden");
    }
  });

  // DISPLAY QUERY RESULT
  function displayQueryResult(result) {
    resAnswer.innerHTML = formatMarkdown(result.answer);
    resSql.textContent = result.sql;
    resBadgeProvider.textContent = `Provider: ${result.llm_provider}`;
    resBadgeTime.textContent = `⚡ ${result.execution_time_ms} ms`;
    resBadgeRows.textContent = `${result.row_count} ${result.row_count === 1 ? "record" : "records"}`;

    // Populate data table
    const thead = resTable.querySelector("thead");
    const tbody = resTable.querySelector("tbody");
    thead.innerHTML = "";
    tbody.innerHTML = "";

    if (result.data && result.data.length > 0) {
      resTableWrap.classList.remove("hidden");
      const columns = Object.keys(result.data[0]);

      // Header
      const trHead = document.createElement("tr");
      columns.forEach(col => {
        const th = document.createElement("th");
        th.textContent = col.replace(/_/g, " ").toUpperCase();
        trHead.appendChild(th);
      });
      thead.appendChild(trHead);

      // Rows
      result.data.slice(0, 30).forEach(row => {
        const tr = document.createElement("tr");
        columns.forEach(col => {
          const td = document.createElement("td");
          td.textContent = row[col] !== null ? row[col] : "—";
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
    } else {
      resTableWrap.classList.add("hidden");
    }

    queryResultCard.classList.remove("hidden");
    queryResultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // COPY SQL
  btnCopySql.addEventListener("click", () => {
    navigator.clipboard.writeText(resSql.textContent);
    btnCopySql.textContent = "Copied!";
    setTimeout(() => { btnCopySql.textContent = "Copy"; }, 1800);
  });

  // FETCH TICKETS FOR TAB 3
  async function fetchTickets() {
    const cat = filterCategory.value;
    const prio = filterPriority.value;
    const stat = filterStatus.value;

    const params = new URLSearchParams({ limit: "50" });
    if (cat) params.append("category", cat);
    if (prio) params.append("priority", prio);
    if (stat) params.append("status", stat);

    try {
      const res = await fetch(`/tickets?${params.toString()}`);
      const data = await res.json();
      const tbody = ticketsTable.querySelector("tbody");
      tbody.innerHTML = "";

      data.tickets.forEach(t => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${t.ticket_id}</strong></td>
          <td>${t.created_at}</td>
          <td>${t.category}</td>
          <td><span class="badge ${getPriorityBadgeClass(t.priority)}">${t.priority}</span></td>
          <td><span class="badge ${getStatusBadgeClass(t.status)}">${t.status}</span></td>
          <td>${t.response_time_hrs}h</td>
          <td>${t.resolution_time_hrs !== null ? t.resolution_time_hrs + "h" : "—"}</td>
          <td>${t.agent_id}</td>
          <td>${t.customer_rating ? t.customer_rating + " ★" : "—"}</td>
          <td>${escapeHtml(t.issue_summary)}</td>
        `;
        tbody.appendChild(tr);
      });
    } catch (e) {
      console.error("Error fetching tickets:", e);
    }
  }

  [filterCategory, filterPriority, filterStatus].forEach(select => {
    select.addEventListener("change", fetchTickets);
  });
  btnRefreshTickets.addEventListener("click", fetchTickets);

  function getPriorityBadgeClass(p) {
    if (p === "Critical") return "badge-gray";
    if (p === "High") return "badge-indigo";
    return "badge-gray";
  }

  function getStatusBadgeClass(s) {
    if (s === "Resolved") return "badge-emerald";
    if (s === "Escalated") return "badge-gray";
    return "badge-indigo";
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function formatMarkdown(text) {
    if (!text) return "";
    let html = text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/\n\n/g, "<br><br>")
      .replace(/\n- (.*?)/g, "<li>$1</li>");
    if (html.includes("<li>")) {
      html = html.replace(/(<li>.*?<\/li>)+/g, "<ul>$&</ul>");
    }
    return html;
  }
});

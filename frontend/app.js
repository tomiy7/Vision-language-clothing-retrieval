const API_BASE = "http://localhost:8000";

  let mode = "text";
  let selectedFile = null;

  const tabText = document.getElementById("tabText");
  const tabImage = document.getElementById("tabImage");
  const textPanel = document.getElementById("textPanel");
  const imagePanel = document.getElementById("imagePanel");
  const queryText = document.getElementById("queryText");
  const dropZone = document.getElementById("dropZone");
  const imageInput = document.getElementById("imageInput");
  const dropZoneEmpty = document.getElementById("dropZoneEmpty");
  const dropZonePreview = document.getElementById("dropZonePreview");
  const previewImg = document.getElementById("previewImg");
  const previewName = document.getElementById("previewName");
  const clearImage = document.getElementById("clearImage");
  const topK = document.getElementById("topK");
  const searchBtn = document.getElementById("searchBtn");
  const resultsSection = document.getElementById("resultsSection");
  const resultsTitle = document.getElementById("resultsTitle");
  const resultsCount = document.getElementById("resultsCount");
  const resultsBody = document.getElementById("resultsBody");

  function setMode(newMode) {
    mode = newMode;
    const isText = mode === "text";
    tabText.classList.toggle("active", isText);
    tabImage.classList.toggle("active", !isText);
    textPanel.style.display = isText ? "block" : "none";
    imagePanel.style.display = isText ? "none" : "block";

    // Rezultati od prethodnog moda (tekst ili slika) se sklanjaju
    // čim se pređe na drugi mod, da ne ostanu vidljivi "stari"
    // rezultati koji ne odgovaraju trenutnom modu pretrage.
    resultsSection.style.display = "none";
    resultsBody.innerHTML = "";
  }

  tabText.addEventListener("click", () => setMode("text"));
  tabImage.addEventListener("click", () => setMode("image"));

  dropZone.addEventListener("click", (e) => {
    if (e.target.id !== "clearImage") imageInput.click();
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  imageInput.addEventListener("change", () => {
    if (imageInput.files.length) handleFile(imageInput.files[0]);
  });

  function handleFile(file) {
    selectedFile = file;
    previewImg.src = URL.createObjectURL(file);
    previewName.textContent = file.name;
    dropZoneEmpty.style.display = "none";
    dropZonePreview.style.display = "flex";
  }

  clearImage.addEventListener("click", (e) => {
    e.stopPropagation();
    selectedFile = null;
    imageInput.value = "";
    dropZoneEmpty.style.display = "block";
    dropZonePreview.style.display = "none";
  });

  searchBtn.addEventListener("click", runSearch);

  async function runSearch() {
    const k = Math.max(1, parseInt(topK.value, 10) || 5);

    if (mode === "text" && !queryText.value.trim()) {
      showError("Add a description first", "Type what the piece looks like, then search.");
      return;
    }
    if (mode === "image" && !selectedFile) {
      showError("Add a photo first", "Drag a photo into the drop zone, or click it to browse.");
      return;
    }

    const formData = new FormData();
    formData.append("query_type", mode);
    formData.append("top_k", String(k));
    if (mode === "text") {
      formData.append("text", queryText.value.trim());
    } else {
      formData.append("image", selectedFile);
    }

    setLoading(k);

    try {
      const res = await fetch(`${API_BASE}/search`, { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        showError(`Request failed (${res.status})`, data.detail || "The server rejected this search.");
        return;
      }

      renderResults(data);
    } catch (err) {
      showError("Couldn't reach the server", `Check that the API is running at ${API_BASE}.`);
    }
  }

  function setLoading(k) {
    searchBtn.disabled = true;
    searchBtn.textContent = "Searching…";
    resultsSection.style.display = "block";
    resultsTitle.textContent = "Results";
    resultsCount.textContent = "";

    if (mode === "text") {
      resultsBody.innerHTML = `<div class="skeleton-grid">${
        Array.from({length: k}, () => '<div class="skeleton-card"></div>').join("")
      }</div>`;
    } else {
      resultsBody.innerHTML = Array.from({length: k}, () =>
        '<div class="skeleton-card" style="aspect-ratio:auto;height:64px;margin-bottom:16px;"></div>'
      ).join("");
    }
  }

  function showError(title, detail) {
    searchBtn.disabled = false;
    searchBtn.textContent = "Search";
    resultsSection.style.display = "block";
    resultsTitle.textContent = "Results";
    resultsCount.textContent = "";
    resultsBody.innerHTML = `
      <div class="error-state">
        <div class="error-title">${escapeHtml(title)}</div>
        <div class="error-detail">${escapeHtml(detail)}</div>
      </div>
    `;
  }

  function renderResults(data) {
    searchBtn.disabled = false;
    searchBtn.textContent = "Search";

    const results = data.results || [];
    resultsCount.textContent = results.length ? `${results.length} found` : "";

    if (!results.length) {
      resultsTitle.textContent = "Results";
      resultsBody.innerHTML = `
        <div class="empty-state">
          <p>Nothing matched closely enough. Try a different description or photo.</p>
        </div>
      `;
      return;
    }

    if (results[0].result_type === "image") {
      resultsTitle.textContent = "Closest images";
      resultsBody.innerHTML = `
        <div class="image-grid">
          ${results.map((r, i) => `
            <div class="result-card">
              <div class="frame">
                <img src="${API_BASE}${r.content}" alt="${escapeHtml(r.sample_id)}" loading="lazy" />
                <div class="score-tag">${(r.score * 100).toFixed(0)}%</div>
              </div>
              <div class="meta"><span class="rank">${i + 1}</span>${escapeHtml(r.sample_id)}</div>
            </div>
          `).join("")}
        </div>
      `;
    } else {
      resultsTitle.textContent = "Closest descriptions";
      resultsBody.innerHTML = `
        <div class="text-results">
          ${results.map((r, i) => `
            <div class="text-result">
              <div class="rank">${i + 1}</div>
              <blockquote>
                "${escapeHtml(r.content)}"
                <span class="sample-id">${escapeHtml(r.sample_id)}</span>
              </blockquote>
              <div class="score">${(r.score * 100).toFixed(0)}%</div>
            </div>
          `).join("")}
        </div>
      `;
    }
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str == null ? "" : String(str);
    return div.innerHTML;
  }

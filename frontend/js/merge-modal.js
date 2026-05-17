// Multi-CSV merge modal.

const MergeModal = (() => {
  const backdrop = document.getElementById('mergeModal');
  const body = document.getElementById('mergeModalBody');
  document.getElementById('mergeModalClose').addEventListener('click', close);
  backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });

  let ctx = { sid: null, profile: null, onApplied: null, otherInfo: null };

  function close() { backdrop.hidden = true; ctx.otherInfo = null; }

  function open({ sid, profile, onApplied }) {
    ctx = { sid, profile, onApplied, otherInfo: null };
    renderUpload();
    backdrop.hidden = false;
  }

  function renderUpload() {
    const mainCols = ctx.profile.columns.map(c => c.name);
    body.innerHTML = `
      <div class="modal-section">
        <h5>Main dataset</h5>
        <div class="muted small">${ctx.profile.rows.toLocaleString()} rows × ${ctx.profile.cols} cols. Columns:</div>
        <div class="col-chips" style="margin-top:6px">
          ${mainCols.map(c => `<span class="chip" style="cursor:default">${escapeHtml(c)}</span>`).join('')}
        </div>
      </div>
      <div class="modal-section">
        <h5>Step 1 — load the 2nd CSV</h5>
        <div class="op-picker">
          <label class="full">Upload file
            <input id="mgFile" type="file" accept=".csv" />
          </label>
          <label class="full">…or paste URL
            <input id="mgUrl" type="url" placeholder="https://example.com/lookup.csv" />
          </label>
          <button class="btn primary full" id="mgInspect">Load &amp; inspect</button>
        </div>
        <div id="mgInspectErr" class="error" style="margin-top:10px" hidden></div>
      </div>
    `;

    document.getElementById('mgInspect').addEventListener('click', loadOther);
    document.getElementById('mgUrl').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); loadOther(); }
    });
  }

  async function loadOther() {
    const file = document.getElementById('mgFile').files?.[0];
    const url = document.getElementById('mgUrl').value.trim();
    const errBox = document.getElementById('mgInspectErr');
    errBox.hidden = true;

    if (!file && !url) {
      errBox.textContent = 'Provide a file or URL.';
      errBox.hidden = false;
      return;
    }

    const btn = document.getElementById('mgInspect');
    btn.disabled = true; btn.innerHTML = '<span class="loading"></span> Loading…';

    const fd = new FormData();
    if (file) fd.append('file', file);
    else fd.append('url', url);

    try {
      const r = await fetch('/api/merge/inspect', { method: 'POST', body: fd });
      if (!r.ok) throw new Error((await r.json()).error || 'failed');
      ctx.otherInfo = await r.json();
      ctx.otherSource = { file, url };
      renderJoinPicker();
    } catch (e) {
      errBox.textContent = e.message;
      errBox.hidden = false;
      btn.disabled = false; btn.innerHTML = 'Load &amp; inspect';
    }
  }

  function renderJoinPicker() {
    const mainCols = ctx.profile.columns.map(c => c.name);
    const other = ctx.otherInfo;

    // suggest a join key — first column with the same name on both sides
    const suggested = mainCols.find(c => other.columns.includes(c)) || mainCols[0];

    body.innerHTML = `
      <div class="modal-section">
        <h5>Step 2 — second dataset loaded</h5>
        <div class="muted small">
          <b>${escapeHtml(other.source)}</b> — ${other.rows.toLocaleString()} rows × ${other.cols} cols
        </div>
        <div class="col-chips" style="margin-top:6px">
          ${other.columns.map(c => `<span class="chip" style="cursor:default">${escapeHtml(c)}</span>`).join('')}
        </div>
      </div>
      <div class="modal-section">
        <h5>Step 3 — pick join keys + merge type</h5>
        <div class="op-picker">
          <label>Main dataset key (left_on)
            <select id="mgLeft">${mainCols.map(c =>
              `<option value="${escAttr(c)}" ${c === suggested ? 'selected' : ''}>${escapeHtml(c)}</option>`
            ).join('')}</select>
          </label>
          <label>Other dataset key (right_on)
            <select id="mgRight">${other.columns.map(c =>
              `<option value="${escAttr(c)}" ${c === suggested ? 'selected' : ''}>${escapeHtml(c)}</option>`
            ).join('')}</select>
          </label>
          <label class="full">Merge type
            <select id="mgHow">
              <option value="inner">Inner — keep only matching rows (default)</option>
              <option value="left">Left — keep all main rows, fill missing from other</option>
              <option value="right">Right — keep all other rows</option>
              <option value="outer">Outer — keep all rows from both</option>
            </select>
          </label>
        </div>
      </div>
      <div class="modal-section">
        <h5>Preview of 2nd dataset (first 8 rows)</h5>
        <div class="table-wrap small">
          <table class="values-table">
            <thead><tr>${other.columns.map(c => `<th>${escapeHtml(c)}</th>`).join('')}</tr></thead>
            <tbody>
              ${other.preview.slice(0, 8).map(row =>
                `<tr>${other.columns.map(c => {
                  const v = row[c];
                  return `<td>${v === null || v === undefined ? '<span class="muted">∅</span>' : escapeHtml(String(v))}</td>`;
                }).join('')}</tr>`
              ).join('')}
            </tbody>
          </table>
        </div>
      </div>
      <div class="modal-section">
        <div style="display:flex;gap:10px">
          <button class="btn ghost" id="mgBack">← Pick a different file</button>
          <button class="btn primary big drawer-apply" id="mgApply">Merge now</button>
        </div>
      </div>
    `;

    document.getElementById('mgBack').addEventListener('click', renderUpload);
    document.getElementById('mgApply').addEventListener('click', applyMerge);
  }

  async function applyMerge() {
    const fd = new FormData();
    if (ctx.otherSource.file) fd.append('file', ctx.otherSource.file);
    else fd.append('url', ctx.otherSource.url);
    fd.append('left_on', document.getElementById('mgLeft').value);
    fd.append('right_on', document.getElementById('mgRight').value);
    fd.append('how', document.getElementById('mgHow').value);

    const btn = document.getElementById('mgApply');
    const orig = btn.innerHTML;
    btn.disabled = true; btn.innerHTML = '<span class="loading"></span> Merging…';
    try {
      const r = await fetch(`/api/merge/${ctx.sid}`, { method: 'POST', body: fd });
      if (!r.ok) throw new Error((await r.json()).error || 'merge failed');
      const res = await r.json();
      showToast(res.message, 'success');
      close();
      ctx.onApplied();
    } catch (e) {
      showToast(e.message, 'error');
      btn.disabled = false; btn.innerHTML = orig;
    }
  }

  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
  function escAttr(s) { return escapeHtml(s); }

  return { open, close };
})();

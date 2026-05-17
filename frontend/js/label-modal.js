// Inconsistent-label normalizer modal.

const LabelModal = (() => {
  const backdrop = document.getElementById('labelModal');
  const sub = document.getElementById('labelModalSub');
  const body = document.getElementById('labelModalBody');
  document.getElementById('labelModalClose').addEventListener('click', close);
  backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });

  function close() { backdrop.hidden = true; }

  async function open({ sid, column, onApplied }) {
    sub.textContent = `Column: ${column}`;
    body.innerHTML = '<div class="muted" style="text-align:center;padding:40px"><span class="loading"></span> Detecting label clusters…</div>';
    backdrop.hidden = false;

    const threshold = 0.85;
    let data;
    try {
      data = await API.labelGroups(sid, column, threshold);
    } catch (e) {
      body.innerHTML = `<div class="error">${escapeHtml(e.message)}</div>`;
      return;
    }

    if (!data.groups || !data.groups.length) {
      body.innerHTML = `
        <div class="muted" style="text-align:center;padding:40px">
          ✓ No inconsistent label clusters detected in '${escapeHtml(column)}'.<br>
          (Threshold ${threshold}. Lower it to find more, e.g. 0.7.)
        </div>
        <div style="text-align:center;margin-top:12px">
          <button class="btn" id="rescanBtn">Re-scan with lower threshold</button>
        </div>`;
      document.getElementById('rescanBtn').addEventListener('click', () => {
        const v = parseFloat(prompt('Similarity threshold (0.5 to 1.0)?', '0.7'));
        if (!Number.isNaN(v)) openInternal(sid, column, v, onApplied);
      });
      return;
    }

    render(sid, column, data, onApplied);
  }

  async function openInternal(sid, column, threshold, onApplied) {
    body.innerHTML = '<div class="muted" style="text-align:center;padding:40px"><span class="loading"></span> Detecting…</div>';
    const data = await API.labelGroups(sid, column, threshold);
    if (!data.groups || !data.groups.length) {
      body.innerHTML = `<div class="muted" style="text-align:center;padding:40px">No clusters at threshold ${threshold}.</div>`;
      return;
    }
    render(sid, column, data, onApplied);
  }

  function render(sid, column, data, onApplied) {
    body.innerHTML = `
      <div class="modal-section">
        <h5>Threshold</h5>
        <div class="op-picker">
          <label>Similarity (lower = more aggressive grouping)
            <input id="lmThresh" type="number" min="0.5" max="1" step="0.05" value="${data.threshold}" />
          </label>
          <button class="btn" id="lmRescan">Re-scan</button>
        </div>
      </div>
      <div class="modal-section">
        <h5>Detected groups — edit canonical or uncheck to skip</h5>
        <div id="lmGroups">${data.groups.map((g, gi) => groupHtml(g, gi)).join('')}</div>
      </div>
      <div class="modal-section">
        <button class="btn primary big drawer-apply" id="lmApply">Apply mapping to '${escapeHtml(column)}'</button>
      </div>
    `;

    document.getElementById('lmRescan').addEventListener('click', () => {
      const t = parseFloat(document.getElementById('lmThresh').value);
      openInternal(sid, column, t, onApplied);
    });

    document.getElementById('lmApply').addEventListener('click', async () => {
      const mapping = buildMapping(data.groups);
      if (!Object.keys(mapping).length) {
        showToast('No mappings selected.', 'error');
        return;
      }
      const btn = document.getElementById('lmApply');
      btn.disabled = true; btn.innerHTML = '<span class="loading"></span> Applying…';
      try {
        const res = await API.clean(sid, {
          family: 'label_norm', strategy: 'normalize',
          column, params: { mapping },
        });
        showToast(res.message, 'success');
        close();
        onApplied();
      } catch (e) {
        showToast(e.message, 'error');
        btn.disabled = false; btn.textContent = `Apply mapping to '${column}'`;
      }
    });
  }

  function groupHtml(g, gi) {
    return `
      <div class="label-group" data-gi="${gi}">
        <div class="label-group-head">
          <label class="lg-checkbox">
            <input type="checkbox" class="lg-toggle" data-gi="${gi}" checked />
            <span>Include this group</span>
          </label>
          <label class="lg-canonical">
            Canonical:
            <input type="text" class="lg-canonical-input" data-gi="${gi}" value="${escAttr(g.canonical)}" />
          </label>
          <span class="muted small">(${g.total} rows total)</span>
        </div>
        <div class="label-members">
          ${g.members.map(m => `
            <span class="member-pill">${escapeHtml(m.value)} <span class="muted">×${m.count}</span></span>
          `).join('')}
        </div>
      </div>`;
  }

  function buildMapping(groups) {
    const mapping = {};
    document.querySelectorAll('.lg-toggle').forEach(cb => {
      if (!cb.checked) return;
      const gi = parseInt(cb.dataset.gi, 10);
      const g = groups[gi];
      const canonInput = document.querySelector(`.lg-canonical-input[data-gi="${gi}"]`);
      const canonical = canonInput.value.trim() || g.canonical;
      for (const m of g.members) {
        if (m.value !== canonical) mapping[m.value] = canonical;
      }
    });
    return mapping;
  }

  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
  function escAttr(s) { return escapeHtml(s); }

  return { open, close };
})();

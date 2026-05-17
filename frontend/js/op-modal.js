// Operation preview modal: shows current values, sample of affected rows,
// preview of the result, and a picker to swap the operation before applying.

const OpModal = (() => {
  const backdrop = document.getElementById('opModal');
  const titleEl = document.getElementById('opModalTitle');
  const reasonEl = document.getElementById('opModalReason');
  const bodyEl = document.getElementById('opModalBody');
  document.getElementById('opModalClose').addEventListener('click', close);
  backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });

  function close() { backdrop.hidden = true; }

  async function open({ suggestion, sid, onApplied }) {
    titleEl.textContent = suggestion.title;
    reasonEl.textContent = suggestion.reason;
    bodyEl.innerHTML = '<div class="muted" style="text-align:center;padding:40px"><span class="loading"></span> Loading preview…</div>';
    backdrop.hidden = false;

    // Fetch preview
    let info;
    try {
      info = await API.previewOp(sid, suggestion.op);
    } catch (e) {
      bodyEl.innerHTML = `<div class="error">${escapeHtml(e.message)}</div>`;
      return;
    }

    render(info, suggestion, sid, onApplied);
  }

  function render(info, suggestion, sid, onApplied) {
    const ci = info.column_info;
    const sections = [];

    // Stats strip
    sections.push(`
      <div class="modal-section">
        <div class="modal-stats">
          ${ci ? `
            <div class="modal-stat"><div class="v">${ci.rows.toLocaleString()}</div><div class="k">rows</div></div>
            <div class="modal-stat"><div class="v">${ci.missing.toLocaleString()}</div><div class="k">missing</div></div>
            <div class="modal-stat"><div class="v">${ci.unique.toLocaleString()}</div><div class="k">unique</div></div>
            <div class="modal-stat"><div class="v">${escapeHtml(ci.dtype)}</div><div class="k">dtype</div></div>
          ` : `
            <div class="modal-stat"><div class="v">${(info.rows_before||0).toLocaleString()}</div><div class="k">rows before</div></div>
            <div class="modal-stat"><div class="v">${(info.rows_after||0).toLocaleString()}</div><div class="k">rows after</div></div>
          `}
        </div>
      </div>
    `);

    // Top values
    if (info.top_values && info.top_values.length) {
      sections.push(`
        <div class="modal-section">
          <h5>Top values in '${escapeHtml(ci.name)}' (from original data)</h5>
          <table class="values-table">
            <thead><tr><th>Value</th><th>Count</th></tr></thead>
            <tbody>
              ${info.top_values.map(t => `
                <tr><td>${escapeHtml(t.value)}</td><td class="num">${t.count.toLocaleString()}</td></tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      `);
    }

    // Outlier sample
    if (info.outlier_sample && info.outlier_sample.length) {
      sections.push(`
        <div class="modal-section">
          <h5>Outlier values (IQR bounds: ${fmtNum(info.outlier_bounds.lo)} … ${fmtNum(info.outlier_bounds.hi)})</h5>
          <div class="muted small">Found ${info.outlier_count} total outliers; showing first ${info.outlier_sample.length}.</div>
          <table class="values-table">
            <thead><tr><th>Row</th><th>Value</th></tr></thead>
            <tbody>
              ${info.outlier_sample.map(o => `
                <tr><td class="num">${o.row}</td><td class="num">${fmtNum(o.value)}</td></tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      `);
    }

    // Missing rows sample
    if (info.sample_missing_rows && info.sample_missing_rows.length) {
      const cols = Object.keys(info.sample_missing_rows[0].values);
      sections.push(`
        <div class="modal-section">
          <h5>Rows where this column is missing (sample)</h5>
          <div class="table-wrap small">
            <table class="values-table">
              <thead><tr><th>row</th>${cols.map(c => `<th>${escapeHtml(c)}</th>`).join('')}</tr></thead>
              <tbody>
                ${info.sample_missing_rows.map(r => `
                  <tr><td class="num">${r.row}</td>
                    ${cols.map(c => {
                      const v = r.values[c];
                      return `<td>${v === null || v === undefined ? '<span class="muted">∅</span>' : escapeHtml(String(v))}</td>`;
                    }).join('')}
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>
      `);
    }

    // Added / removed columns
    if (info.added_cols && info.added_cols.length) {
      sections.push(`
        <div class="modal-section">
          <h5>New columns created</h5>
          <div>${info.added_cols.map(c => `<span class="added-pill">+ ${escapeHtml(c)}</span>`).join('')}</div>
        </div>
      `);
    }
    if (info.removed_cols && info.removed_cols.length) {
      sections.push(`
        <div class="modal-section">
          <h5>Columns removed</h5>
          <div>${info.removed_cols.map(c => `<span class="removed-pill">− ${escapeHtml(c)}</span>`).join('')}</div>
        </div>
      `);
    }

    // Result preview
    if (info.result_preview && info.result_preview.length) {
      const cols = Object.keys(info.result_preview[0]);
      sections.push(`
        <div class="modal-section">
          <h5>Result preview after operation (${info.rows_after.toLocaleString()} rows)</h5>
          <div class="table-wrap small">
            <table class="values-table">
              <thead><tr>${cols.map(c => `<th>${escapeHtml(c)}</th>`).join('')}</tr></thead>
              <tbody>
                ${info.result_preview.map(r => `
                  <tr>${cols.map(c => {
                    const v = r[c];
                    return `<td>${v === null || v === undefined ? '<span class="muted">∅</span>' : escapeHtml(String(v))}</td>`;
                  }).join('')}</tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>
      `);
    }

    // Message + code
    if (info.message) {
      sections.push(`
        <div class="modal-section">
          <h5>What this will do</h5>
          <div class="muted">${escapeHtml(info.message)}</div>
        </div>
      `);
    }
    if (info.code) {
      sections.push(`
        <div class="modal-section">
          <h5>Generated code</h5>
          <pre class="code-block">${escapeHtml(info.code)}</pre>
        </div>
      `);
    }

    // Op picker — alternative operation
    sections.push(`
      <div class="modal-section">
        <h5>Use this operation, or pick a different one</h5>
        <div class="op-picker">
          <label>Category
            <select id="omCat"></select>
          </label>
          <label>Strategy
            <select id="omStrat"></select>
          </label>
          <label class="full" id="omParamWrap" hidden>
            <span id="omParamLabel">Parameter</span>
            <input id="omParamInput" type="text" />
          </label>
          <button class="btn primary big full" id="omApply">Apply this operation</button>
        </div>
      </div>
    `);

    bodyEl.innerHTML = sections.join('');

    setupPicker(info, suggestion, sid, onApplied);
  }

  function setupPicker(info, suggestion, sid, onApplied) {
    const catSel = document.getElementById('omCat');
    const stratSel = document.getElementById('omStrat');
    const paramWrap = document.getElementById('omParamWrap');
    const paramLabel = document.getElementById('omParamLabel');
    const paramInput = document.getElementById('omParamInput');
    const applyBtn = document.getElementById('omApply');

    const currentFamily = suggestion.op.family;
    const currentStrategy = suggestion.op.strategy;
    const currentCategory = opCategoryForFamily(currentFamily);
    const ci = info.column_info;
    const column = ci ? ci.name : suggestion.op.column;

    catSel.innerHTML = Object.keys(OP_CATEGORIES).map(c =>
      `<option value="${c}" ${c === currentCategory ? 'selected' : ''}>${c}</option>`
    ).join('');

    function refreshStrats() {
      const cat = OP_CATEGORIES[catSel.value];
      stratSel.innerHTML = cat.strategies.map(([code, label]) =>
        `<option value="${code}" ${code === currentStrategy ? 'selected' : ''}>${label}</option>`
      ).join('');
      refreshParams();
      // NOTE: do not auto-reload preview here — that would re-render the
      // entire modal body and close any dropdown the user just opened.
      // Reload only fires from the explicit change handlers below.
    }
    function refreshParams() {
      const cat = OP_CATEGORIES[catSel.value];
      const strat = cat.strategies.find(s => s[0] === stratSel.value);
      if (!strat) return;
      const [, , kind, def] = strat;
      if (kind) {
        paramWrap.hidden = false;
        paramLabel.textContent =
          kind === 'value' ? 'Fill value' :
          kind === 'number' ? 'Numeric threshold' :
          kind === 'list' ? 'Comma-separated values' : 'Parameter';
        paramInput.type = kind === 'number' ? 'number' : 'text';
        paramInput.value = (suggestion.op.params && (suggestion.op.params.value ?? suggestion.op.params.threshold)) || def || '';
      } else {
        paramWrap.hidden = true;
        paramInput.value = '';
      }
    }
    async function reloadPreview() {
      const op = buildOp();
      if (!op) return;
      // Update body preview sections only (lightweight refresh)
      const newSuggestion = { ...suggestion, op };
      try {
        const newInfo = await API.previewOp(sid, op);
        render(newInfo, newSuggestion, sid, onApplied);
      } catch (e) {
        showToast(e.message, 'error');
      }
    }
    function buildOp() {
      const cat = OP_CATEGORIES[catSel.value];
      const stratDef = cat.strategies.find(s => s[0] === stratSel.value);
      if (!stratDef) return null;
      const kind = stratDef[2];
      const params = {};
      if (kind === 'value') {
        let v = paramInput.value;
        if (stratSel.value === 'target') params.target = v;
        else params.value = v;
      } else if (kind === 'number') {
        params.threshold = parseFloat(paramInput.value);
      } else if (kind === 'list') {
        params.order = paramInput.value.split(',').map(s => s.trim()).filter(Boolean);
      }
      return { family: cat.family, strategy: stratSel.value, column, params };
    }

    catSel.addEventListener('change', () => {
      refreshStrats();
      reloadPreview();
    });
    stratSel.addEventListener('change', () => {
      refreshParams();
      reloadPreview();
    });
    paramInput.addEventListener('change', reloadPreview);

    applyBtn.addEventListener('click', async () => {
      const op = buildOp();
      if (!op) return;
      const orig = applyBtn.innerHTML;
      applyBtn.disabled = true;
      applyBtn.innerHTML = '<span class="loading"></span> Applying…';
      try {
        const res = await API.clean(sid, op);
        showToast(res.message, 'success');
        close();
        onApplied();
      } catch (e) {
        showToast(e.message, 'error');
        applyBtn.disabled = false;
        applyBtn.innerHTML = orig;
      }
    });

    refreshStrats();
  }

  function fmtNum(v) {
    if (v == null) return '—';
    if (typeof v !== 'number') return String(v);
    if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
    return Math.round(v * 1000) / 1000;
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }

  return { open, close };
})();

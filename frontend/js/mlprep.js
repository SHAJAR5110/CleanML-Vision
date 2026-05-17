// ML Preparation modal — 4 tools using one generic modal shell.

const MLPrep = (() => {
  const backdrop = document.getElementById('mlpModal');
  const titleEl = document.getElementById('mlpTitle');
  const subEl = document.getElementById('mlpSub');
  const bodyEl = document.getElementById('mlpBody');
  document.getElementById('mlpClose').addEventListener('click', close);
  backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });

  let ctx = { sid: null, profile: null, onApplied: null };

  function close() { backdrop.hidden = true; }
  function show(title, sub, html) {
    titleEl.textContent = title;
    subEl.textContent = sub;
    bodyEl.innerHTML = html;
    backdrop.hidden = false;
  }

  // ---------- B3.1 Engineer feature ----------
  function openEngineer({ sid, profile, onApplied }) {
    ctx = { sid, profile, onApplied };
    const cols = profile.columns.map(c => c.name);
    const numCols = profile.columns.filter(c => c.inferred_type === 'numeric').map(c => c.name);
    show('🧮 Engineer feature',
      'Create a new column from a formula. Refer to columns by name.',
      `
      <div class="modal-section">
        <h5>New column name</h5>
        <input id="feName" type="text" class="full-width" placeholder="e.g. BMI" />
      </div>
      <div class="modal-section">
        <h5>Formula <span class="muted">— uses column names (case-sensitive)</span></h5>
        <input id="feFormula" type="text" class="full-width" placeholder="weight / (height/100)**2" />
        <div class="muted small" style="margin-top:6px">
          Supported: <code>+ - * / ** % //</code>, comparisons, <code>&amp;</code> / <code>|</code>.
          Examples: <code>price * quantity</code>, <code>age &gt;= 18</code>, <code>(a + b) / 2</code>
        </div>
      </div>
      <div class="modal-section">
        <h5>Available columns (click to insert)</h5>
        <div class="col-chips">${cols.map(c => `<button class="chip" data-col="${escAttr(c)}">${escapeHtml(c)}</button>`).join('')}</div>
      </div>
      <div class="modal-section">
        <button class="btn primary big drawer-apply" id="feApply">Create column</button>
      </div>
    `);

    const nameEl = document.getElementById('feName');
    const formulaEl = document.getElementById('feFormula');
    bodyEl.querySelectorAll('.col-chips .chip').forEach(b => b.addEventListener('click', () => {
      formulaEl.value += (formulaEl.value ? ' ' : '') + b.dataset.col;
      formulaEl.focus();
    }));
    document.getElementById('feApply').addEventListener('click', async () => {
      const name = nameEl.value.trim();
      const formula = formulaEl.value.trim();
      if (!name || !formula) return showToast('Name and formula are required.', 'error');
      const btn = document.getElementById('feApply');
      btn.disabled = true; btn.innerHTML = '<span class="loading"></span> Creating…';
      try {
        const res = await API.clean(sid, {
          family: 'feature_eng', strategy: 'create',
          params: { name, formula },
        });
        showToast(res.message, 'success');
        close(); onApplied();
      } catch (e) {
        showToast(e.message, 'error');
        btn.disabled = false; btn.textContent = 'Create column';
      }
    });
  }

  // ---------- B3.2 Train/test split ----------
  function openSplit({ sid, profile, onApplied }) {
    ctx = { sid, profile, onApplied };
    const cols = profile.columns.map(c => c.name);
    show('✂ Train / Test split',
      'Stratified or random split. Click "Run split" then "Download zip".',
      `
      <div class="modal-section">
        <h5>Target column (for stratification)</h5>
        <select id="splTarget" class="full-width">
          <option value="">— random split, no stratification —</option>
          ${cols.map(c => `<option value="${escAttr(c)}">${escapeHtml(c)}</option>`).join('')}
        </select>
      </div>
      <div class="modal-section">
        <h5>Test size <span class="muted" id="splSizeLbl">20%</span></h5>
        <input id="splSize" type="range" min="0.05" max="0.5" step="0.05" value="0.2" class="full-width" />
      </div>
      <div class="modal-section">
        <h5>Random seed</h5>
        <input id="splSeed" type="number" value="42" class="full-width" />
      </div>
      <div class="modal-section">
        <div style="display:flex;gap:10px">
          <button class="btn primary" id="splRun">Run split</button>
          <button class="btn" id="splDownload" disabled>⬇ Download zip</button>
        </div>
        <div id="splResult" class="muted" style="margin-top:10px"></div>
      </div>
    `);

    const sizeInput = document.getElementById('splSize');
    const sizeLbl = document.getElementById('splSizeLbl');
    sizeInput.addEventListener('input', () => {
      sizeLbl.textContent = `${Math.round(sizeInput.value * 100)}%`;
    });

    document.getElementById('splRun').addEventListener('click', async () => {
      const btn = document.getElementById('splRun');
      const orig = btn.innerHTML;
      btn.disabled = true; btn.innerHTML = '<span class="loading"></span> Splitting…';
      try {
        const res = await fetch(`/api/split/${sid}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            target: document.getElementById('splTarget').value || null,
            test_size: parseFloat(sizeInput.value),
            random_state: parseInt(document.getElementById('splSeed').value, 10),
          }),
        }).then(r => r.ok ? r.json() : r.json().then(d => Promise.reject(new Error(d.error))));
        showToast(res.message, 'success');
        document.getElementById('splResult').textContent =
          `Train rows: ${res.train_rows.toLocaleString()} · Test rows: ${res.test_rows.toLocaleString()}`;
        document.getElementById('splDownload').disabled = false;
        onApplied();
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        btn.disabled = false; btn.innerHTML = orig;
      }
    });
    document.getElementById('splDownload').addEventListener('click', () => {
      window.location.href = `/api/download-split/${sid}`;
    });
  }

  // ---------- B3.3 Balance classes ----------
  function openBalance({ sid, profile, onApplied }) {
    ctx = { sid, profile, onApplied };
    const cols = profile.columns.map(c => c.name);
    show('⚖ Balance classes',
      'Pick the target column and a strategy. Use SMOTE only with all-numeric features.',
      `
      <div class="modal-section">
        <h5>Target column</h5>
        <select id="balTarget" class="full-width">
          ${cols.map(c => `<option value="${escAttr(c)}">${escapeHtml(c)}</option>`).join('')}
        </select>
      </div>
      <div class="modal-section">
        <h5>Strategy</h5>
        <select id="balStrat" class="full-width">
          <option value="oversample">Random oversample (duplicate minority)</option>
          <option value="undersample">Random undersample (shrink majority)</option>
          <option value="smote">SMOTE (synthetic minority — needs all-numeric features)</option>
        </select>
      </div>
      <div class="modal-section">
        <h5>Random seed</h5>
        <input id="balSeed" type="number" value="42" class="full-width" />
      </div>
      <div class="modal-section">
        <button class="btn primary big drawer-apply" id="balApply">Apply balancing</button>
      </div>
    `);

    document.getElementById('balApply').addEventListener('click', async () => {
      const btn = document.getElementById('balApply');
      btn.disabled = true; btn.innerHTML = '<span class="loading"></span> Balancing…';
      try {
        const res = await API.clean(sid, {
          family: 'balance',
          strategy: document.getElementById('balStrat').value,
          params: {
            target: document.getElementById('balTarget').value,
            random_state: parseInt(document.getElementById('balSeed').value, 10),
          },
        });
        showToast(res.message, 'success');
        close(); onApplied();
      } catch (e) {
        showToast(e.message, 'error');
        btn.disabled = false; btn.textContent = 'Apply balancing';
      }
    });
  }

  // ---------- B3.4 Reduce features ----------
  function openReduce({ sid, profile, onApplied }) {
    ctx = { sid, profile, onApplied };
    const cols = profile.columns.map(c => c.name);
    show('📉 Reduce features',
      'PCA for dimensionality reduction, or feature selection by variance / k-best.',
      `
      <div class="modal-section">
        <h5>Method</h5>
        <select id="redMethod" class="full-width">
          <option value="pca">PCA — compress numeric cols into N components</option>
          <option value="variance_threshold">Variance threshold — drop near-constant cols</option>
          <option value="select_k_best">SelectKBest — keep top K most predictive cols</option>
        </select>
      </div>
      <div class="modal-section" id="redParams"></div>
      <div class="modal-section">
        <button class="btn primary big drawer-apply" id="redApply">Apply reduction</button>
      </div>
    `);

    const methodSel = document.getElementById('redMethod');
    const params = document.getElementById('redParams');

    function renderParams() {
      const m = methodSel.value;
      const optsCols = `<option value="">— none —</option>` +
        cols.map(c => `<option value="${escAttr(c)}">${escapeHtml(c)}</option>`).join('');
      if (m === 'pca') {
        params.innerHTML = `
          <h5>PCA parameters</h5>
          <div class="op-picker">
            <label>Number of components<input id="redN" type="number" value="5" min="1" max="50" /></label>
            <label>Target column (preserved if set)<select id="redTarget">${optsCols}</select></label>
            <label class="full"><input id="redDrop" type="checkbox" checked /> Drop original numeric columns</label>
          </div>`;
      } else if (m === 'variance_threshold') {
        params.innerHTML = `
          <h5>Variance threshold parameters</h5>
          <div class="op-picker">
            <label>Threshold (cols with variance ≤ this dropped)<input id="redThresh" type="number" value="0" step="0.01" /></label>
            <label>Target column (excluded from selection)<select id="redTarget">${optsCols}</select></label>
          </div>`;
      } else {
        params.innerHTML = `
          <h5>SelectKBest parameters</h5>
          <div class="op-picker">
            <label>K (top features to keep)<input id="redK" type="number" value="10" min="1" max="100" /></label>
            <label>Target column (required)<select id="redTarget">${optsCols}</select></label>
            <label class="full">Score function
              <select id="redScore">
                <option value="f_classif">f_classif (categorical target)</option>
                <option value="f_regression">f_regression (numeric target)</option>
                <option value="mutual_info_classif">mutual_info_classif</option>
                <option value="mutual_info_regression">mutual_info_regression</option>
              </select>
            </label>
          </div>`;
      }
    }
    methodSel.addEventListener('change', renderParams);
    renderParams();

    document.getElementById('redApply').addEventListener('click', async () => {
      const m = methodSel.value;
      const params = {};
      const tgt = document.getElementById('redTarget')?.value;
      if (tgt) params.target = tgt;
      if (m === 'pca') {
        params.n_components = parseInt(document.getElementById('redN').value, 10);
        params.drop_original = document.getElementById('redDrop').checked;
      } else if (m === 'variance_threshold') {
        params.threshold = parseFloat(document.getElementById('redThresh').value);
      } else {
        params.k = parseInt(document.getElementById('redK').value, 10);
        params.score_func = document.getElementById('redScore').value;
        if (!tgt) return showToast('Target column required for SelectKBest.', 'error');
      }
      const btn = document.getElementById('redApply');
      btn.disabled = true; btn.innerHTML = '<span class="loading"></span> Reducing…';
      try {
        const res = await API.clean(sid, {
          family: 'reduce', strategy: m, params,
        });
        showToast(res.message, 'success');
        close(); onApplied();
      } catch (e) {
        showToast(e.message, 'error');
        btn.disabled = false; btn.textContent = 'Apply reduction';
      }
    });
  }

  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
  function escAttr(s) { return escapeHtml(s); }

  return { openEngineer, openSplit, openBalance, openReduce, close };
})();

// Visualization workshop: chart selector + column dropdowns + drag-and-drop thumbnails.

const Viz = (() => {
  const typeSel  = document.getElementById('chartTypeSel');
  const colWrap  = document.getElementById('colSelectWrap');
  const colSel   = document.getElementById('colSel');
  const colWrapY = document.getElementById('colSelectWrapY');
  const colSelY  = document.getElementById('colSelY');
  const thumbsEl = document.getElementById('thumbs');
  const mainEl   = document.getElementById('mainChart');

  const NEEDS_X = new Set(['histogram', 'box', 'bar']);
  const NEEDS_XY = new Set(['scatter']);
  const NO_COL = new Set(['missing', 'issues', 'correlation', 'gauge']);

  let state = { sid: null, profile: null };

  function init(sid, profile) {
    state.sid = sid; state.profile = profile;
    populateColumnDropdowns(profile);
    renderThumbnails(profile);
    render();
  }

  function populateColumnDropdowns(profile) {
    const numeric = profile.columns.filter(c => c.inferred_type === 'numeric');
    const categorical = profile.columns.filter(c => c.inferred_type === 'categorical' || c.inferred_type === 'boolean');
    const all = profile.columns;

    fill(colSel, all);
    fill(colSelY, numeric);

    function fill(sel, cols) {
      sel.innerHTML = cols.map(c => `<option value="${escAttr(c.name)}">${escapeHtml(c.name)} (${c.inferred_type})</option>`).join('');
    }
  }

  function renderThumbnails(profile) {
    thumbsEl.innerHTML = profile.columns.map(c => {
      const meta = c.inferred_type === 'numeric'
        ? `μ=${fmtNum(c.mean)} σ=${fmtNum(c.std)}`
        : c.inferred_type === 'categorical' || c.inferred_type === 'boolean'
        ? `${c.unique} uniq`
        : c.inferred_type === 'datetime'
        ? `${c.unique} dates`
        : `${c.missing_pct}% miss`;
      return `
        <div class="thumb" draggable="true" data-col="${escAttr(c.name)}" data-type="${c.inferred_type}">
          <div>
            <div class="thumb-name" title="${escAttr(c.name)}">${escapeHtml(c.name)}</div>
            <span class="thumb-type col-type ${c.inferred_type}">${c.inferred_type}</span>
          </div>
          <div class="thumb-meta">${meta}</div>
        </div>`;
    }).join('');

    thumbsEl.querySelectorAll('.thumb').forEach(t => {
      t.addEventListener('dragstart', (e) => {
        t.classList.add('dragging');
        e.dataTransfer.setData('text/col', t.dataset.col);
        e.dataTransfer.setData('text/type', t.dataset.type);
      });
      t.addEventListener('dragend', () => t.classList.remove('dragging'));
      t.addEventListener('click', () => {
        const col = t.dataset.col;
        const type = t.dataset.type;
        // pick a sensible default chart based on the column type
        const chart = type === 'numeric'
          ? 'histogram'
          : (type === 'categorical' || type === 'boolean') ? 'bar'
          : (type === 'datetime') ? 'bar' : 'histogram';
        typeSel.value = chart;
        colSel.value = col;
        toggleColumnSelectors();
        render();
      });
    });

    mainEl.addEventListener('dragover', (e) => {
      e.preventDefault();
      mainEl.classList.add('drag-over');
    });
    mainEl.addEventListener('dragleave', () => mainEl.classList.remove('drag-over'));
    mainEl.addEventListener('drop', (e) => {
      e.preventDefault();
      mainEl.classList.remove('drag-over');
      const col = e.dataTransfer.getData('text/col');
      const type = e.dataTransfer.getData('text/type');
      if (!col) return;
      const chart = type === 'numeric' ? 'histogram'
                 : (type === 'categorical' || type === 'boolean') ? 'bar'
                 : 'histogram';
      typeSel.value = chart;
      colSel.value = col;
      toggleColumnSelectors();
      render();
    });
  }

  function toggleColumnSelectors() {
    const t = typeSel.value;
    colWrap.hidden = !(NEEDS_X.has(t) || NEEDS_XY.has(t));
    colWrapY.hidden = !NEEDS_XY.has(t);
  }

  typeSel.addEventListener('change', () => { toggleColumnSelectors(); render(); });
  colSel.addEventListener('change', render);
  colSelY.addEventListener('change', render);

  async function render() {
    if (!state.profile) return;
    const type = typeSel.value;
    const col = colSel.value;
    const colY = colSelY.value;
    try {
      if (type === 'missing') {
        Plots.missingHeatmap(state.profile, 'mainChart');
      } else if (type === 'issues') {
        Plots.issuesChart(state.profile, 'mainChart');
      } else if (type === 'gauge') {
        Plots.qualityGauge('mainChart', state.profile.quality_score, state.profile.grade);
      } else if (type === 'correlation') {
        const d = await API.correlation(state.sid);
        Plots.correlationHeatmap('mainChart', d.columns, d.matrix);
      } else if (type === 'histogram') {
        const d = await API.column(state.sid, col);
        Plots.histogram('mainChart', col, d.values);
      } else if (type === 'box') {
        const d = await API.column(state.sid, col);
        Plots.box('mainChart', col, d.values);
      } else if (type === 'bar') {
        const d = await API.column(state.sid, col);
        if (d.kind === 'categorical') {
          Plots.bar('mainChart', col, d.categories, d.counts);
        } else if (d.kind === 'numeric') {
          Plots.histogram('mainChart', col, d.values);
        } else {
          // datetime - show year counts
          const years = (d.values || []).map(v => String(v).slice(0, 4));
          const map = {};
          years.forEach(y => map[y] = (map[y] || 0) + 1);
          const cats = Object.keys(map).sort();
          Plots.bar('mainChart', col + ' (year)', cats, cats.map(c => map[c]));
        }
      } else if (type === 'scatter') {
        const d = await API.scatter(state.sid, col, colY);
        Plots.scatter('mainChart', col, colY, d.x_values, d.y_values);
      }
    } catch (e) {
      console.error('viz render failed', e);
      mainEl.innerHTML = `<div class="muted" style="text-align:center;padding:80px 0">${escapeHtml(e.message)}</div>`;
    }
  }

  function fmtNum(v) {
    if (v == null) return '—';
    if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 1 });
    return Math.round(v * 100) / 100;
  }
  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
  function escAttr(s) { return escapeHtml(s); }

  return { init, render };
})();

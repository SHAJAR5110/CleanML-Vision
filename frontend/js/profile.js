// Render dashboard cards, KPIs, column list, and preview table.

const Profile = (() => {
  function renderKPIs(p) {
    document.getElementById('kpiRows').textContent = p.rows.toLocaleString();
    document.getElementById('kpiCols').textContent = p.cols;
    document.getElementById('kpiMissing').textContent = `${p.missing_pct}%`;
    document.getElementById('kpiDups').textContent = p.duplicate_rows.toLocaleString();
    document.getElementById('kpiScore').textContent = p.quality_score;
    const g = document.getElementById('kpiGrade');
    g.textContent = p.grade;
    g.dataset.g = p.grade;
  }

  function renderColumns(p, onClick) {
    const wrap = document.getElementById('columnsTable');
    wrap.innerHTML = p.columns.map((c, i) => {
      const stats = statsHtml(c);
      const warns = (c.warnings || []).map(w => {
        const bad = ['constant_column', 'very_high_missing', 'id_like_column'].includes(w);
        return `<span class="warn-tag ${bad ? 'bad' : ''}">${prettyWarn(w)}</span>`;
      }).join('');
      return `
        <div class="col-card" data-idx="${i}">
          <div class="col-name">
            <span title="${escapeHtml(c.name)}">${escapeHtml(c.name)}</span>
            <span class="col-type ${c.inferred_type}">${c.inferred_type}</span>
          </div>
          <div class="col-stats">${stats}</div>
          ${warns ? `<div class="col-warns">${warns}</div>` : ''}
        </div>`;
    }).join('');
    if (typeof onClick === 'function') {
      wrap.querySelectorAll('.col-card').forEach(card => {
        card.addEventListener('click', () => {
          const idx = parseInt(card.dataset.idx, 10);
          onClick(p.columns[idx]);
        });
      });
    }
  }

  function statsHtml(c) {
    const items = [];
    items.push(`<span>missing: <b>${c.missing_pct}%</b></span>`);
    items.push(`<span>unique: <b>${c.unique}</b></span>`);
    if (c.inferred_type === 'numeric') {
      if (c.mean != null) items.push(`<span>mean: <b>${fmtNum(c.mean)}</b></span>`);
      if (c.median != null) items.push(`<span>median: <b>${fmtNum(c.median)}</b></span>`);
      if (c.min != null) items.push(`<span>min: <b>${fmtNum(c.min)}</b></span>`);
      if (c.max != null) items.push(`<span>max: <b>${fmtNum(c.max)}</b></span>`);
      if (c.outlier_count != null) items.push(`<span>outliers: <b>${c.outlier_count}</b></span>`);
      if (c.skew != null) items.push(`<span>skew: <b>${fmtNum(c.skew)}</b></span>`);
    } else if (c.top_values && c.top_values.length) {
      const top = c.top_values[0];
      items.push(`<span>top: <b title="${escapeHtml(top.value)}">${escapeHtml(truncate(top.value, 18))}</b></span>`);
      items.push(`<span>count: <b>${top.count}</b></span>`);
    } else if (c.inferred_type === 'datetime') {
      if (c.min) items.push(`<span>from: <b>${escapeHtml(truncate(c.min, 12))}</b></span>`);
      if (c.max) items.push(`<span>to: <b>${escapeHtml(truncate(c.max, 12))}</b></span>`);
    }
    items.push(`<span>card: <b>${c.cardinality}</b></span>`);
    return items.join('');
  }

  function renderPreview(preview, elId = 'previewTable') {
    const wrap = document.getElementById(elId);
    if (!wrap) return;
    if (!preview || !preview.length) {
      wrap.innerHTML = '<div class="muted" style="padding:14px">No rows.</div>';
      return;
    }
    const cols = Object.keys(preview[0]);
    const head = cols.map(c => `<th>${escapeHtml(c)}</th>`).join('');
    const body = preview.map(row =>
      `<tr>${cols.map(c => {
        const v = row[c];
        if (v === null || v === undefined || v === '') return `<td class="null">∅</td>`;
        return `<td title="${escapeHtml(String(v))}">${escapeHtml(truncate(String(v), 64))}</td>`;
      }).join('')}</tr>`
    ).join('');
    wrap.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  }

  function prettyWarn(w) {
    return ({
      very_high_missing: '>50% missing',
      high_missing: '>10% missing',
      embedded_nan_strings: 'NaN strings',
      constant_column: 'constant',
      id_like_column: 'ID-like',
      many_outliers: 'many outliers',
      highly_skewed: 'skewed',
    })[w] || w;
  }

  function fmtNum(v) {
    if (v == null) return '—';
    if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 1 });
    if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(2);
    return Math.round(v * 100) / 100;
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }
  function truncate(s, n) { return s.length > n ? s.slice(0, n - 1) + '…' : s; }

  return { renderKPIs, renderColumns, renderPreview };
})();

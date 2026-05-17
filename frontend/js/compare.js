// Before/After compare screen.
//
// Builds a quantified picture of what cleaning changed:
//   1. Hero strip — 5 big metrics with delta arrows
//   2. Two gauges (before / after score)
//   3. Radar chart on 5 data-health dimensions
//   4. Missing % per column (existing)
//   5. Ops breakdown donut
//   6. Column types before/after (existing)
//   7. Pipeline timeline

const Compare = (() => {

  // ---------- helpers ----------

  function fmt(v) {
    if (v == null) return '—';
    if (typeof v === 'number') return v.toLocaleString();
    return String(v);
  }

  function deltaSpan(before, after, opts = {}) {
    const d = after - before;
    if (d === 0) return '<span class="delta-zero">±0</span>';
    const positive = opts.invert ? d < 0 : d > 0;
    const cls = positive ? 'delta-up' : 'delta-down';
    const arrow = d > 0 ? '▲' : '▼';
    const sign = d > 0 ? '+' : '';
    const suffix = opts.suffix || '';
    return `<span class="delta ${cls}">${arrow} ${sign}${fmt(d)}${suffix}</span>`;
  }

  function gradeColor(grade) {
    return {
      A: '#20d4a0', B: '#6ed27a', C: '#ffb454',
      D: '#ff8a4a', F: '#ff5d7a',
    }[grade] || '#9aa1b4';
  }

  // ---------- 5-dimension health score ----------
  //
  // Each dimension is a 0-100 number derived from the profile:
  //   completeness   = 100 - missing %
  //   deduplication  = 100 - (duplicate rows / total rows × 100)
  //   type_clarity   = % of columns that aren't `text`, `id_like`, or `constant`
  //   outlier_control = 100 - (cells flagged as outliers / total cells × 100)
  //   quality        = the overall quality_score the profiler already computes

  function healthDimensions(profile) {
    const cols = profile.columns || [];
    const rows = profile.rows || 0;
    const totalCells = rows * cols.length;

    const completeness = Math.max(0, 100 - (profile.missing_pct || 0));
    const deduplication = rows
      ? Math.max(0, 100 - (profile.duplicate_rows / rows) * 100)
      : 100;

    const usableCols = cols.filter(c =>
      !['text', 'id_like', 'constant'].includes(c.inferred_type)
    ).length;
    const type_clarity = cols.length ? (usableCols / cols.length) * 100 : 100;

    const totalOutliers = cols.reduce((s, c) => s + (c.outlier_count || 0), 0);
    const outlier_control = totalCells
      ? Math.max(0, 100 - (totalOutliers / totalCells) * 100)
      : 100;

    return {
      completeness: Math.round(completeness),
      deduplication: Math.round(deduplication),
      type_clarity: Math.round(type_clarity),
      outlier_control: Math.round(outlier_control),
      quality: profile.quality_score || 0,
    };
  }

  // ---------- hero strip ----------

  function renderHero(before, after) {
    const metrics = [
      { label: 'rows',        b: before.rows,           a: after.rows,         invertDelta: false },
      { label: 'columns',     b: before.cols,           a: after.cols,         invertDelta: false },
      { label: 'missing %',   b: before.missing_pct,    a: after.missing_pct,  invertDelta: true, suffix: '%' },
      { label: 'duplicates',  b: before.duplicate_rows, a: after.duplicate_rows, invertDelta: true },
      {
        label: 'quality score',
        b: before.quality_score, a: after.quality_score,
        invertDelta: false,
        gradeBefore: before.grade, gradeAfter: after.grade,
      },
    ];

    const el = document.getElementById('compareHero');
    el.innerHTML = `
      <div class="hero-grid">
        ${metrics.map(m => `
          <div class="hero-metric">
            <div class="hero-lbl">${m.label}</div>
            <div class="hero-row">
              <div class="hero-side before">
                <div class="hero-val">${fmt(m.b)}</div>
                ${m.gradeBefore ? `<span class="grade-pill" style="background:${gradeColor(m.gradeBefore)}20;color:${gradeColor(m.gradeBefore)}">${m.gradeBefore}</span>` : ''}
              </div>
              <div class="hero-arrow">→</div>
              <div class="hero-side after">
                <div class="hero-val">${fmt(m.a)}</div>
                ${m.gradeAfter ? `<span class="grade-pill" style="background:${gradeColor(m.gradeAfter)}20;color:${gradeColor(m.gradeAfter)}">${m.gradeAfter}</span>` : ''}
              </div>
            </div>
            <div class="hero-delta-row">${deltaSpan(m.b, m.a, { invert: m.invertDelta, suffix: m.suffix || '' })}</div>
          </div>
        `).join('')}
      </div>
    `;
  }

  // ---------- pipeline timeline ----------

  const FAMILY_META = {
    missing:     { icon: '⊖', color: '#ffb454' },
    outliers:    { icon: '◔', color: '#7c5cff' },
    encoders:    { icon: '#', color: '#20d4a0' },
    scalers:     { icon: '∿', color: '#6ed27a' },
    duplicates:  { icon: '⇄', color: '#ff8a4a' },
    text:        { icon: 'A', color: '#c7baff' },
    datetime:    { icon: '⏱', color: '#ffb454' },
    dtype:       { icon: '⚙', color: '#9aa1b4' },
    label_norm:  { icon: '🏷', color: '#20d4a0' },
    validate:    { icon: '✓', color: '#6ed27a' },
    feature_eng: { icon: '🧮', color: '#7c5cff' },
    splitter:    { icon: '✂', color: '#ff8a4a' },
    balance:     { icon: '⚖', color: '#ffb454' },
    reduce:      { icon: '📉', color: '#ff5d7a' },
    merge:       { icon: '🔗', color: '#20d4a0' },
    quality:     { icon: '🌫', color: '#ffb454' },
    dedup:       { icon: '🔁', color: '#7c5cff' },
    transforms:  { icon: '🎨', color: '#20d4a0' },
    augment:     { icon: '➕', color: '#6ed27a' },
    pair:        { icon: '🔗', color: '#c7baff' },
  };

  function renderTimeline(history) {
    const el = document.getElementById('opsTimeline');
    if (!history || !history.length) {
      el.innerHTML = '<div class="muted" style="padding:14px">No operations applied yet.</div>';
      return;
    }
    el.innerHTML = history.map((h, i) => {
      const fam = (h.op && h.op.family) || 'other';
      const meta = FAMILY_META[fam] || { icon: '•', color: '#9aa1b4' };
      const strat = (h.op && h.op.strategy) || '';
      return `
        <div class="timeline-row">
          <div class="timeline-step">${i + 1}</div>
          <div class="timeline-dot" style="background:${meta.color}">${meta.icon}</div>
          <div class="timeline-body">
            <div class="timeline-head">
              <span class="timeline-fam">${fam}.${strat}</span>
              ${h.op && h.op.column ? `<span class="timeline-col">${escapeHtml(h.op.column)}</span>` : ''}
            </div>
            <div class="timeline-msg">${escapeHtml(h.message || '')}</div>
          </div>
        </div>
      `;
    }).join('');
  }

  // ---------- main entry ----------

  function render(data) {
    renderHero(data.before, data.after);

    Plots.scoreGauge('chartBeforeGauge', data.before.quality_score, data.before.grade);
    Plots.scoreGauge('chartAfterGauge', data.after.quality_score, data.after.grade);

    Plots.healthRadar(
      'chartHealthRadar',
      healthDimensions(data.before),
      healthDimensions(data.after),
    );

    // Missing % per column
    const beforeMap = new Map(data.before.columns.map(c => [c.name, c.missing_pct]));
    const afterMap = new Map(data.after.columns.map(c => [c.name, c.missing_pct]));
    const allCols = Array.from(new Set([...beforeMap.keys(), ...afterMap.keys()]));
    Plots.missingCompare('chartMissingCompare', allCols, beforeMap, afterMap);

    // Operations breakdown donut
    Plots.opsBreakdown('chartOpsBreakdown', data.history || []);

    // Type counts
    const typeCount = (cols) =>
      cols.reduce((m, c) => (m[c.inferred_type] = (m[c.inferred_type] || 0) + 1, m), {});
    Plots.typeCompare('chartTypeCompare', typeCount(data.before.columns), typeCount(data.after.columns));

    // Timeline
    renderTimeline(data.history);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }

  return { render };
})();

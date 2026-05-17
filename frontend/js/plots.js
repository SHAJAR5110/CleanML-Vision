// Plotly charts — all chart types supported by the visualizer.

const Plots = (() => {
  const LAYOUT_BASE = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#9aa1b4', family: 'ui-sans-serif, Segoe UI, sans-serif', size: 11 },
    margin: { l: 60, r: 30, t: 30, b: 60 },
    xaxis: { gridcolor: '#232838', zerolinecolor: '#2f3547' },
    yaxis: { gridcolor: '#232838', zerolinecolor: '#2f3547' },
  };
  const CONFIG = { displayModeBar: false, responsive: true };

  function missingHeatmap(profile, divId = 'mainChart') {
    const z = profile.missing_matrix;
    const cols = profile.missing_matrix_cols;
    if (!z || !z.length || !cols.length) {
      Plotly.purge(divId);
      document.getElementById(divId).innerHTML =
        '<div class="muted" style="text-align:center;padding:60px 0">No data.</div>';
      return;
    }
    const trace = {
      z, x: z[0].map((_, i) => i), y: cols, type: 'heatmap',
      colorscale: [[0, '#161a23'], [1, '#ff5d7a']],
      showscale: false,
      hovertemplate: 'col: %{y}<br>row: %{x}<br>missing: %{z}<extra></extra>',
    };
    const layout = {
      ...LAYOUT_BASE, height: 420,
      xaxis: { ...LAYOUT_BASE.xaxis, title: 'row', showticklabels: false },
      yaxis: { ...LAYOUT_BASE.yaxis, automargin: true },
      title: { text: 'Missing-value heatmap', font: { color: '#e6e9f2', size: 13 } },
    };
    Plotly.newPlot(divId, [trace], layout, CONFIG);
  }

  function issuesChart(profile, divId = 'mainChart') {
    const cols = profile.columns;
    const labels = cols.map(c => c.name);
    const missing = cols.map(c => c.missing_pct);
    const outliersPct = cols.map(c => {
      if (c.outlier_count == null || !c.count) return 0;
      return Math.round((c.outlier_count / c.count) * 1000) / 10;
    });
    const t1 = {
      x: labels, y: missing, type: 'bar', name: 'missing %',
      marker: { color: '#ff8a4a' },
      hovertemplate: '%{x}<br>missing: %{y}%<extra></extra>',
    };
    const t2 = {
      x: labels, y: outliersPct, type: 'bar', name: 'outlier %',
      marker: { color: '#7c5cff' },
      hovertemplate: '%{x}<br>outliers: %{y}%<extra></extra>',
    };
    const layout = {
      ...LAYOUT_BASE, height: 420, barmode: 'group',
      xaxis: { ...LAYOUT_BASE.xaxis, tickangle: -35, automargin: true },
      yaxis: { ...LAYOUT_BASE.yaxis, ticksuffix: '%', rangemode: 'tozero' },
      legend: { orientation: 'h', y: 1.12, x: 0 },
      title: { text: 'Column quality issues', font: { color: '#e6e9f2', size: 13 } },
    };
    Plotly.newPlot(divId, [t1, t2], layout, CONFIG);
  }

  function histogram(divId, col, values) {
    const trace = {
      x: values, type: 'histogram', nbinsx: 40,
      marker: { color: '#7c5cff', line: { color: '#5e44d6', width: 1 } },
      hovertemplate: 'range: %{x}<br>count: %{y}<extra></extra>',
    };
    const layout = {
      ...LAYOUT_BASE, height: 420,
      xaxis: { ...LAYOUT_BASE.xaxis, title: col },
      yaxis: { ...LAYOUT_BASE.yaxis, title: 'count' },
      title: { text: `Histogram — ${col}`, font: { color: '#e6e9f2', size: 13 } },
      bargap: 0.05,
    };
    Plotly.newPlot(divId, [trace], layout, CONFIG);
  }

  function box(divId, col, values) {
    const trace = {
      y: values, type: 'box', name: col,
      marker: { color: '#20d4a0' }, boxpoints: 'outliers',
      jitter: 0.3, line: { color: '#20d4a0' },
    };
    const layout = {
      ...LAYOUT_BASE, height: 420,
      yaxis: { ...LAYOUT_BASE.yaxis, title: col },
      title: { text: `Box plot — ${col}`, font: { color: '#e6e9f2', size: 13 } },
      showlegend: false,
    };
    Plotly.newPlot(divId, [trace], layout, CONFIG);
  }

  function bar(divId, col, categories, counts) {
    const trace = {
      x: categories, y: counts, type: 'bar',
      marker: { color: '#20d4a0' },
      hovertemplate: '%{x}<br>count: %{y}<extra></extra>',
    };
    const layout = {
      ...LAYOUT_BASE, height: 420,
      xaxis: { ...LAYOUT_BASE.xaxis, tickangle: -35, automargin: true },
      yaxis: { ...LAYOUT_BASE.yaxis, title: 'count', rangemode: 'tozero' },
      title: { text: `Bar chart — ${col}`, font: { color: '#e6e9f2', size: 13 } },
    };
    Plotly.newPlot(divId, [trace], layout, CONFIG);
  }

  function scatter(divId, xCol, yCol, xs, ys) {
    const trace = {
      x: xs, y: ys, mode: 'markers', type: 'scatter',
      marker: { color: '#7c5cff', size: 6, opacity: 0.65,
                line: { color: '#5e44d6', width: 0.5 } },
      hovertemplate: `${xCol}: %{x}<br>${yCol}: %{y}<extra></extra>`,
    };
    const layout = {
      ...LAYOUT_BASE, height: 420,
      xaxis: { ...LAYOUT_BASE.xaxis, title: xCol },
      yaxis: { ...LAYOUT_BASE.yaxis, title: yCol },
      title: { text: `${xCol} vs ${yCol}`, font: { color: '#e6e9f2', size: 13 } },
    };
    Plotly.newPlot(divId, [trace], layout, CONFIG);
  }

  function correlationHeatmap(divId, cols, matrix) {
    if (!cols || !cols.length) {
      document.getElementById(divId).innerHTML =
        '<div class="muted" style="text-align:center;padding:60px 0">Need 2+ numeric columns.</div>';
      return;
    }
    const trace = {
      z: matrix, x: cols, y: cols, type: 'heatmap',
      colorscale: [
        [0,   '#ff5d7a'], [0.25, '#ff8a4a'],
        [0.5, '#161a23'], [0.75, '#6ed27a'], [1, '#20d4a0'],
      ],
      zmin: -1, zmax: 1, showscale: true,
      hovertemplate: '%{x} vs %{y}<br>corr: %{z:.3f}<extra></extra>',
    };
    const layout = {
      ...LAYOUT_BASE, height: 480,
      xaxis: { ...LAYOUT_BASE.xaxis, tickangle: -35, automargin: true },
      yaxis: { ...LAYOUT_BASE.yaxis, automargin: true },
      title: { text: 'Correlation heatmap', font: { color: '#e6e9f2', size: 13 } },
    };
    Plotly.newPlot(divId, [trace], layout, CONFIG);
  }

  function qualityGauge(divId, score, grade) {
    const trace = {
      type: 'indicator', mode: 'gauge+number',
      value: score,
      title: { text: `Quality grade: ${grade}`, font: { color: '#e6e9f2', size: 16 } },
      number: { font: { color: '#e6e9f2', size: 48 } },
      gauge: {
        axis: { range: [0, 100], tickcolor: '#9aa1b4' },
        bar: { color:
          score >= 90 ? '#20d4a0' :
          score >= 75 ? '#6ed27a' :
          score >= 60 ? '#ffb454' :
          score >= 40 ? '#ff8a4a' : '#ff5d7a'
        },
        bgcolor: '#11141b',
        steps: [
          { range: [0, 40],  color: 'rgba(255,93,122,0.2)' },
          { range: [40, 60], color: 'rgba(255,138,74,0.2)' },
          { range: [60, 75], color: 'rgba(255,180,84,0.2)' },
          { range: [75, 90], color: 'rgba(110,210,122,0.2)' },
          { range: [90, 100],color: 'rgba(32,212,160,0.2)' },
        ],
        borderwidth: 0,
      },
    };
    const layout = { ...LAYOUT_BASE, height: 420 };
    Plotly.newPlot(divId, [trace], layout, CONFIG);
  }

  function scoreDelta(divId, before, after) {
    const delta = after - before;
    const color = delta > 0 ? '#20d4a0' : delta < 0 ? '#ff5d7a' : '#9aa1b4';
    const trace = {
      type: 'bar', orientation: 'h',
      x: [before, after], y: ['Before', 'After'],
      text: [`${before}`, `${after}`], textposition: 'inside',
      marker: { color: ['#6b7388', color], line: { color: '#232838', width: 1 } },
      hovertemplate: '%{y}: %{x}<extra></extra>',
    };
    const layout = {
      ...LAYOUT_BASE, height: 180,
      xaxis: { ...LAYOUT_BASE.xaxis, range: [0, 100], title: 'Quality score' },
      annotations: [{
        x: Math.max(before, after) + 8, y: 0.5, xref: 'x', yref: 'paper',
        text: `Δ ${delta >= 0 ? '+' : ''}${delta}`,
        showarrow: false, font: { color, size: 16 },
      }],
    };
    Plotly.newPlot(divId, [trace], layout, CONFIG);
  }

  function missingCompare(divId, cols, beforeMap, afterMap) {
    const b = cols.map(c => beforeMap.get(c) ?? null);
    const a = cols.map(c => afterMap.get(c) ?? null);
    const t1 = { x: cols, y: b, name: 'before', type: 'bar', marker: { color: '#6b7388' } };
    const t2 = { x: cols, y: a, name: 'after',  type: 'bar', marker: { color: '#20d4a0' } };
    const layout = {
      ...LAYOUT_BASE, height: 320, barmode: 'group',
      xaxis: { ...LAYOUT_BASE.xaxis, tickangle: -35, automargin: true },
      yaxis: { ...LAYOUT_BASE.yaxis, ticksuffix: '%', rangemode: 'tozero' },
      legend: { orientation: 'h', y: 1.12, x: 0 },
    };
    Plotly.newPlot(divId, [t1, t2], layout, CONFIG);
  }

  function typeCompare(divId, before, after) {
    const types = Array.from(new Set([...Object.keys(before), ...Object.keys(after)]));
    const b = types.map(t => before[t] || 0);
    const a = types.map(t => after[t] || 0);
    const t1 = { x: types, y: b, name: 'before', type: 'bar', marker: { color: '#6b7388' } };
    const t2 = { x: types, y: a, name: 'after',  type: 'bar', marker: { color: '#7c5cff' } };
    const layout = {
      ...LAYOUT_BASE, height: 320, barmode: 'group',
      xaxis: { ...LAYOUT_BASE.xaxis, automargin: true },
      yaxis: { ...LAYOUT_BASE.yaxis, rangemode: 'tozero', dtick: 1 },
      legend: { orientation: 'h', y: 1.12, x: 0 },
    };
    Plotly.newPlot(divId, [t1, t2], layout, CONFIG);
  }

  // ---------- compare-screen charts ----------

  function scoreGauge(divId, score, grade, label) {
    const color =
      score >= 90 ? '#20d4a0' :
      score >= 75 ? '#6ed27a' :
      score >= 60 ? '#ffb454' :
      score >= 40 ? '#ff8a4a' : '#ff5d7a';
    const trace = {
      type: 'indicator', mode: 'gauge+number',
      value: score,
      title: { text: label ? `${label} — ${grade}` : grade, font: { color: '#e6e9f2', size: 14 } },
      number: { font: { color, size: 42 } },
      gauge: {
        axis: { range: [0, 100], tickcolor: '#9aa1b4', tickfont: { color: '#6b7388' } },
        bar: { color },
        bgcolor: '#11141b',
        borderwidth: 0,
        steps: [
          { range: [0, 40],  color: 'rgba(255,93,122,0.18)' },
          { range: [40, 60], color: 'rgba(255,138,74,0.18)' },
          { range: [60, 75], color: 'rgba(255,180,84,0.18)' },
          { range: [75, 90], color: 'rgba(110,210,122,0.18)' },
          { range: [90, 100], color: 'rgba(32,212,160,0.22)' },
        ],
      },
    };
    Plotly.newPlot(divId, [trace], { ...LAYOUT_BASE, height: 280 }, CONFIG);
  }

  function healthRadar(divId, before, after) {
    // before / after each have { completeness, deduplication, type_clarity,
    //                            outlier_control, quality }
    const dims = ['Completeness', 'Deduplication', 'Type clarity',
                  'Outlier control', 'Quality score'];
    const b = [before.completeness, before.deduplication, before.type_clarity,
               before.outlier_control, before.quality];
    const a = [after.completeness, after.deduplication, after.type_clarity,
               after.outlier_control, after.quality];
    // Plotly closes the polygon if the first point is repeated at the end
    const close = arr => [...arr, arr[0]];
    const dimsClosed = close(dims);
    const traces = [
      {
        type: 'scatterpolar',
        r: close(b), theta: dimsClosed,
        fill: 'toself', name: 'before',
        line: { color: '#6b7388', width: 2 },
        fillcolor: 'rgba(107,115,136,0.25)',
        hovertemplate: '%{theta}: %{r:.0f}<extra>before</extra>',
      },
      {
        type: 'scatterpolar',
        r: close(a), theta: dimsClosed,
        fill: 'toself', name: 'after',
        line: { color: '#20d4a0', width: 2 },
        fillcolor: 'rgba(32,212,160,0.28)',
        hovertemplate: '%{theta}: %{r:.0f}<extra>after</extra>',
      },
    ];
    const layout = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#9aa1b4', family: 'ui-sans-serif, Segoe UI, sans-serif', size: 12 },
      polar: {
        bgcolor: 'rgba(17,20,27,0.6)',
        radialaxis: {
          visible: true, range: [0, 100], gridcolor: '#232838',
          tickfont: { color: '#6b7388', size: 10 }, tickvals: [25, 50, 75, 100],
        },
        angularaxis: {
          gridcolor: '#232838', linecolor: '#2f3547',
          tickfont: { color: '#e6e9f2', size: 12 },
        },
      },
      legend: { orientation: 'h', y: 1.06, x: 0.5, xanchor: 'center' },
      height: 380,
      margin: { l: 60, r: 60, t: 60, b: 40 },
    };
    Plotly.newPlot(divId, traces, layout, CONFIG);
  }

  function opsBreakdown(divId, history) {
    const counts = {};
    history.forEach(h => {
      const fam = (h.op && h.op.family) || 'other';
      counts[fam] = (counts[fam] || 0) + 1;
    });
    const labels = Object.keys(counts);
    const values = Object.values(counts);
    if (!labels.length) {
      document.getElementById(divId).innerHTML =
        '<div class="muted" style="text-align:center;padding:80px 0">No operations applied yet.</div>';
      return;
    }
    const palette = [
      '#7c5cff', '#20d4a0', '#ffb454', '#ff8a4a',
      '#6ed27a', '#ff5d7a', '#5e44d6', '#c7baff',
    ];
    const trace = {
      type: 'pie', hole: 0.55,
      labels, values,
      marker: { colors: labels.map((_, i) => palette[i % palette.length]),
                line: { color: '#11141b', width: 2 } },
      textinfo: 'label+value',
      textfont: { color: '#e6e9f2', size: 12 },
      hovertemplate: '%{label}: %{value} ops (%{percent})<extra></extra>',
    };
    const layout = {
      ...LAYOUT_BASE, height: 320,
      showlegend: true,
      legend: { orientation: 'v', y: 0.5, x: 1.05, font: { color: '#9aa1b4', size: 11 } },
      margin: { l: 20, r: 100, t: 20, b: 20 },
      annotations: [{
        text: `<b>${history.length}</b><br><span style="font-size:11px;color:#6b7388">ops</span>`,
        showarrow: false, font: { color: '#e6e9f2', size: 22 },
        x: 0.5, y: 0.5, xanchor: 'center', yanchor: 'middle',
      }],
    };
    Plotly.newPlot(divId, [trace], layout, CONFIG);
  }

  return {
    missingHeatmap, issuesChart, histogram, box, bar, scatter,
    correlationHeatmap, qualityGauge,
    scoreDelta, missingCompare, typeCompare,
    scoreGauge, healthRadar, opsBreakdown,
  };
})();

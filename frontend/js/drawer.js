// Per-column cleaning drawer — dropdown-driven UI.

// Shared category → strategies map (also used by op-modal.js).
// Each strategy: [code, label, kind?, defaultValue?]
// kind: 'value' (free text), 'number' (numeric), 'list' (comma-separated), null (no params)
const OP_CATEGORIES = {
    'Missing values': {
      family: 'missing',
      strategies: [
        ['mean',            'Fill NaN with mean'],
        ['median',          'Fill NaN with median'],
        ['mode',            'Fill NaN with mode (most common)'],
        ['knn',             'KNN imputer (uses all numeric cols)'],
        ['ffill',           'Forward-fill NaN'],
        ['bfill',           'Back-fill NaN'],
        ['constant',        'Fill NaN with custom value…', 'value', 'Unknown'],
        ['standardize_nan', 'Convert "?", "N/A", "null" tokens to NaN'],
        ['drop_rows',       'Drop rows with NaN'],
        ['drop_column',     'Drop this column'],
      ],
    },
    'Outliers': {
      family: 'outliers',
      strategies: [
        ['iqr_remove',       'Remove IQR outliers (1.5× IQR)'],
        ['iqr_cap',          'Cap IQR outliers'],
        ['zscore_remove',    'Remove |z|>threshold outliers', 'number', '3'],
        ['isolation_forest', 'Isolation Forest (multivariate)'],
        ['dbscan',           'DBSCAN density anomalies (multivariate)'],
      ],
    },
    'Encoding': {
      family: 'encoders',
      strategies: [
        ['onehot',     'One-hot encode'],
        ['label',      'Label encode (0,1,2…)'],
        ['ordinal',    'Ordinal encode (specify order)', 'list', 'low,medium,high'],
        ['frequency',  'Frequency encode'],
        ['target',     'Target encode (use target column)', 'value', 'target_column_name'],
        ['binary',     'Binary encode (2 values only)'],
      ],
    },
    'Scaling': {
      family: 'scalers',
      strategies: [
        ['standard',  'Standard-scale (mean=0, std=1)'],
        ['minmax',    'Min-max scale to [0,1]'],
        ['robust',    'Robust scale (median/IQR)'],
        ['normalize', 'L2 normalize'],
        ['log',       'Log transform'],
      ],
    },
    'Text cleanup': {
      family: 'text',
      strategies: [
        ['strip',              'Strip whitespace'],
        ['lowercase',          'Lowercase'],
        ['collapse_spaces',    'Collapse whitespace'],
        ['remove_special',     'Remove special characters'],
        ['remove_punctuation', 'Remove punctuation'],
        ['remove_stopwords',   'Remove English stopwords'],
        ['alphabetic_only',    'Keep only letters'],
        ['word_count',         'Add word-count feature column'],
        ['char_count',         'Add char-count feature column'],
      ],
    },
    'Datetime': {
      family: 'datetime',
      strategies: [
        ['parse',   'Parse as datetime'],
        ['extract', 'Extract year/month/weekday'],
      ],
    },
  'Dtype repair': {
    family: 'dtype',
    strategies: [
      ['to_numeric',  'Force numeric (strip $, %, …)'],
      ['to_boolean',  'Coerce to True/False'],
    ],
  },
};

// Recommended category per inferred type.
const OP_DEFAULT_CATEGORY = {
  numeric:     'Missing values',
  categorical: 'Encoding',
  boolean:     'Missing values',
  datetime:    'Datetime',
  text:        'Text cleanup',
  id_like:     'Missing values',
  constant:    'Missing values',
};

// Find which category a family belongs to (reverse lookup).
function opCategoryForFamily(family) {
  for (const [name, cfg] of Object.entries(OP_CATEGORIES)) {
    if (cfg.family === family) return name;
  }
  return 'Missing values';
}

const Drawer = (() => {
  const el = document.getElementById('colDrawer');
  const nameEl = document.getElementById('drawerName');
  const subEl = document.getElementById('drawerSub');
  const bodyEl = document.getElementById('drawerBody');
  document.getElementById('drawerClose').addEventListener('click', close);

  const CATEGORIES = OP_CATEGORIES;
  const DEFAULT_CATEGORY = OP_DEFAULT_CATEGORY;

  function open(colProfile, sid, onApplied) {
    nameEl.textContent = colProfile.name;
    subEl.textContent =
      `${colProfile.inferred_type} · ${colProfile.missing_pct}% missing · ${colProfile.unique} unique · ${colProfile.cardinality}`;

    const defaultCat = DEFAULT_CATEGORY[colProfile.inferred_type] || 'Missing values';

    const isTextLike = ['categorical', 'text', 'boolean', 'id_like'].includes(colProfile.inferred_type);
    bodyEl.innerHTML = `
      ${isTextLike ? `
      <div class="drawer-section">
        <h4>Quick tools</h4>
        <button class="op-btn" id="openLabelNorm">
          🏷 Normalize inconsistent labels
          <small>group similar values like "Male"/"M"/"male"</small>
        </button>
      </div>` : ''}
      <div class="drawer-section">
        <h4>Operation category</h4>
        <select id="drawerCat">
          ${Object.keys(CATEGORIES).map(c =>
            `<option value="${c}" ${c === defaultCat ? 'selected' : ''}>${c}</option>`
          ).join('')}
        </select>
      </div>
      <div class="drawer-section">
        <h4>Strategy</h4>
        <select id="drawerStrat"></select>
        <div class="strat-desc muted" id="stratDesc"></div>
      </div>
      <div class="drawer-section" id="paramSection" hidden>
        <h4 id="paramLabel">Parameter</h4>
        <input id="paramInput" type="text" />
      </div>
      <div class="drawer-section">
        <button class="btn primary big drawer-apply" id="applyBtn">Apply operation</button>
      </div>
    `;

    const catSel = document.getElementById('drawerCat');
    const stratSel = document.getElementById('drawerStrat');
    const stratDesc = document.getElementById('stratDesc');
    const paramSection = document.getElementById('paramSection');
    const paramLabel = document.getElementById('paramLabel');
    const paramInput = document.getElementById('paramInput');
    const applyBtn = document.getElementById('applyBtn');

    function refreshStrats() {
      const cat = CATEGORIES[catSel.value];
      stratSel.innerHTML = cat.strategies.map(([code, label]) =>
        `<option value="${code}">${label}</option>`).join('');
      refreshParams();
    }

    function refreshParams() {
      const cat = CATEGORIES[catSel.value];
      const strat = cat.strategies.find(s => s[0] === stratSel.value);
      if (!strat) return;
      const [, label, kind, def] = strat;
      stratDesc.textContent = label;
      if (kind) {
        paramSection.hidden = false;
        paramLabel.textContent =
          kind === 'value' ? 'Fill value' :
          kind === 'number' ? 'Numeric threshold' :
          kind === 'list' ? 'Comma-separated values' : 'Parameter';
        paramInput.type = kind === 'number' ? 'number' : 'text';
        paramInput.value = def || '';
        paramInput.placeholder = def || '';
      } else {
        paramSection.hidden = true;
        paramInput.value = '';
      }
    }

    catSel.addEventListener('change', refreshStrats);
    stratSel.addEventListener('change', refreshParams);
    refreshStrats();

    const labelBtn = document.getElementById('openLabelNorm');
    if (labelBtn) {
      labelBtn.addEventListener('click', () => {
        close();
        LabelModal.open({ sid, column: colProfile.name, onApplied });
      });
    }

    applyBtn.addEventListener('click', async () => {
      const cat = CATEGORIES[catSel.value];
      const family = cat.family;
      const strategy = stratSel.value;
      const stratDef = cat.strategies.find(s => s[0] === strategy);
      const kind = stratDef[2];

      const params = {};
      if (kind === 'value') {
        let v = paramInput.value;
        if (colProfile.inferred_type === 'numeric') {
          const n = parseFloat(v);
          if (!Number.isNaN(n)) v = n;
        }
        if (strategy === 'target') params.target = v;
        else params.value = v;
      } else if (kind === 'number') {
        params.threshold = parseFloat(paramInput.value);
      } else if (kind === 'list') {
        params.order = paramInput.value.split(',').map(s => s.trim()).filter(Boolean);
      }

      const orig = applyBtn.innerHTML;
      applyBtn.disabled = true;
      applyBtn.innerHTML = '<span class="loading"></span> Applying…';
      try {
        const res = await API.clean(sid, {
          family, strategy, column: colProfile.name, params,
        });
        showToast(res.message, 'success');
        close();
        onApplied(res);
      } catch (e) {
        showToast(e.message, 'error');
        applyBtn.disabled = false;
        applyBtn.innerHTML = orig;
      }
    });

    el.hidden = false;
  }

  function close() { el.hidden = true; }

  return { open, close };
})();

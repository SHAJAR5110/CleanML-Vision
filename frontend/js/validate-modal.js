// Cross-field validation modal.

const ValidateModal = (() => {
  const backdrop = document.getElementById('validateModal');
  const body = document.getElementById('validateModalBody');
  document.getElementById('validateModalClose').addEventListener('click', close);
  backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });

  const RULES = {
    less_than:      { label: 'A < B (e.g. start_date < end_date)',     fields: ['a', 'b'] },
    less_or_equal:  { label: 'A ≤ B',                                  fields: ['a', 'b'] },
    equal:          { label: 'A = B (consistency check)',              fields: ['a', 'b'] },
    sum_equals:     { label: 'sum(cols) == total_col',                 fields: ['cols', 'total_col'] },
    age_dob:        { label: 'age column matches DOB column',          fields: ['age_col', 'dob_col'] },
  };

  function close() { backdrop.hidden = true; }

  function open({ sid, profile, onApplied }) {
    const cols = profile.columns.map(c => c.name);
    body.innerHTML = `
      <div class="modal-section">
        <h5>Pick rule</h5>
        <select id="vmRule" class="full-width">
          ${Object.entries(RULES).map(([k, r]) =>
            `<option value="${k}">${r.label}</option>`
          ).join('')}
        </select>
      </div>
      <div class="modal-section" id="vmFields"></div>
      <div class="modal-section">
        <h5>Action</h5>
        <select id="vmAction">
          <option value="check">Flag violations in a new column (non-destructive)</option>
          <option value="drop_violations">Drop violating rows</option>
        </select>
      </div>
      <div class="modal-section">
        <button class="btn primary big drawer-apply" id="vmApply">Apply</button>
      </div>
    `;

    const ruleSel = document.getElementById('vmRule');
    const fields = document.getElementById('vmFields');

    function renderFields() {
      const rule = ruleSel.value;
      const def = RULES[rule];
      fields.innerHTML = `<h5>Columns</h5><div class="op-picker">` +
        def.fields.map(f => {
          if (f === 'cols') {
            return `<label class="full">${prettyLabel(f)} (Ctrl-click to multi-select)
              <select id="vm_${f}" multiple size="5">${cols.map(c => `<option value="${escAttr(c)}">${escapeHtml(c)}</option>`).join('')}</select>
            </label>`;
          }
          return `<label>${prettyLabel(f)}
            <select id="vm_${f}">${cols.map(c => `<option value="${escAttr(c)}">${escapeHtml(c)}</option>`).join('')}</select>
          </label>`;
        }).join('') + `</div>`;
    }
    ruleSel.addEventListener('change', renderFields);
    renderFields();

    document.getElementById('vmApply').addEventListener('click', async () => {
      const rule = ruleSel.value;
      const action = document.getElementById('vmAction').value;
      const def = RULES[rule];
      const params = { rule };
      for (const f of def.fields) {
        if (f === 'cols') {
          const sel = document.getElementById('vm_cols');
          params.cols = Array.from(sel.selectedOptions).map(o => o.value);
          if (params.cols.length < 1) return showToast('Pick at least one column to sum.', 'error');
        } else {
          params[f] = document.getElementById(`vm_${f}`).value;
        }
      }

      const btn = document.getElementById('vmApply');
      btn.disabled = true; btn.innerHTML = '<span class="loading"></span> Applying…';
      try {
        const res = await API.clean(sid, {
          family: 'validate', strategy: action, params,
        });
        showToast(res.message, 'success');
        close();
        onApplied();
      } catch (e) {
        showToast(e.message, 'error');
        btn.disabled = false; btn.textContent = 'Apply';
      }
    });
  }

  function prettyLabel(f) {
    return ({
      a: 'Column A', b: 'Column B',
      cols: 'Columns to sum',
      total_col: 'Total column (expected sum)',
      age_col: 'Age column', dob_col: 'Date-of-birth column',
    })[f] || f;
  }
  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
  function escAttr(s) { return escapeHtml(s); }

  return { open, close };
})();

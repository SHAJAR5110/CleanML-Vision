// Suggested next-steps panel.

const Suggest = (() => {
  function render(suggestions, sid, onApplied) {
    const el = document.getElementById('suggestList');
    if (!suggestions || !suggestions.length) {
      el.innerHTML = `
        <div class="muted" style="padding:14px;text-align:center">
          ✓ No further suggestions — your data looks clean.
        </div>`;
      return;
    }
    el.innerHTML = suggestions.map((s, i) => `
      <div class="suggest-item" data-id="${s.id}">
        <div class="suggest-head">
          <div class="suggest-title">
            <span class="impact-dot impact-${s.impact}" title="${s.impact} impact"></span>
            ${escapeHtml(s.title)}
          </div>
          <span class="suggest-badge ${s.badge}">${s.badge}</span>
        </div>
        <div class="suggest-reason">${escapeHtml(s.reason)}</div>
        <div class="suggest-actions">
          <button class="btn" data-see="${i}">👁 See</button>
          <button class="btn primary" data-apply="${i}">Apply</button>
          <button class="btn ghost dismiss-btn" data-dismiss="${i}">Dismiss</button>
        </div>
      </div>
    `).join('');

    el.querySelectorAll('[data-see]').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.see, 10);
        OpModal.open({
          suggestion: suggestions[idx],
          sid,
          onApplied,
        });
      });
    });
    el.querySelectorAll('[data-apply]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const idx = parseInt(btn.dataset.apply, 10);
        const s = suggestions[idx];
        btn.disabled = true;
        btn.innerHTML = '<span class="loading"></span>';
        try {
          const res = await API.clean(sid, s.op);
          showToast(res.message, 'success');
          onApplied();
        } catch (e) {
          showToast(e.message, 'error');
          btn.disabled = false;
          btn.textContent = 'Apply';
        }
      });
    });
    el.querySelectorAll('.dismiss-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.dismiss, 10);
        const card = el.querySelector(`[data-id="${suggestions[idx].id}"]`);
        if (card) card.remove();
      });
    });
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }

  return { render };
})();

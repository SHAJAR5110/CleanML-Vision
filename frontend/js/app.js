// Main app controller.

(function () {
  const state = {
    sid: null,
    source: null,
    profile: null,
    originalPreview: null,
    historyLen: 0,
  };

  function goto(stepName) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(`screen-${stepName}`).classList.add('active');
    document.querySelectorAll('.step').forEach(s => {
      s.classList.toggle('active', s.dataset.step === stepName);
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function setHistoryControls(n) {
    state.historyLen = n;
    document.getElementById('undoBtn').disabled = n === 0;
    document.getElementById('resetBtn').disabled = n === 0;
    document.getElementById('compareBtn').disabled = n === 0;
    document.getElementById('exportBar').hidden = n === 0;
    document.getElementById('historyCard').hidden = n === 0;
  }

  function renderHistory(history) {
    const el = document.getElementById('historyList');
    if (!history || !history.length) {
      el.innerHTML = '<div class="muted">No operations applied yet.</div>';
      return;
    }
    el.innerHTML = history.map((h, i) => `
      <div class="history-item">
        <span class="history-step">#${i + 1}</span>
        <span class="history-fam">${h.op.family}.${h.op.strategy}</span>
        <span class="history-msg">${escapeHtml(h.message || '')}</span>
      </div>
    `).join('');
  }

  async function refreshDashboard() {
    const p = await API.profile(state.sid);
    state.profile = p;
    Profile.renderKPIs(p);
    Profile.renderColumns(p, onColumnClick);

    // Viz workshop
    Viz.init(state.sid, p);

    // Cache current preview for the popup
    try {
      const prev = await API.preview(state.sid);
      state.currentPreview = prev.preview;
      state.currentRows = prev.rows;
      state.currentCols = prev.cols;
      document.getElementById('viewCurrentMeta').textContent =
        `${prev.rows.toLocaleString()} rows × ${prev.cols} cols · read-only`;
    } catch {}

    if (state.originalPreview) {
      document.getElementById('viewOriginalMeta').textContent =
        `${state.originalRows.toLocaleString()} rows × ${state.originalCols} cols · before any cleaning`;
    }

    // History
    try {
      const h = await API.history(state.sid);
      renderHistory(h.history);
      setHistoryControls(h.history.length);
    } catch {}

    // Suggestions
    try {
      const s = await API.suggest(state.sid);
      Suggest.render(s.suggestions, state.sid, refreshDashboard);
    } catch (e) { console.error('suggest failed', e); }
  }

  function onColumnClick(col) {
    Drawer.open(col, state.sid, async () => refreshDashboard());
  }

  async function onLoaded(data, source) {
    state.sid = data.sid;
    state.source = source;
    
    // Check if this is an image dataset
    if (data.mode === 'image') {
      // Image dataset flow
      goto('image-profile');
      document.getElementById('imageDashMeta').textContent = 
        `${source || 'Image dataset'} · session ${data.sid}`;
      
      // Render image profile
      const profile = data.profile;
      state.profile = profile;
      ImageProfile.reset();
      ImageProfile.renderKPIs(profile, data.sid);
      ImageProfile.renderQualityIssues(profile);
      ImageProfile.renderGrid(profile, data.sid);
      
      // Initialize cleaning panel
      ImageClean.init(data.sid, profile);
      
      // Wire up image-specific buttons
      document.getElementById('imageBackToUpload').onclick = () => {
        Object.assign(state, { sid: null, profile: null });
        goto('upload');
      };
      
      document.getElementById('imageExportBtn').onclick = () => {
        ImageExport.open({ sid: data.sid, profile: state.profile });
      };
      
      document.getElementById('imageUndoBtn').onclick = async () => {
        try {
          await API.undo(data.sid);
          showToast('Undid last operation', 'success');
          const newProfile = await API.imageProfile(data.sid);
          state.profile = newProfile;
          ImageProfile.reset();
          ImageProfile.renderKPIs(newProfile, data.sid);
          ImageProfile.renderQualityIssues(newProfile);
          ImageProfile.renderGrid(newProfile, data.sid);
        } catch (e) { showToast(e.message, 'error'); }
      };
      
      document.getElementById('imageResetBtn').onclick = async () => {
        if (!confirm('Reset to the original dataset? All applied operations will be discarded.')) return;
        try {
          await API.reset(data.sid);
          showToast('Reset to original', 'success');
          const newProfile = await API.imageProfile(data.sid);
          state.profile = newProfile;
          ImageProfile.reset();
          ImageProfile.renderKPIs(newProfile, data.sid);
          ImageProfile.renderQualityIssues(newProfile);
          ImageProfile.renderGrid(newProfile, data.sid);
        } catch (e) { showToast(e.message, 'error'); }
      };
      
      document.getElementById('imageMagicBtn').onclick = async () => {
        const btn = document.getElementById('imageMagicBtn');
        const orig = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<span class="loading"></span> Magic Clean…';
        try {
          const res = await API.imageMagic(data.sid);
          showToast(`✨ Magic Clean done — ${res.applied.length} ops applied`, 'success');
          
          // Refresh image profile
          const newProfile = await API.imageProfile(data.sid);
          ImageProfile.reset();
          ImageProfile.renderKPIs(newProfile, data.sid);
          ImageProfile.renderQualityIssues(newProfile);
          ImageProfile.renderGrid(newProfile, data.sid);
        } catch (e) {
          showToast(e.message, 'error');
        } finally {
          btn.disabled = false;
          btn.innerHTML = orig;
        }
      };
      
      return;
    }
    
    // Tabular dataset flow (existing)
    state.originalPreview = data.preview;
    state.originalRows = data.rows;
    state.originalCols = data.cols;
    goto('profile');
    document.getElementById('dashTitle').textContent = source || 'Dataset profile';
    document.getElementById('dashMeta').textContent =
      `${data.rows.toLocaleString()} rows × ${data.cols} cols · session ${data.sid}`;
    await refreshDashboard();
  }

  // Preview modal — read-only popup with full data table.
  function openPreviewModal(which) {
    const modal = document.getElementById('previewModal');
    const title = document.getElementById('previewModalTitle');
    const sub = document.getElementById('previewModalSub');
    let rows, totalRows, totalCols, label;
    if (which === 'original') {
      rows = state.originalPreview || [];
      totalRows = state.originalRows;
      totalCols = state.originalCols;
      label = 'Original data (read-only)';
    } else {
      rows = state.currentPreview || [];
      totalRows = state.currentRows;
      totalCols = state.currentCols;
      label = 'Current data (read-only)';
    }
    title.textContent = label;
    sub.textContent = `${(totalRows || 0).toLocaleString()} rows × ${totalCols || 0} cols total · showing first ${rows.length}`;
    Profile.renderPreview(rows, 'previewModalTable');
    modal.hidden = false;
  }
  document.getElementById('viewOriginalBtn').addEventListener('click', () => openPreviewModal('original'));
  document.getElementById('viewCurrentBtn').addEventListener('click', () => openPreviewModal('current'));
  document.getElementById('previewModalClose').addEventListener('click', () => {
    document.getElementById('previewModal').hidden = true;
  });
  document.getElementById('previewModal').addEventListener('click', (e) => {
    if (e.target.id === 'previewModal') e.currentTarget.hidden = true;
  });

  // --- top-bar actions ---

  document.getElementById('backToUpload').addEventListener('click', () => {
    Object.assign(state, { sid: null, profile: null, originalPreview: null });
    goto('upload');
  });

  document.getElementById('engineerBtn').addEventListener('click', () => {
    if (state.sid && state.profile) MLPrep.openEngineer({ sid: state.sid, profile: state.profile, onApplied: refreshDashboard });
  });
  document.getElementById('splitBtn').addEventListener('click', () => {
    if (state.sid && state.profile) MLPrep.openSplit({ sid: state.sid, profile: state.profile, onApplied: refreshDashboard });
  });
  document.getElementById('balanceBtn').addEventListener('click', () => {
    if (state.sid && state.profile) MLPrep.openBalance({ sid: state.sid, profile: state.profile, onApplied: refreshDashboard });
  });
  document.getElementById('reduceBtn').addEventListener('click', () => {
    if (state.sid && state.profile) MLPrep.openReduce({ sid: state.sid, profile: state.profile, onApplied: refreshDashboard });
  });
  document.getElementById('mergeBtn').addEventListener('click', () => {
    if (state.sid && state.profile) MergeModal.open({ sid: state.sid, profile: state.profile, onApplied: refreshDashboard });
  });

  document.getElementById('validateBtn').addEventListener('click', () => {
    if (!state.sid || !state.profile) return;
    ValidateModal.open({
      sid: state.sid, profile: state.profile,
      onApplied: refreshDashboard,
    });
  });

  document.getElementById('fixNanBtn').addEventListener('click', async () => {
    if (!state.sid) return;
    const btn = document.getElementById('fixNanBtn');
    const orig = btn.innerHTML;
    btn.disabled = true; btn.innerHTML = '<span class="loading"></span> Fixing NaN…';
    try {
      const res = await API.clean(state.sid, { family: 'missing', strategy: 'fill_all_smart' });
      showToast(res.message, 'success');
      await refreshDashboard();
    } catch (e) {
      showToast(e.message, 'error');
    } finally {
      btn.disabled = false; btn.innerHTML = orig;
    }
  });

  document.getElementById('magicBtn').addEventListener('click', async () => {
    if (!state.sid) return;
    const btn = document.getElementById('magicBtn');
    const orig = btn.innerHTML;
    btn.disabled = true; btn.innerHTML = '<span class="loading"></span> Magic Clean…';
    try {
      const res = await API.magic(state.sid);
      showToast(`✨ Magic Clean done — ${res.applied.length} ops applied`, 'success');
      await refreshDashboard();
    } catch (e) {
      showToast(e.message, 'error');
    } finally {
      btn.disabled = false; btn.innerHTML = orig;
    }
  });

  document.getElementById('undoBtn').addEventListener('click', async () => {
    try {
      await API.undo(state.sid);
      showToast('Undid last operation', 'success');
      await refreshDashboard();
    } catch (e) { showToast(e.message, 'error'); }
  });

  document.getElementById('resetBtn').addEventListener('click', async () => {
    if (!confirm('Reset to the original dataset? All applied operations will be discarded.')) return;
    try {
      await API.reset(state.sid);
      showToast('Reset to original', 'success');
      await refreshDashboard();
    } catch (e) { showToast(e.message, 'error'); }
  });

  document.getElementById('compareBtn').addEventListener('click', async () => {
    try {
      const data = await API.compare(state.sid);
      Compare.render(data);
      goto('compare');
    } catch (e) { showToast(e.message, 'error'); }
  });

  document.getElementById('backToProfile').addEventListener('click', () => goto('profile'));

  document.getElementById('downloadCsvBtn').addEventListener('click', () => {
    window.location.href = API.downloadCsvUrl(state.sid);
  });
  document.getElementById('downloadNbBtn').addEventListener('click', () => {
    window.location.href = API.downloadNotebookUrl(state.sid);
  });

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }

  Upload.bind(onLoaded);
})();

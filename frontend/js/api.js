// API wrapper around the Flask backend.
const API = {
  async _json(r) {
    if (!r.ok) {
      let msg = 'request failed';
      try { msg = (await r.json()).error || msg; } catch {}
      throw new Error(msg);
    }
    return r.json();
  },
  uploadFile(file) {
    const fd = new FormData();
    fd.append('file', file);
    return fetch('/api/upload', { method: 'POST', body: fd }).then(this._json);
  },
  loadUrl(url) {
    return fetch('/api/load-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    }).then(this._json);
  },
  profile(sid) { return fetch(`/api/profile/${sid}`).then(this._json); },
  preview(sid) { return fetch(`/api/preview/${sid}`).then(this._json); },
  clean(sid, op) {
    return fetch(`/api/clean/${sid}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(op),
    }).then(this._json);
  },
  magic(sid) { return fetch(`/api/magic/${sid}`, { method: 'POST' }).then(this._json); },
  history(sid) { return fetch(`/api/history/${sid}`).then(this._json); },
  undo(sid) { return fetch(`/api/undo/${sid}`, { method: 'POST' }).then(this._json); },
  reset(sid) { return fetch(`/api/reset/${sid}`, { method: 'POST' }).then(this._json); },
  compare(sid) { return fetch(`/api/compare/${sid}`).then(this._json); },
  suggest(sid) { return fetch(`/api/suggest/${sid}`).then(this._json); },
  labelGroups(sid, col, threshold = 0.85) {
    return fetch(`/api/label-groups/${sid}/${encodeURIComponent(col)}?threshold=${threshold}`).then(this._json);
  },
  previewOp(sid, op) {
    return fetch(`/api/preview-op/${sid}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(op),
    }).then(this._json);
  },
  column(sid, col) {
    return fetch(`/api/column/${sid}/${encodeURIComponent(col)}`).then(this._json);
  },
  correlation(sid) { return fetch(`/api/correlation/${sid}`).then(this._json); },
  scatter(sid, x, y) {
    const q = new URLSearchParams({ x, y });
    return fetch(`/api/scatter/${sid}?${q}`).then(this._json);
  },
  downloadCsvUrl(sid) { return `/api/download/${sid}`; },
  downloadNotebookUrl(sid) { return `/api/notebook/${sid}`; },
  
  // Image dataset APIs
  imageUpload(file) {
    const fd = new FormData();
    fd.append('file', file);
    return fetch('/api/image/upload', { method: 'POST', body: fd }).then(this._json);
  },
  imageProfile(sid) { return fetch(`/api/image/profile/${sid}`).then(this._json); },
  imageClean(sid, op) {
    return fetch(`/api/image/clean/${sid}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(op),
    }).then(this._json);
  },
  imageMagic(sid) { return fetch(`/api/image/magic/${sid}`, { method: 'POST' }).then(this._json); },
  imageExport(sid, formData) {
    return fetch(`/api/image/export/${sid}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData),
    }).then(async (r) => {
      if (!r.ok) {
        let msg = 'Export failed';
        try { msg = (await r.json()).error || msg; } catch {}
        throw new Error(msg);
      }
      return r.blob();
    });
  },
  imageThumbnailUrl(sid, imageId) { return `/api/image/thumbnail/${sid}/${imageId}`; },
  imageFullUrl(sid, imageId) { return `/api/image/full/${sid}/${imageId}`; },
};

function showToast(msg, kind = '') {
  const t = document.getElementById('toast');
  t.className = 'toast' + (kind ? ' ' + kind : '');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => { t.hidden = true; }, 3000);
}

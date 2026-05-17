// Upload screen wiring: drag/drop, file picker, URL load, sample chips.

const Upload = (() => {
  const dz = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const browseBtn = document.getElementById('browseBtn');
  const urlInput = document.getElementById('urlInput');
  const urlBtn = document.getElementById('urlBtn');
  const errBox = document.getElementById('uploadError');
  const samples = document.querySelectorAll('.chip');

  function showError(msg) {
    errBox.textContent = msg;
    errBox.hidden = false;
  }
  function clearError() {
    errBox.hidden = true;
    errBox.textContent = '';
  }

  function bind(onLoaded) {
    dz.addEventListener('click', () => fileInput.click());
    browseBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });

    ['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, (e) => {
      e.preventDefault(); dz.classList.add('drag');
    }));
    ['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, (e) => {
      e.preventDefault(); dz.classList.remove('drag');
    }));

    dz.addEventListener('drop', async (e) => {
      const f = e.dataTransfer.files?.[0];
      if (f) await handleFile(f, onLoaded);
    });
    fileInput.addEventListener('change', async (e) => {
      const f = e.target.files?.[0];
      if (f) await handleFile(f, onLoaded);
    });

    urlBtn.addEventListener('click', async () => {
      const u = urlInput.value.trim();
      if (!u) { showError('Enter a CSV URL'); return; }
      await handleUrl(u, onLoaded);
    });
    urlInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') urlBtn.click();
    });

    samples.forEach(s => s.addEventListener('click', async () => {
      const url = s.dataset.sample;
      if (url.toLowerCase().endsWith('.zip')) {
        // Sample is an image ZIP — download it, then run the normal
        // image-upload flow (handleFile auto-routes ZIPs to /api/image/upload).
        await handleSampleZip(url, s, onLoaded);
      } else {
        urlInput.value = url;
        await handleUrl(url, onLoaded);
      }
    }));
  }

  async function handleSampleZip(url, chipEl, onLoaded) {
    clearError();
    const orig = chipEl.innerHTML;
    chipEl.disabled = true;
    chipEl.innerHTML = '<span class="loading"></span> downloading…';
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to fetch sample (HTTP ${response.status})`);
      }
      const blob = await response.blob();
      const filename = url.split('/').pop() || 'demo_images.zip';
      const file = new File([blob], filename, { type: 'application/zip' });
      await handleFile(file, onLoaded);
    } catch (e) {
      showError(e.message);
    } finally {
      chipEl.disabled = false;
      chipEl.innerHTML = orig;
    }
  }

  async function handleFile(file, onLoaded) {
    clearError();
    const fileName = file.name.toLowerCase();
    
    // Check file type
    if (!fileName.endsWith('.csv') && !fileName.endsWith('.zip')) {
      showError('Only .csv and .zip files are supported.');
      return;
    }
    
    setBusy(true);
    const sizeMB = file.size / (1024 * 1024);
    showProgress(0, `Uploading ${file.name} (${sizeMB.toFixed(2)} MB)…`);

    try {
      let data;
      const endpoint = fileName.endsWith('.zip') ? '/api/image/upload' : '/api/upload';
      const processingLabel = fileName.endsWith('.zip')
        ? 'Processing images on server (extract + profile + hash)…'
        : 'Processing CSV on server (parse + auto-profile)…';

      data = await uploadWithProgress(
        file, endpoint,
        (percent) => showProgress(percent, `Uploading ${file.name}…`),
        () => showProcessing(processingLabel),
      );

      if (fileName.endsWith('.zip')) {
        data.mode = 'image';
        if (data.profile && data.profile.has_labels) {
          showToast('✓ labels.csv detected', 'success');
        } else {
          showToast('ℹ No labels.csv found', 'info');
        }
      } else {
        data.mode = 'tabular';
      }

      hideProgress();
      onLoaded(data, file.name);
    } catch (e) {
      hideProgress();
      showError(e.message);
    } finally {
      setBusy(false);
    }
  }

  function uploadWithProgress(file, endpoint, onProgress, onUploadDone) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const formData = new FormData();
      formData.append('file', file);

      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          const percent = Math.round((e.loaded / e.total) * 100);
          onProgress(percent);
        }
      });

      // Fires when the browser is done sending bytes but the server is
      // still working. This is where image ZIPs spend most of their time.
      xhr.upload.addEventListener('load', () => {
        onProgress(100);
        if (onUploadDone) onUploadDone();
      });

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const data = JSON.parse(xhr.responseText);
            resolve(data);
          } catch (e) {
            reject(new Error('Invalid response from server'));
          }
        } else {
          try {
            const err = JSON.parse(xhr.responseText);
            reject(new Error(err.error || 'Upload failed'));
          } catch (e) {
            reject(new Error(`Upload failed with status ${xhr.status}`));
          }
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('Network error during upload'));
      });

      xhr.addEventListener('abort', () => {
        reject(new Error('Upload cancelled'));
      });

      xhr.open('POST', endpoint);
      xhr.send(formData);
    });
  }

  // ---------- two-phase progress UI ----------
  // Phase 1 (showProgress): real upload bytes 0..100%.
  // Phase 2 (showProcessing): server is working after the upload finished —
  //                           bar shows an indeterminate shimmer until the
  //                           response arrives.

  function showProgress(percent, label) {
    const progressDiv = document.getElementById('uploadProgress');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const progressLabel = document.getElementById('progressLabel');

    progressDiv.hidden = false;
    progressDiv.classList.remove('processing');
    progressBar.style.width = `${percent}%`;
    progressText.textContent = `${percent}%`;
    if (label) progressLabel.textContent = label;
  }

  function showProcessing(label) {
    const progressDiv = document.getElementById('uploadProgress');
    const progressText = document.getElementById('progressText');
    const progressLabel = document.getElementById('progressLabel');

    progressDiv.hidden = false;
    progressDiv.classList.add('processing');
    progressText.textContent = 'working…';
    progressLabel.textContent = label || 'Processing on server…';
  }

  function hideProgress() {
    const progressDiv = document.getElementById('uploadProgress');
    progressDiv.hidden = true;
    progressDiv.classList.remove('processing');
  }

  async function handleUrl(url, onLoaded) {
    clearError();
    const urlLower = url.toLowerCase();
    
    // Check if URL is for ZIP (image dataset) - currently not supported via URL
    if (urlLower.endsWith('.zip')) {
      showError('ZIP files must be uploaded directly (drag & drop or browse).');
      return;
    }
    
    setBusy(true);
    try {
      const data = await API.loadUrl(url);
      data.mode = 'tabular';
      onLoaded(data, url.split('/').pop());
    } catch (e) {
      showError(e.message);
    } finally {
      setBusy(false);
    }
  }

  function setBusy(b) {
    dz.style.opacity = b ? 0.6 : 1;
    urlBtn.disabled = b;
    urlBtn.innerHTML = b ? '<span class="loading"></span> Loading…' : 'Load CSV';
  }

  return { bind };
})();

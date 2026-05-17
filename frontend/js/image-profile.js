/**
 * Image dataset profile rendering
 */
const ImageProfile = {
  currentSid: null,
  currentProfile: null,
  displayedImages: 0,
  imagesPerPage: 60,

  /**
   * Render KPI cards from profile
   */
  renderKPIs(profile, sid) {
    this.currentSid = sid;
    this.currentProfile = profile;

    // Total images
    document.getElementById('imageKpiTotal').textContent = profile.total_images || 0;

    // Formats
    const formats = profile.formats || {};
    const formatList = Object.keys(formats).join(', ') || 'N/A';
    document.getElementById('imageKpiFormats').textContent = formatList;

    // Average dimensions
    const avgW = Math.round(profile.avg_width || 0);
    const avgH = Math.round(profile.avg_height || 0);
    document.getElementById('imageKpiDims').textContent = `${avgW}×${avgH}`;

    // Duplicates — show "—" if hashes haven't been computed yet (initial state)
    const dupsEl = document.getElementById('imageKpiDups');
    if (profile.duplicates_pending) {
      dupsEl.textContent = '—';
      dupsEl.title = 'Run Magic Clean or "Compute hashes" to detect duplicates';
    } else {
      dupsEl.textContent = profile.duplicates || 0;
      dupsEl.title = '';
    }

    // Quality score — backend returns `grade` (not `quality_grade`).
    // The CSS grade pill uses [data-g="A"] selectors, so set dataset.g.
    const score = profile.quality_score || 0;
    const grade = (profile.grade || profile.quality_grade || 'F').toString().toUpperCase();
    document.getElementById('imageKpiScore').textContent = score;
    const gradeEl = document.getElementById('imageKpiGrade');
    gradeEl.textContent = grade;
    gradeEl.className = 'grade';
    gradeEl.dataset.g = grade;

    // Meta info
    const meta = `${profile.total_images || 0} images · ${Object.keys(formats).length} format(s)`;
    document.getElementById('imageDashMeta').textContent = meta;
  },

  /**
   * Render quality issues list
   */
  renderQualityIssues(profile) {
    const container = document.getElementById('qualityIssuesList');
    container.innerHTML = '';

    const issues = [];

    // Collect warnings from profile
    if (profile.warnings && profile.warnings.length > 0) {
      profile.warnings.forEach(w => {
        issues.push({ icon: '⚠️', text: w });
      });
    }

    // Aggregate quality flags from images
    const flagCounts = {};
    if (profile.images && Array.isArray(profile.images)) {
      profile.images.forEach(img => {
        if (img.quality_flags && Array.isArray(img.quality_flags)) {
          img.quality_flags.forEach(flag => {
            flagCounts[flag] = (flagCounts[flag] || 0) + 1;
          });
        }
      });
    }

    // Add flag-based issues
    if (flagCounts.blur) {
      issues.push({ icon: '🌫️', text: `${flagCounts.blur} blurry image${flagCounts.blur > 1 ? 's' : ''}` });
    }
    if (flagCounts.overexposed) {
      issues.push({ icon: '☀️', text: `${flagCounts.overexposed} overexposed image${flagCounts.overexposed > 1 ? 's' : ''}` });
    }
    if (flagCounts.underexposed) {
      issues.push({ icon: '🌙', text: `${flagCounts.underexposed} underexposed image${flagCounts.underexposed > 1 ? 's' : ''}` });
    }
    if (flagCounts.corrupt) {
      issues.push({ icon: '❌', text: `${flagCounts.corrupt} corrupt file${flagCounts.corrupt > 1 ? 's' : ''} skipped` });
    }

    // Duplicates
    if (profile.duplicates && profile.duplicates > 0) {
      issues.push({ icon: '🔁', text: `${profile.duplicates} duplicate image${profile.duplicates > 1 ? 's' : ''}` });
    }

    // Render issues
    if (issues.length === 0) {
      container.innerHTML = '<div class="quality-issue-empty">✅ No quality issues detected</div>';
    } else {
      issues.forEach(issue => {
        const div = document.createElement('div');
        div.className = 'quality-issue-item';
        div.innerHTML = `<span class="quality-issue-icon">${issue.icon}</span><span>${issue.text}</span>`;
        container.appendChild(div);
      });
    }
  },

  /**
   * Render image grid with thumbnails
   */
  renderGrid(profile, sid) {
    const container = document.getElementById('imageGrid');
    const loadMoreBtn = document.getElementById('loadMoreBtn');
    
    if (!profile.images || profile.images.length === 0) {
      container.innerHTML = '<div class="empty-state">No images to display</div>';
      loadMoreBtn.hidden = true;
      return;
    }

    // Initial render or append more
    const startIdx = this.displayedImages;
    const endIdx = Math.min(startIdx + this.imagesPerPage, profile.images.length);
    const imagesToRender = profile.images.slice(startIdx, endIdx);

    imagesToRender.forEach(img => {
      const tile = document.createElement('div');
      tile.className = 'image-tile';
      tile.dataset.imageId = img.image_id;

      // Thumbnail
      const imgEl = document.createElement('img');
      imgEl.src = `/api/image/thumbnail/${sid}/${img.image_id}`;
      imgEl.loading = 'lazy';
      imgEl.alt = img.filename;

      // Metadata
      const meta = document.createElement('div');
      meta.className = 'image-tile-meta';

      const filename = document.createElement('div');
      filename.className = 'filename';
      filename.textContent = img.filename;
      filename.title = img.filename;

      const dims = document.createElement('div');
      dims.className = 'dims';
      dims.textContent = `${img.width}×${img.height} · ${img.format} · ${img.mode}`;

      const badges = document.createElement('div');
      badges.className = 'badges';
      if (img.quality_flags && img.quality_flags.length > 0) {
        img.quality_flags.forEach(flag => {
          const badge = document.createElement('span');
          badge.className = `quality-badge quality-badge-${flag}`;
          badge.textContent = flag;
          badges.appendChild(badge);
        });
      }

      meta.appendChild(filename);
      meta.appendChild(dims);
      meta.appendChild(badges);

      tile.appendChild(imgEl);
      tile.appendChild(meta);

      // Click handler - show full image
      tile.addEventListener('click', () => {
        this.showFullImage(sid, img);
      });

      container.appendChild(tile);
    });

    this.displayedImages = endIdx;

    // Show/hide load more button
    if (this.displayedImages < profile.images.length) {
      loadMoreBtn.hidden = false;
      loadMoreBtn.onclick = () => this.renderGrid(profile, sid);
    } else {
      loadMoreBtn.hidden = true;
    }
  },

  /**
   * Show full-resolution image in modal
   */
  showFullImage(sid, img) {
    // Create modal
    const backdrop = document.createElement('div');
    backdrop.className = 'modal-backdrop full-image-modal';
    backdrop.style.display = 'flex';

    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.style.maxWidth = '90vw';
    modal.style.maxHeight = '90vh';
    modal.style.overflow = 'auto';

    const head = document.createElement('div');
    head.className = 'modal-head';
    head.innerHTML = `
      <div>
        <div class="modal-title">${img.filename}</div>
        <div class="muted">${img.width}×${img.height} · ${img.format} · ${img.mode}</div>
      </div>
      <button class="btn ghost close-modal">✕</button>
    `;

    const body = document.createElement('div');
    body.className = 'modal-body';
    body.style.textAlign = 'center';

    const fullImg = document.createElement('img');
    fullImg.src = `/api/image/full/${sid}/${img.image_id}`;
    fullImg.style.maxWidth = '100%';
    fullImg.style.maxHeight = '80vh';
    fullImg.style.objectFit = 'contain';

    body.appendChild(fullImg);
    modal.appendChild(head);
    modal.appendChild(body);
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);

    // Close handlers
    const closeBtn = head.querySelector('.close-modal');
    closeBtn.onclick = () => backdrop.remove();
    backdrop.onclick = (e) => {
      if (e.target === backdrop) backdrop.remove();
    };
  },

  /**
   * Reset display state
   */
  reset() {
    this.currentSid = null;
    this.currentProfile = null;
    this.displayedImages = 0;
    document.getElementById('imageGrid').innerHTML = '';
  }
};

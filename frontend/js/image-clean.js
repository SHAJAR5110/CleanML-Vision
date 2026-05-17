/**
 * image-clean.js
 * Inline cleaning operations panel for image datasets
 */

const ImageClean = (() => {
  let currentSid = null;
  let currentProfile = null;

  // Operation definitions matching backend modules
  const OPERATIONS = {
    quality: {
      remove_blurry: {
        params: [
          { name: 'threshold', type: 'number', default: 100, label: 'Blur threshold' }
        ]
      },
      remove_corrupt: {
        params: []
      },
      flag_low_quality: {
        params: []
      }
    },
    dedup: {
      compute_hashes: {
        params: []
      },
      remove_duplicates: {
        params: [
          { name: 'threshold', type: 'number', default: 5, label: 'Hash distance threshold' }
        ]
      }
    },
    transforms: {
      resize: {
        params: [
          { name: 'width', type: 'number', default: 224, label: 'Width (px)' },
          { name: 'height', type: 'number', default: 224, label: 'Height (px)' },
          { name: 'mode', type: 'select', default: 'pad', label: 'Resize mode', options: ['stretch', 'pad', 'crop'] }
        ]
      },
      convert_color: {
        params: [
          { name: 'mode', type: 'select', default: 'RGB', label: 'Color mode', options: ['RGB', 'L', 'RGBA'] }
        ]
      },
      normalize: {
        params: [
          { name: 'method', type: 'select', default: 'imagenet', label: 'Normalization method', options: ['imagenet', '0-1', 'z-score'] }
        ]
      },
      center_crop: {
        params: [
          { name: 'width', type: 'number', default: 224, label: 'Width (px)' },
          { name: 'height', type: 'number', default: 224, label: 'Height (px)' }
        ]
      }
    },
    augment: {
      rotate: {
        params: [
          { name: 'angle', type: 'select', default: '90', label: 'Rotation angle', options: ['90', '180', '270', 'random'] }
        ]
      },
      flip: {
        params: [
          { name: 'direction', type: 'select', default: 'horizontal', label: 'Flip direction', options: ['horizontal', 'vertical', 'both'] }
        ]
      },
      brightness: {
        params: [
          { name: 'factor', type: 'number', default: 1.2, label: 'Brightness factor', step: 0.1 }
        ]
      },
      contrast: {
        params: [
          { name: 'factor', type: 'number', default: 1.2, label: 'Contrast factor', step: 0.1 }
        ]
      },
      random_crop: {
        params: [
          { name: 'width', type: 'number', default: 224, label: 'Width (px)' },
          { name: 'height', type: 'number', default: 224, label: 'Height (px)' },
          { name: 'count', type: 'number', default: 5, label: 'Number of crops' }
        ]
      }
    },
    pair: {
      join_labels: {
        params: [
          { name: 'image_col', type: 'text', default: 'filename', label: 'Image column name' },
          { name: 'label_col', type: 'text', default: 'label', label: 'Label column name' }
        ]
      },
      filter_by_label: {
        params: [
          { name: 'labels', type: 'text', default: '', label: 'Labels (comma-separated)' }
        ]
      },
      balance_classes: {
        params: [
          { name: 'method', type: 'select', default: 'undersample', label: 'Balance method', options: ['undersample', 'oversample'] }
        ]
      },
      stratified_split: {
        params: [
          { name: 'test_size', type: 'number', default: 0.2, label: 'Test size (0.05-0.5)', step: 0.05, min: 0.05, max: 0.5 },
          { name: 'target_col', type: 'text', default: 'label', label: 'Target column' }
        ]
      }
    }
  };

  function init(sid, profile) {
    currentSid = sid;
    currentProfile = profile;

    const catSelect = document.getElementById('imageCleanCat');
    const stratSelect = document.getElementById('imageCleanStrat');
    const paramsDiv = document.getElementById('imageCleanParams');
    const applyBtn = document.getElementById('imageCleanApply');
    const statusDiv = document.getElementById('imageCleanStatus');

    // Check if labels are available to enable/disable pair category
    const hasLabels = profile.labels_detected || false;
    const pairOption = catSelect.querySelector('option[value="pair"]');
    if (pairOption) {
      pairOption.disabled = !hasLabels;
      if (!hasLabels) {
        pairOption.textContent = 'Pair with labels (no labels detected)';
      }
    }

    // Category change handler
    catSelect.addEventListener('change', () => {
      const category = catSelect.value;
      stratSelect.innerHTML = '<option value="">Select operation...</option>';
      paramsDiv.innerHTML = '';
      applyBtn.disabled = true;

      if (category && OPERATIONS[category]) {
        stratSelect.disabled = false;
        const strategies = OPERATIONS[category];
        Object.keys(strategies).forEach(strat => {
          const opt = document.createElement('option');
          opt.value = strat;
          opt.textContent = strat.replace(/_/g, ' ');
          stratSelect.appendChild(opt);
        });
      } else {
        stratSelect.disabled = true;
      }
    });

    // Strategy change handler
    stratSelect.addEventListener('change', () => {
      const category = catSelect.value;
      const strategy = stratSelect.value;
      paramsDiv.innerHTML = '';
      applyBtn.disabled = !strategy;

      if (category && strategy && OPERATIONS[category][strategy]) {
        const opDef = OPERATIONS[category][strategy];
        renderParams(opDef.params, paramsDiv);
      }
    });

    // Apply button handler
    applyBtn.addEventListener('click', async () => {
      const category = catSelect.value;
      const strategy = stratSelect.value;
      if (!category || !strategy) return;

      const params = collectParams(paramsDiv);
      await applyOperation(category, strategy, params, statusDiv);
    });
  }

  function renderParams(paramDefs, container) {
    paramDefs.forEach(param => {
      const wrapper = document.createElement('div');
      wrapper.style.display = 'flex';
      wrapper.style.flexDirection = 'column';
      wrapper.style.gap = '4px';

      const label = document.createElement('label');
      label.textContent = param.label;
      label.style.fontSize = '12px';
      label.style.color = '#999';
      wrapper.appendChild(label);

      let input;
      if (param.type === 'select') {
        input = document.createElement('select');
        input.className = 'clean-input';
        param.options.forEach(opt => {
          const option = document.createElement('option');
          option.value = opt;
          option.textContent = opt;
          if (opt === param.default) option.selected = true;
          input.appendChild(option);
        });
      } else if (param.type === 'number') {
        input = document.createElement('input');
        input.type = 'number';
        input.className = 'clean-input';
        input.value = param.default;
        if (param.step) input.step = param.step;
        if (param.min !== undefined) input.min = param.min;
        if (param.max !== undefined) input.max = param.max;
      } else {
        input = document.createElement('input');
        input.type = 'text';
        input.className = 'clean-input';
        input.value = param.default || '';
        if (param.placeholder) input.placeholder = param.placeholder;
      }

      input.dataset.paramName = param.name;
      wrapper.appendChild(input);
      container.appendChild(wrapper);
    });
  }

  function collectParams(container) {
    const params = {};
    const inputs = container.querySelectorAll('[data-param-name]');
    inputs.forEach(input => {
      const name = input.dataset.paramName;
      let value = input.value;
      
      // Convert to appropriate type
      if (input.type === 'number') {
        value = parseFloat(value);
      }
      
      params[name] = value;
    });
    return params;
  }

  async function applyOperation(category, strategy, params, statusDiv) {
    const applyBtn = document.getElementById('imageCleanApply');
    applyBtn.disabled = true;
    applyBtn.textContent = 'Applying...';
    statusDiv.textContent = 'Processing...';
    statusDiv.className = 'clean-status muted';

    try {
      const family = `image_${category}`;
      const result = await API.imageClean(currentSid, { family, strategy, params });

      if (result.success) {
        statusDiv.textContent = result.message || 'Operation applied successfully';
        statusDiv.className = 'clean-status success';
        showToast(result.message || 'Operation applied', 'success');

        // Refetch profile and update UI
        const newProfile = await API.imageProfile(currentSid);
        currentProfile = newProfile;

        // Re-render every panel with the fresh profile
        if (typeof ImageProfile.reset === 'function') ImageProfile.reset();
        ImageProfile.renderKPIs(newProfile, currentSid);
        ImageProfile.renderQualityIssues(newProfile);
        ImageProfile.renderGrid(newProfile, currentSid);

        // Enable undo/reset buttons
        document.getElementById('imageUndoBtn').disabled = false;
        document.getElementById('imageResetBtn').disabled = false;
      } else {
        throw new Error(result.error || 'Operation failed');
      }
    } catch (err) {
      console.error('Clean operation error:', err);
      statusDiv.textContent = `Error: ${err.message}`;
      statusDiv.className = 'clean-status error';
      showToast(`Error: ${err.message}`, 'error');
    } finally {
      applyBtn.disabled = false;
      applyBtn.textContent = 'Apply';
    }
  }

  function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.hidden = false;
    setTimeout(() => {
      toast.hidden = true;
    }, 3000);
  }

  return { init };
})();
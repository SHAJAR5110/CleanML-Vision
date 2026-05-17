/**
 * image-export.js
 * Export modal for image datasets
 */

const ImageExport = (() => {
  let currentSid = null;
  let currentProfile = null;

  function open({ sid, profile }) {
    currentSid = sid;
    currentProfile = profile;

    const modal = document.getElementById('imageExportModal');
    const formatSelect = document.getElementById('imageExportFormat');
    const labelsCheckbox = document.getElementById('imageExportLabels');
    const splitCheckbox = document.getElementById('imageExportSplit');
    const splitOptions = document.getElementById('imageExportSplitOptions');
    const testSizeSlider = document.getElementById('imageExportTestSize');
    const testSizeVal = document.getElementById('imageExportTestSizeVal');
    const targetColInput = document.getElementById('imageExportTargetCol');
    const submitBtn = document.getElementById('imageExportSubmit');
    const cancelBtn = document.getElementById('imageExportCancel');
    const closeBtn = document.getElementById('imageExportClose');

    // Check if labels are available
    const hasLabels = profile.labels_detected || false;
    labelsCheckbox.disabled = !hasLabels;
    if (!hasLabels) {
      labelsCheckbox.checked = false;
      labelsCheckbox.parentElement.style.opacity = '0.5';
      labelsCheckbox.parentElement.title = 'No labels detected in dataset';
    } else {
      labelsCheckbox.parentElement.style.opacity = '1';
      labelsCheckbox.parentElement.title = '';
    }

    // Split toggle handler
    splitCheckbox.addEventListener('change', () => {
      splitOptions.hidden = !splitCheckbox.checked;
    });

    // Test size slider handler
    testSizeSlider.addEventListener('input', () => {
      testSizeVal.textContent = `${testSizeSlider.value}%`;
    });

    // Submit handler
    const handleSubmit = async () => {
      submitBtn.disabled = true;
      submitBtn.textContent = 'Exporting...';

      try {
        const formData = {
          format: formatSelect.value,
          include_labels: labelsCheckbox.checked,
          split: splitCheckbox.checked
        };

        if (splitCheckbox.checked) {
          formData.test_size = parseFloat(testSizeSlider.value) / 100;
          formData.target_col = targetColInput.value.trim() || 'label';
        }

        const blob = await API.imageExport(currentSid, formData);
        
        // Trigger download
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        
        // Determine filename based on format
        let filename = 'dataset';
        if (formData.format === 'zip') {
          filename += '.zip';
        } else if (formData.format === 'numpy') {
          filename += '.npy';
        } else if (formData.format === 'pytorch') {
          filename += '.pt';
        }
        
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        showToast('Export completed successfully', 'success');
        close();
      } catch (err) {
        console.error('Export error:', err);
        showToast(`Export failed: ${err.message}`, 'error');
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Export';
      }
    };

    // Wire up buttons
    submitBtn.onclick = handleSubmit;
    cancelBtn.onclick = close;
    closeBtn.onclick = close;

    // Show modal
    modal.hidden = false;
  }

  function close() {
    const modal = document.getElementById('imageExportModal');
    modal.hidden = true;

    // Reset form
    document.getElementById('imageExportFormat').value = 'zip';
    document.getElementById('imageExportLabels').checked = false;
    document.getElementById('imageExportSplit').checked = false;
    document.getElementById('imageExportSplitOptions').hidden = true;
    document.getElementById('imageExportTestSize').value = 20;
    document.getElementById('imageExportTestSizeVal').textContent = '20%';
    document.getElementById('imageExportTargetCol').value = '';
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

  return { open, close };
})();
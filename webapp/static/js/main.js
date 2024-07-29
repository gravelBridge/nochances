document.addEventListener('DOMContentLoaded', function() {
    const donateModal = new bootstrap.Modal(document.getElementById('donateModal'));
    donateModal.show();

    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    })
    const form = document.getElementById('predictionForm');
    const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
    const resultsModal = new bootstrap.Modal(document.getElementById('resultsModal'));

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        loadingModal.show();

        fetch(form.action, {
            method: 'POST',
            body: new FormData(form),
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        })
        .then(response => response.json().then(data => ({status: response.status, body: data})))
        .then(({status, body}) => {
            loadingModal.hide();
            if (status === 200) {
                displayResults(body);
                resultsModal.show();
            } else {
                displayErrors(body);
            }
        })
        .catch(error => {
            loadingModal.hide();
            alert('An error occurred. Please try again.');
            console.error('Error:', error);
        })
        .finally(() => {
            resetHCaptcha(); // Reset hCaptcha after form submission
        });
    });

    function displayResults(data) {
        const resultsModalBody = document.getElementById('resultsModalBody');
        resultsModalBody.innerHTML = `
            <div class="prediction-result" style="background-color: ${data.color};">
                <p>Probability of Acceptance:<br><strong>${(data.acceptance_probability * 100).toFixed(2)}%</strong></p>
            </div>
            <div class="result-item">
                <span class="result-label">School Category:</span> ${data.school_category}
            </div>
            <div class="result-item">
                <span class="result-label">Ensemble Prediction:</span> ${data.nn_prediction.toFixed(2)}
            </div>
        `;
        resultsModalBody.classList.add('results');
    }

    function displayErrors(data) {
        let errorMessage = 'Please correct the following errors:\n';
        if (data.error) {
            errorMessage += data.error;
        } else if (data.errors) {
            for (const [field, errors] of Object.entries(data.errors)) {
                errorMessage += `${field}: ${errors.join(', ')}\n`;
            }
        }
        alert(errorMessage);
    }

    function resetHCaptcha() {
        if (typeof hcaptcha !== 'undefined') {
            hcaptcha.reset();
        }
    }
});
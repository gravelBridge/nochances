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

    let donationQueue = [];
    let requestsLeft = 0;
    let lastCheckTimestamp = Math.floor(Date.now() / 1000);

    function updateRequestCount(count) {
        requestsLeft = count;
        const display = document.getElementById('requestCountDisplay');
        display.textContent = `${requestsLeft} requests left`;
        
        if (requestsLeft > 1000) {
            display.style.color = '#28a745';
            display.textContent += ' 😄';
        } else if (requestsLeft > 500) {
            display.style.color = '#ffc107';
            display.textContent += ' 😐';
        } else {
            display.style.color = '#dc3545';
            display.textContent += ' 😟';
        }
    }

    function showDonationNotification(donation) {
        const notification = document.getElementById('donationNotification');
        notification.textContent = `${donation.name} donated $${donation.amount}: ${donation.message}`;
        notification.style.display = 'block';
        
        setTimeout(() => {
            notification.style.display = 'none';
            if (donationQueue.length > 0) {
                showDonationNotification(donationQueue.shift());
            }
        }, 10000);
    }

    function checkForUpdates() {
        fetch('/get_updates')
            .then(response => response.json())
            .then(data => {
                updateRequestCount(data.requests_left);
                data.donations.forEach(donation => {
                    if (donation.timestamp > lastCheckTimestamp) {
                        donationQueue.push(donation);
                    }
                });
                lastCheckTimestamp = Math.floor(Date.now() / 1000);
                if (donationQueue.length > 0 && !document.getElementById('donationNotification').style.display) {
                    showDonationNotification(donationQueue.shift());
                }
            });
    }

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
                <span class="result-label">Ensemble Prediction:</span> ${data.ensemble_prediction.toFixed(2)}
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

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        loadingModal.show();

        // Decrement the request count immediately for responsive UI
        updateRequestCount(requestsLeft - 1);

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
            resetHCaptcha();
            checkForUpdates(); // Update the request count after submission
        });
    });

    // Initial update check
    checkForUpdates();

    // Set up periodic checks
    setInterval(checkForUpdates, 60000); // Check every minute
});
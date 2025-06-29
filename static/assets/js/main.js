document.getElementById('predict-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const data = {};
    formData.forEach((value, key) => { data[key] = value; });

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: new URLSearchParams(data),
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
        });
        const result = await response.json();
        const resultDiv = document.getElementById('result');
        resultDiv.classList.remove('d-none', 'alert-success', 'alert-danger');
        if (result.error) {
            resultDiv.classList.add('alert-danger');
            resultDiv.innerText = `Error: ${result.error}`;
        } else {
            resultDiv.classList.add('alert-success');
            resultDiv.innerText = `Prediction: ${result.prediction}`;
        }
    } catch (error) {
        const resultDiv = document.getElementById('result');
        resultDiv.classList.remove('d-none', 'alert-success');
        resultDiv.classList.add('alert-danger');
        resultDiv.innerText = `Error: ${error.message}`;
    }
});
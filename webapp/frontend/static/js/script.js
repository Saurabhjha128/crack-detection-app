document.getElementById('upload-form').addEventListener('submit', async function (e) {
    e.preventDefault();
    const formData = new FormData(this);
    const resultDiv = document.getElementById('result');
    const outputImage = document.getElementById('output-image');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.error) {
            resultDiv.innerHTML = `<p class="text-danger">Error: ${data.error}</p>`;
            outputImage.style.display = 'none';
        } else {
            resultDiv.innerHTML = `<p>Prediction: ${data.label} (Confidence: ${data.confidence})</p>`;
            if (data.image_url) {
                outputImage.src = data.image_url;
                outputImage.style.display = 'block';
            } else {
                outputImage.style.display = 'none';
            }
        }
    } catch (error) {
        resultDiv.innerHTML = `<p class="text-danger">Error: ${error.message}</p>`;
        outputImage.style.display = 'none';
    }
});
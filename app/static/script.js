document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const text = document.getElementById('inputText').value;
    const resultArea = document.getElementById('resultArea');
    const sentimentDisplay = document.getElementById('sentimentDisplay');
    const loader = document.getElementById('loader');

    if (!text.trim()) {
        alert("Please enter some text.");
        return;
    }

    // UI Reset
    resultArea.classList.remove('hidden');
    sentimentDisplay.innerHTML = '';
    loader.classList.remove('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();
        
        loader.classList.add('hidden');

        if (response.ok) {
            displayResults(data.sentiment);
        } else {
            sentimentDisplay.innerHTML = `<p style="color: red;">Error: ${data.detail || 'Unknown error'}</p>`;
        }

    } catch (error) {
        loader.classList.add('hidden');
        sentimentDisplay.innerHTML = `<p style="color: red;">Network Error: ${error.message}</p>`;
    }
});

function displayResults(sentimentData) {
    const display = document.getElementById('sentimentDisplay');
    
    // The pipeline usually returns a list of lists: [[{label, score}, ...]]
    const scores = sentimentData[0]; // Access the first (and only) prediction set

    // Sort by score descending
    scores.sort((a, b) => b.score - a.score);

    scores.forEach(item => {
        const container = document.createElement('div');
        container.className = 'sentiment-bar-container';

        const percentage = (item.score * 100).toFixed(1);
        const labelClass = item.label.toLowerCase(); // for css coloring

        container.innerHTML = `
            <div class="label-row">
                <span>${item.label}</span>
                <span>${percentage}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill ${labelClass}" style="width: ${percentage}%"></div>
            </div>
        `;
        display.appendChild(container);
    });
}

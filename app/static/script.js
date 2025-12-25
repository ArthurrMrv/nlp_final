// Helper to create sentiment bars
function createSentimentBars(sentimentData) {
    const container = document.createElement('div');
    
    // Sort by score descending
    const scores = [...sentimentData].sort((a, b) => b.score - a.score);

    scores.forEach(item => {
        const row = document.createElement('div');
        row.className = 'sentiment-bar-container';

        const percentage = (item.score * 100).toFixed(1);
        const labelClass = item.label.toLowerCase(); 

        row.innerHTML = `
            <div class="label-row">
                <span>${item.label}</span>
                <span>${percentage}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill ${labelClass}" style="width: ${percentage}%"></div>
            </div>
        `;
        container.appendChild(row);
    });
    return container;
}

// --- Custom Text Analysis ---
document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const text = document.getElementById('inputText').value;
    const resultArea = document.getElementById('resultArea');
    const sentimentDisplay = document.getElementById('sentimentDisplay');
    const loader = document.getElementById('loader');

    if (!text.trim()) {
        alert("Please enter some text.");
        return;
    }

    resultArea.classList.remove('hidden');
    sentimentDisplay.innerHTML = '';
    loader.classList.remove('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ text: text })
        });
        const data = await response.json();
        loader.classList.add('hidden');

        if (response.ok) {
            // data.sentiment is usually [[{label, score}, ...]]
            const bars = createSentimentBars(data.sentiment[0]);
            sentimentDisplay.appendChild(bars);
        } else {
            sentimentDisplay.innerHTML = `<p style="color: red;">Error: ${data.detail || 'Unknown error'}</p>`;
        }
    } catch (error) {
        loader.classList.add('hidden');
        sentimentDisplay.innerHTML = `<p style="color: red;">Network Error: ${error.message}</p>`;
    }
});

// --- Ticker Analysis ---
document.getElementById('analyzeTickerBtn').addEventListener('click', async () => {
    const ticker = document.getElementById('tickerSelect').value;
    const resultArea = document.getElementById('tickerResultArea');
    const display = document.getElementById('tickerDisplay');
    const loader = document.getElementById('tickerLoader');

    resultArea.classList.remove('hidden');
    display.innerHTML = '';
    loader.classList.remove('hidden');

    try {
        const response = await fetch('/analyze_ticker', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ symbol: ticker })
        });
        const json = await response.json();
        loader.classList.add('hidden');

        if (response.ok) {
            if (!json.data || json.data.length === 0) {
                display.innerHTML = '<p>No recent news found for this ticker.</p>';
                return;
            }

            json.data.forEach(item => {
                const newsCard = document.createElement('div');
                newsCard.className = 'news-item';
                newsCard.style.marginBottom = '1.5rem';
                newsCard.style.paddingBottom = '1.5rem';
                newsCard.style.borderBottom = '1px solid #e2e8f0';

                // Date formatting
                let dateStr = '';
                if (item.published) {
                    dateStr = new Date(item.published * 1000).toLocaleDateString();
                }

                newsCard.innerHTML = `
                    <h4 style="margin: 0 0 0.5rem 0;">
                        <a href="${item.link}" target="_blank" style="text-decoration: none; color: inherit; hover: color: var(--primary-color);">${item.title}</a>
                    </h4>
                    <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 0.8rem;">
                        ${item.publisher} • ${dateStr}
                    </div>
                `;

                // Add sentiment bars
                // item.sentiment is usually [{label, score}, ...] (not nested list)
                // because we processed it in main.py zip loop
                const bars = createSentimentBars(item.sentiment);
                newsCard.appendChild(bars);
                display.appendChild(newsCard);
            });

        } else {
            display.innerHTML = `<p style="color: red;">Error: ${json.detail || 'Unknown error'}</p>`;
        }
    } catch (error) {
        loader.classList.add('hidden');
        display.innerHTML = `<p style="color: red;">Network Error: ${error.message}</p>`;
    }
});
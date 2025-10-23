document.getElementById('predict-btn').onclick = async function() {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = "Загрузка...";
    try {
        const response = await fetch('http://127.0.0.1:8000/predict');
        if (!response.ok) throw new Error('API error: ' + response.status);
        const data = await response.json();
        resultDiv.innerHTML = "Количество сделок: " + data.predicted_deals;
    } catch (err) {
        resultDiv.innerHTML = "Ошибка: " + err.message;
    }
};

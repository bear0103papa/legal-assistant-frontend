// 取得 DOM 元素
const qaForm = document.getElementById('qa-form');
const questionTextarea = document.getElementById('question');
const submitButton = document.getElementById('submit-button');
const loadingIndicator = document.getElementById('loading-indicator');
const errorMessageDiv = document.getElementById('error-message');
const resultsDiv = document.getElementById('results');
const answerDiv = document.getElementById('answer');
const sourcesUl = document.getElementById('sources');

// --- 設定後端 API 位址 ---
// 開發時使用本地後端
const API_ENDPOINT = 'https://legal-assistant-api.onrender.com/api/ask';
// 部署後端到 Render 後，需要將此處改為 Render 提供的 URL
// const API_ENDPOINT = 'YOUR_RENDER_BACKEND_URL/api/ask';

// 監聽表單提交事件
qaForm.addEventListener('submit', async (event) => {
    event.preventDefault(); // 防止表單傳統提交

    const userQuestion = questionTextarea.value.trim();

    // 基本驗證
    if (!userQuestion) {
        displayError("請輸入您的問題。");
        return;
    }

    // UI 狀態更新：顯示載入中、清除舊結果、禁用按鈕
    loadingIndicator.classList.remove('hidden');
    resultsDiv.classList.add('hidden');
    errorMessageDiv.classList.add('hidden'); // 清除舊錯誤
    answerDiv.textContent = ''; // 清除舊答案
    sourcesUl.innerHTML = ''; // 清除舊來源
    submitButton.disabled = true;
    submitButton.textContent = '處理中...';

    try {
        // 發送請求到後端 API
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: userQuestion }),
        });

        // 檢查回應狀態
        if (!response.ok) {
            // 嘗試解析後端回傳的錯誤訊息
            let errorData;
            try {
                errorData = await response.json();
            } catch (parseError) {
                // 如果無法解析 JSON，顯示通用錯誤
                throw new Error(`伺服器錯誤，狀態碼：${response.status}`);
            }
            // 顯示後端提供的錯誤訊息或通用訊息
            throw new Error(errorData?.error || `請求失敗，狀態碼：${response.status}`);
        }

        // 解析成功的 JSON 回應
        const data = await response.json();

        // 顯示結果
        displayResults(data.answer, data.sources);

    } catch (error) {
        // 捕捉 fetch 錯誤或手動拋出的錯誤
        console.error("請求錯誤:", error);
        displayError(`發生錯誤：${error.message}`);
    } finally {
        // 無論成功或失敗，都要隱藏載入中並啟用按鈕
        loadingIndicator.classList.add('hidden');
        submitButton.disabled = false;
        submitButton.textContent = '提出問題';
    }
});

// 顯示結果的函數
function displayResults(answer, sources) {
    resultsDiv.classList.remove('hidden');
    answerDiv.textContent = answer || "AI 未提供回答內容。"; // 處理空回答

    sourcesUl.innerHTML = ''; // 清空舊來源
    if (sources && sources.length > 0) {
        sources.forEach(source => {
            const li = document.createElement('li');
            // 顯示主要的中繼資料
            li.innerHTML = `
                <strong>法規標題:</strong> ${escapeHtml(source.title || 'N/A')}<br>
                <strong>來源檔案:</strong> ${escapeHtml(source.source_file || 'N/A')}<br>
                <strong>原始行號:</strong> ${escapeHtml(source.source_row || 'N/A')}<br>
                <strong>文件日期:</strong> ${escapeHtml(source.date || 'N/A')}<br>
                <strong>文號:</strong> ${escapeHtml(source.number || 'N/A')}
            `;
            // 可以在這裡添加一個按鈕或連結，未來用於顯示完整的原始文本 (如果後端回傳的話)
            sourcesUl.appendChild(li);
        });
    } else {
        sourcesUl.innerHTML = '<li>未找到相關的法規來源。</li>';
    }
}

// 顯示錯誤訊息的函數
function displayError(message) {
    errorMessageDiv.textContent = message;
    errorMessageDiv.classList.remove('hidden');
    resultsDiv.classList.add('hidden'); // 隱藏結果區
}

// 簡易的 HTML 轉義函數，防止 XSS
function escapeHtml(unsafe) {
    if (typeof unsafe !== 'string') return unsafe; // 只處理字串
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}
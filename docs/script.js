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
const HEALTH_CHECK_ENDPOINT = 'https://legal-assistant-api.onrender.com/api/health'; // 健康檢查端點
const API_ENDPOINT = 'https://legal-assistant-api.onrender.com/api/ask'; // 主要 API 端點
// 部署後端到 Render 後，需要將此處改為 Render 提供的 URL
// const API_ENDPOINT = 'YOUR_RENDER_BACKEND_URL/api/ask';

// 監聽表單提交事件
qaForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const userQuestion = questionTextarea.value.trim();

    if (!userQuestion) {
        displayError("請輸入您的問題。");
        return;
    }

    // --- UI 更新：初始狀態 ---
    setLoadingState(true, "正在連接伺服器..."); // 初始訊息
    clearResultsAndErrors();

    let backendAwake = false;
    try {
        // --- 步驟 1: 嘗試呼叫健康檢查端點喚醒伺服器 ---
        console.log("正在執行健康檢查...");
        const healthResponse = await fetch(HEALTH_CHECK_ENDPOINT);

        if (!healthResponse.ok) {
            // 如果健康檢查失敗，可能是伺服器真的有問題
            throw new Error(`伺服器健康檢查失敗，狀態碼：${healthResponse.status}`);
        }

        const healthData = await healthResponse.json();
        if (healthData.status === 'ok') {
            backendAwake = true;
            console.log("健康檢查成功，後端已喚醒或正在運行。");
        } else {
             throw new Error("伺服器健康檢查回應異常。");
        }

        // --- 步驟 2: 如果健康檢查成功，才呼叫主要 API ---
        setLoadingState(true, "正在處理您的問題..."); // 更新訊息

        const apiResponse = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: userQuestion }),
        });

        if (!apiResponse.ok) {
            let errorData;
            try {
                errorData = await apiResponse.json();
            } catch (parseError) {
                throw new Error(`伺服器錯誤，狀態碼：${apiResponse.status}`);
            }
            throw new Error(errorData?.error || `請求失敗，狀態碼：${apiResponse.status}`);
        }

        const data = await apiResponse.json();
        displayResults(data.answer, data.sources);

    } catch (error) {
        console.error("請求錯誤:", error);
        // 如果是健康檢查就失敗了，給出不同的提示
        if (!backendAwake) {
             displayError(`無法連接到 AI 伺服器：${error.message}`);
        } else {
             displayError(`處理問題時發生錯誤：${error.message}`);
        }
    } finally {
        // 無論如何，結束載入狀態
        setLoadingState(false);
    }
});

// --- Helper 函數 ---

function setLoadingState(isLoading, message = "處理中...") {
    if (isLoading) {
        loadingIndicator.classList.remove('hidden');
        loadingIndicator.querySelector('p').textContent = message; // 更新載入訊息
        submitButton.disabled = true;
        submitButton.textContent = message.includes("處理中") ? "處理中..." : "請稍候..."; // 按鈕文字可以簡單些
    } else {
        loadingIndicator.classList.add('hidden');
        submitButton.disabled = false;
        submitButton.textContent = '提出問題';
    }
}

function clearResultsAndErrors() {
    resultsDiv.classList.add('hidden');
    errorMessageDiv.classList.add('hidden');
    answerDiv.textContent = '';
    sourcesUl.innerHTML = '';
}

// 顯示結果的函數
function displayResults(answer, sources) {
    resultsDiv.classList.remove('hidden');
    answerDiv.textContent = answer || "AI 未提供回答內容。"; // 處理空回答

    sourcesUl.innerHTML = ''; // 清空舊來源
    if (sources && sources.length > 0) {
        sources.forEach(source => { // 現在 source 包含 metadata 和 text
            const li = document.createElement('li');
            const metadata = source.metadata || {}; // 取出 metadata
            const text = source.text || "N/A"; // 取出 text

            // 顯示主要的中繼資料和文本
            li.innerHTML = `
                <strong>法規標題:</strong> ${escapeHtml(metadata.title || 'N/A')}<br>
                <strong>來源檔案:</strong> ${escapeHtml(metadata.source_file || 'N/A')}<br>
                <strong>原始行號:</strong> ${escapeHtml(metadata.source_row || 'N/A')}<br>
                <strong>文件日期:</strong> ${escapeHtml(metadata.date || 'N/A')}<br>
                <strong>文號:</strong> ${escapeHtml(metadata.number || 'N/A')}<br>
                <strong style="margin-top: 5px; display: block;">內容片段:</strong>
                <div class="source-text">${escapeHtml(text)}</div>
            `; // <--- 加入顯示 text 的部分
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
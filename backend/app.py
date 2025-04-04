import os
import json
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS # 處理跨來源請求
from sklearn.metrics.pairwise import cosine_similarity

# --- 配置與初始化 ---
load_dotenv() # 載入 .env 文件中的環境變數
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("請設定 GOOGLE_API_KEY 環境變數")

genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
# 允許 GitHub Pages 的基礎 URL 和包含子路徑的 URL
origins = [
    "https://bear0103papa.github.io", # <--- 添加這個基礎 URL
    "https://bear0103papa.github.io/legal-assistant-frontend/" # 保留原有的以防萬一
]
# 如果你有自訂網域，也加進去
# origins = ["https://your-username.github.io", "https://www.yourdomain.com"]

CORS(app, origins=origins) # 明確指定允許的來源

# --- 全域變數 (用於儲存資料與向量) ---
regulation_chunks = [] # 儲存處理後的法規片段 text
regulation_data = []   # 儲存完整的法規資料 (包含 metadata)
chunk_embeddings = None # 儲存法規片段的向量
# --- 新增：定義檔案路徑 ---
DATA_DIR = "data"
REGULATIONS_JSON_PATH = os.path.join(DATA_DIR, "regulations.json")
EMBEDDINGS_NPY_PATH = os.path.join(DATA_DIR, "embeddings.npy")

# --- 資料載入與向量化函數 (優化版) ---
def load_and_embed_data(json_path=REGULATIONS_JSON_PATH, embeddings_path=EMBEDDINGS_NPY_PATH, model_name="models/text-embedding-004"):
    """
    載入JSON資料。如果存在預計算的向量檔案則載入，否則計算向量並儲存。
    """
    global regulation_chunks, chunk_embeddings, regulation_data
    print(f"開始載入與處理資料: {json_path}")

    # 1. 載入主要的 JSON 資料
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"錯誤：找不到主要的資料檔案 {json_path}。無法繼續。")
        return
    except json.JSONDecodeError:
        print(f"錯誤：無法解析 JSON 檔案 {json_path}。無法繼續。")
        return

    # 過濾並儲存有效資料
    regulation_data = [item for item in data if item.get('text')]
    regulation_chunks = [item.get('text', '') for item in regulation_data]
    num_docs = len(regulation_chunks)
    print(f"從 JSON 載入 {num_docs} 個有效的法規片段")

    if num_docs == 0:
        print("錯誤：未載入任何有效的法規片段。")
        return

    # 2. 嘗試載入預計算的向量
    recalculate_embeddings = False
    if os.path.exists(embeddings_path):
        try:
            print(f"找到預計算的向量檔案: {embeddings_path}，嘗試載入...")
            loaded_embeddings = np.load(embeddings_path)
            print(f"成功載入向量，形狀: {loaded_embeddings.shape}")

            # **重要驗證**：檢查向量數量是否與文件數量匹配
            if loaded_embeddings.shape[0] == num_docs:
                chunk_embeddings = loaded_embeddings
                print("向量數量與文件數量匹配，使用已載入的向量。")
            else:
                print(f"警告：載入的向量數量 ({loaded_embeddings.shape[0]}) 與 JSON 文件數量 ({num_docs}) 不符。將重新計算向量。")
                recalculate_embeddings = True
        except Exception as e:
            print(f"載入向量檔案 {embeddings_path} 時發生錯誤: {e}。將重新計算向量。")
            recalculate_embeddings = True
    else:
        print(f"未找到預計算的向量檔案: {embeddings_path}。將計算新向量。")
        recalculate_embeddings = True

    # 3. 如果需要，計算並儲存向量
    if recalculate_embeddings:
        print(f"開始使用模型 {model_name} 生成向量...")
        try:
            # 注意：免費方案可能有請求頻率限制，大量文本可能需要分批處理
            # 考慮加入分批處理邏輯 (如果 chunks 數量非常大)
            batch_size = 100 # Google API 建議批次大小不超過 100
            all_embeddings = []
            for i in range(0, num_docs, batch_size):
                 batch_chunks = regulation_chunks[i:min(i + batch_size, num_docs)]
                 print(f"  處理批次 {i // batch_size + 1} / { (num_docs + batch_size - 1) // batch_size } (大小: {len(batch_chunks)})")
                 result = genai.embed_content(
                     model=model_name,
                     content=batch_chunks,
                     task_type="RETRIEVAL_DOCUMENT"
                 )
                 all_embeddings.extend(result['embedding'])

            chunk_embeddings = np.array(all_embeddings)
            print(f"向量生成完畢，形狀: {chunk_embeddings.shape}")

            # 儲存新計算的向量到 .npy 檔案
            try:
                 # 確保 data 目錄存在
                 os.makedirs(DATA_DIR, exist_ok=True)
                 np.save(embeddings_path, chunk_embeddings)
                 print(f"已將新計算的向量儲存到: {embeddings_path}")
            except Exception as e:
                 print(f"儲存向量檔案 {embeddings_path} 時發生錯誤: {e}")

        except Exception as e:
            print(f"生成向量時發生錯誤: {e}")
            # 如果計算失敗，確保 embedding 變數是 None
            chunk_embeddings = None
            regulation_chunks = []
            regulation_data = []


# --- 在應用程式啟動時載入資料 ---
load_and_embed_data() # 現在這個函數會處理載入或計算

# --- 向量搜尋函數 ---
def find_top_n_similar(query_embedding, doc_embeddings, n=5):
    """計算查詢向量與所有文件向量的相似度，返回最相似的 n 個索引"""
    if doc_embeddings is None or query_embedding is None:
        return []
    # 計算餘弦相似度
    # query_embedding 是 (1, dim), doc_embeddings 是 (N, dim)
    # cosine_similarity 需要 2D 陣列輸入
    similarities = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings)[0]
    # 獲取分數最高的 n 個索引 (使用 argsort)
    # [::-1] 是為了反轉排序，得到降序索引
    top_n_indices = np.argsort(similarities)[::-1][:n]
    return top_n_indices

# --- API 端點 ---
@app.route('/api/ask', methods=['POST'])
def ask_question():
    """接收問題，執行 RAG 流程，返回答案"""
    global regulation_chunks, chunk_embeddings, regulation_data

    # 1. 檢查資料是否已載入
    if chunk_embeddings is None or not regulation_chunks:
        return jsonify({"error": "後端資料尚未準備完成或載入失敗，請查看後端日誌。"}), 503 # Service Unavailable

    # 2. 獲取請求中的問題
    data = request.get_json()
    if not data or 'question' not in data or not data['question'].strip():
        return jsonify({"error": "請求中未包含 'question' 或問題為空。"}), 400 # Bad Request

    user_question = data['question'].strip()
    print(f"收到問題: {user_question}")

    try:
        # 3. 為問題生成向量
        print("正在為問題生成向量...")
        question_embedding_result = genai.embed_content(
            model="models/text-embedding-004",
            content=user_question,
            task_type="RETRIEVAL_QUERY" # 指明這是用於檢索的查詢向量
        )
        question_embedding = np.array(question_embedding_result['embedding'])
        print("問題向量生成完畢。")

        # 4. 執行向量搜尋，找到最相關的 chunk 索引
        print("正在搜尋相關法規片段...")
        top_indices = find_top_n_similar(question_embedding, chunk_embeddings, n=20) # <--- 調整 n 值
        print(f"找到最相關的索引: {top_indices}")

        # 5. 組合上下文 (Context)
        context = ""
        if len(top_indices) > 0:
            context += "根據以下法規內容：\n\n"
            for index in top_indices:
                 # 從 regulation_data 中獲取完整資訊，而不僅是 text
                 item = regulation_data[index]
                 metadata = item.get('metadata', {})
                 context += f"--- 法規來源: {metadata.get('title','未知標題')} (檔案: {metadata.get('source_file','未知檔案')}, 行: {metadata.get('source_row','未知')}) ---\n"
                 context += f"{item['text']}\n\n" # item['text'] 就是 regulation_chunks[index]
        else:
            context = "沒有找到直接相關的法規片段。"
            print("警告：未找到相關的法規片段。")

        # 6. 建構 Prompt
        prompt = f"""
        你是一位專精於台灣法律的 AI 助理。請根據以下提供的「相關法規內容」來回答「使用者的問題」。

        你的回答應該：
        - 嚴格基於提供的法規內容。
        - 如果提供的內容不足以回答問題，請明確說明無法根據現有資訊回答。
        - 保持客觀、中立和專業。
        - 使用繁體中文回答。

        --- 相關法規內容 ---
        {context}
        --- 使用者的問題 ---
        {user_question}

        --- 你的回答 ---
        """

        # 7. 呼叫 Gemini 模型生成答案
        print("正在呼叫 Gemini 模型生成答案...")
        # 選擇一個生成模型，例如 gemini-2.5-pro-exp-03-25
        model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25') # <--- 檢查模型名稱
        response = model.generate_content(prompt)
        print("Gemini 模型回應完成。")

        # 8. 提取並返回答案
        answer = response.text

        # 同時返回找到的來源資訊 (可選，供前端顯示)
        sources = [regulation_data[i]['metadata'] for i in top_indices]

        return jsonify({
            "answer": answer,
            "sources": sources # 返回相關來源的 metadata
            })

    except Exception as e:
        print(f"處理請求時發生錯誤: {e}")
        # 在生產環境中，這裡應該記錄更詳細的錯誤資訊
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"處理請求時發生內部錯誤: {e}"}), 500 # Internal Server Error


if __name__ == '__main__':
    # 應用程式啟動時即載入資料並生成向量
    # load_and_embed_data() # 這行已經在全域範圍被調用
    app.run(debug=True, port=5001)

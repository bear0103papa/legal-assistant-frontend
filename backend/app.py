import os
import json
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS # 處理跨來源請求
from sklearn.metrics.pairwise import cosine_similarity
import traceback # 保持 import

# --- 新增 Import (根據 google-generativeai SDK 的結構) ---
# 通常 Tool 和 GoogleSearchRetrieval 會在 types 或 protos 子模組中
# 嘗試從 google.ai.generativelanguage 導入 (根據 SDK 版本可能不同)
try:
    from google.ai.generativelanguage import Tool, GoogleSearchRetrieval
except ImportError:
    # 如果上述導入失敗，嘗試從 types 或 protos (較舊版本可能方式)
    try:
        # from google.generativeai.types import Tool, GoogleSearchRetrieval # 較新 SDK 可能的方式
        # 或者直接使用 protos (如果需要)
        from google.protobuf.struct_pb2 import Struct
        # 創建一個 GoogleSearchRetrieval 實例可能不需要直接導入，看 API 設計
        print("注意：無法直接導入 Tool/GoogleSearchRetrieval，將嘗試使用簡易配置。")
        # 在下面的 API 調用中，我們將使用更簡單的方式啟用 grounding
        Tool = None # 設為 None 以觸發後面的簡易配置
        GoogleSearchRetrieval = None
    except ImportError:
         print("警告：無法找到 Tool 或 GoogleSearchRetrieval 的導入方式，網路搜尋功能可能無法啟用。請檢查 google-generativeai SDK 版本和文檔。")
         Tool = None
         GoogleSearchRetrieval = None

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
def find_top_n_similar(query_embedding, doc_embeddings, n=3000): # <-- 將 n 調回一個較小的值，例如 5 或 10
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
        top_indices = find_top_n_similar(question_embedding, chunk_embeddings, n=3000) # 使用較小的 n
        print(f"找到最相關的索引: {top_indices}")

        # 5. 組合上下文 (Context)
        context = ""
        if len(top_indices) > 0:
            context += "--- 以下是從我們的法規資料庫中找到的相關內容 ---\n\n" # <-- 標示來源
            for index in top_indices:
                 # 從 regulation_data 中獲取完整資訊，而不僅是 text
                 item = regulation_data[index]
                 metadata = item.get('metadata', {})
                 context += f"來源: {metadata.get('title','未知標題')} (檔案: {metadata.get('source_file','未知檔案')}, 行: {metadata.get('source_row','未知')})\n"
                 context += f"內容: {item['text']}\n\n" # item['text'] 就是 regulation_chunks[index]
        else:
            context = "我們的法規資料庫中沒有找到直接相關的片段。"
            print("警告：未找到相關的法規片段。")

        # 6. 建構 Prompt
        prompt = f"""
        你是一位專精於台灣法律的 AI 助理。請結合以下提供的「法規資料庫內容」和必要的「網路搜尋結果」來回答「使用者的問題」。

        你的回答應該：
        1.  **優先** 參考「法規資料庫內容」。如果這部分內容足以回答，請主要依據它。
        2.  **僅在必要時** 使用「網路搜尋結果」來補充法規資料庫中沒有的資訊、最新的發展或進行事實核查。
        3.  **非常重要：** 在你的回答中，必須 **明確區分** 資訊來源。使用如「根據我們資料庫中的《XX法》...」、「根據網路搜尋的最新資訊...」等字句來標示。
        4.  如果兩個來源的資訊有衝突，請指出衝突點。
        5.  如果法規資料庫和網路搜尋都無法提供答案，請明確說明。
        6.  保持客觀、中立和專業。
        7.  使用繁體中文回答。

        --- 法規資料庫內容 ---
        {context}
        --- 使用者的問題 ---
        {user_question}

        --- 你的回答 (請務必區分資訊來源) ---
        """

        # 7. 呼叫 Gemini 模型生成答案
        print("正在呼叫 Gemini 模型生成答案 (已啟用網路搜尋)...")
        # 選擇一個生成模型，例如 gemini-2.5-pro-exp-03-25
        model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25') # <--- 檢查模型名稱

        # --- *** 修改 API 呼叫以啟用 Google 搜尋 *** ---
        tools_config = None
        if Tool and GoogleSearchRetrieval:
             # 使用導入的類 (較佳方式)
             gs_retrieval = GoogleSearchRetrieval() # 可以添加參數，如 disable_attribution=False
             tools_config = [Tool(google_search_retrieval=gs_retrieval)]
             print("使用 Tool/GoogleSearchRetrieval 配置網路搜尋。")
        else:
             # 簡易配置方式 (如果導入失敗，嘗試此方式，可能隨 SDK 更新)
             # 創建一個簡單的 proto message 或 dict 來啟用
             # 注意：這種方式的具體結構可能依賴 SDK 版本
             try:
                  # 嘗試創建一個表示啟用搜尋的 proto message
                  # 這部分語法可能需要根據 google.protobuf 和 SDK 的實際情況調整
                  search_tool_struct = Struct()
                  # search_tool_struct.fields['google_search_retrieval'].CopyFrom(Struct()) # 空結構體表示啟用，無特定參數
                  # 假設工具列表可以直接接受這種結構 (或簡化為字典)
                  # tools_config = [{'google_search_retrieval': {}}] # 另一種可能的簡化配置
                  # *** 最可能有效的簡易配置是直接使用 google.ai.generativelanguage 的 protos ***
                  from google.ai.generativelanguage import Tool as ProtoTool, GoogleSearchRetrieval as ProtoGoogleSearchRetrieval
                  tools_config = [ProtoTool(google_search_retrieval=ProtoGoogleSearchRetrieval())]
                  print("使用 google.ai.generativelanguage.Tool 配置網路搜尋。")
             except Exception as import_err:
                  print(f"創建簡易網路搜尋配置時出錯: {import_err}。將不啟用網路搜尋。")
                  tools_config = None


        # 根據 tools_config 是否成功創建來決定是否傳遞 tools 參數
        if tools_config:
             response = model.generate_content(prompt, tools=tools_config)
        else:
             response = model.generate_content(prompt) # 不啟用搜尋

        print("Gemini 模型回應完成。")

        # 8. 提取並返回答案
        answer = response.text

        # 修改 sources 的產生方式
        sources = []
        for index in top_indices:
            item = regulation_data[index]
            source_info = {
                "metadata": item.get('metadata', {}),
                "text": item.get('text', '') # <-- 加入 text 欄位
            }
            sources.append(source_info)

        return jsonify({
            "answer": answer,
            "sources": sources # 現在 sources 包含 text 了
            })

    except Exception as e:
        print(f"處理請求時發生錯誤: {e}")
        # 在生產環境中，這裡應該記錄更詳細的錯誤資訊
        traceback.print_exc()
        return jsonify({"error": f"處理請求時發生內部錯誤: {e}"}), 500 # Internal Server Error


if __name__ == '__main__':
    # 應用程式啟動時即載入資料並生成向量
    # load_and_embed_data() # 這行已經在全域範圍被調用
    print("警告：正在使用 Flask 開發伺服器運行。部署時應使用 Gunicorn。")
    # 確保 debug=False，避免在生產中暴露敏感信息
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)), debug=False)

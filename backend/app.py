import os
import json
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
import traceback

# --- *** 嘗試再次導入 Protobuf Message *** ---
try:
    # 這是啟用 grounding 最可能正確的 proto message
    from google.ai.generativelanguage import Tool as ProtoTool
    from google.ai.generativelanguage import GoogleSearchRetrieval as ProtoGoogleSearchRetrieval
    print("成功導入 google.ai.generativelanguage.Tool 和 GoogleSearchRetrieval protos.")
    PROTO_AVAILABLE = True
except ImportError:
    print("警告：無法導入 google.ai.generativelanguage protos。網路搜尋功能可能無法正確啟用。")
    PROTO_AVAILABLE = False
    ProtoTool = None # 設為 None 以便後續檢查
    ProtoGoogleSearchRetrieval = None


# --- 配置與初始化 (保持不變) ---
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("請設定 GOOGLE_API_KEY 環境變數")
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
origins = [
    "https://bear0103papa.github.io",
    "https://bear0103papa.github.io/legal-assistant-frontend/"
]
CORS(app, origins=origins)

# --- 全域變數與資料載入 (保持不變) ---
regulation_chunks = []
regulation_data = []
chunk_embeddings = None
DATA_DIR = "data"
REGULATIONS_JSON_PATH = os.path.join(DATA_DIR, "regulations.json")
EMBEDDINGS_NPY_PATH = os.path.join(DATA_DIR, "embeddings.npy")

def load_and_embed_data(json_path=REGULATIONS_JSON_PATH, embeddings_path=EMBEDDINGS_NPY_PATH, model_name="models/text-embedding-004"):
    # ... (函數內容保持不變) ...
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

load_and_embed_data()

# --- 向量搜尋函數 (保持不變, 確保 n 較小) ---
def find_top_n_similar(query_embedding, doc_embeddings, n=100):
    """計算查詢向量與所有文件向量的相似度，返回最相似的 n 個索引"""
    if doc_embeddings is None or query_embedding is None:
        return []
    similarities = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings)[0]
    top_n_indices = np.argsort(similarities)[::-1][:n]
    return top_n_indices

# --- API 端點 ---
@app.route('/api/ask', methods=['POST'])
def ask_question():
    global regulation_chunks, chunk_embeddings, regulation_data

    if chunk_embeddings is None or not regulation_chunks:
        return jsonify({"error": "後端資料尚未準備完成或載入失敗，請查看後端日誌。"}), 503

    data = request.get_json()
    if not data or 'question' not in data or not data['question'].strip():
        return jsonify({"error": "請求中未包含 'question' 或問題為空。"}), 400

    user_question = data['question'].strip()
    print(f"收到問題: {user_question}")

    try:
        print("正在為問題生成向量...")
        question_embedding_result = genai.embed_content(
            model="models/text-embedding-004",
            content=user_question,
            task_type="RETRIEVAL_QUERY"
        )
        question_embedding = np.array(question_embedding_result['embedding'])
        print("問題向量生成完畢。")

        print("正在搜尋相關法規片段...")
        top_indices = find_top_n_similar(question_embedding, chunk_embeddings, n=100) # 使用較小的 n
        print(f"找到最相關的索引 (n={len(top_indices)}): {top_indices}")

        context = ""
        if len(top_indices) > 0:
            context += "--- 以下是從我們的法規資料庫中找到的相關內容 ---\n\n"
            for index in top_indices:
                 item = regulation_data[index]
                 metadata = item.get('metadata', {})
                 context += f"來源: {metadata.get('title','未知標題')} (檔案: {metadata.get('source_file','未知檔案')}, 行: {metadata.get('source_row','未知')})\n"
                 context += f"內容: {item['text']}\n\n"
        else:
            context = "我們的法規資料庫中沒有找到直接相關的片段。"
            print("警告：未找到相關的法規片段。")

        # --- Prompt 保持之前的修改，強調區分來源 ---
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

        print("正在呼叫 Gemini 模型生成答案 (嘗試啟用網路搜尋)...")
        model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')

        # --- *** 嘗試使用 Protobuf Message 配置 Google 搜尋 *** ---
        tools_config = None # 先初始化為 None
        if PROTO_AVAILABLE:
            try:
                # 創建 GoogleSearchRetrieval proto message 實例 (空實例表示預設行為)
                google_search_retrieval_proto = ProtoGoogleSearchRetrieval()
                # 將其包裹在 Tool proto message 中
                tool_proto = ProtoTool(google_search_retrieval=google_search_retrieval_proto)
                # 放入列表中
                tools_config = [tool_proto]
                print("使用 Protobuf Message (google.ai.generativelanguage.Tool) 配置網路搜尋。")
            except Exception as proto_err:
                print(f"使用 Protobuf Message 配置網路搜尋時出錯: {proto_err}。將不啟用網路搜尋。")
                traceback.print_exc()
                tools_config = None # 配置失敗，確保設回 None
        else:
             print("Protobuf Message 無法導入，不啟用網路搜尋。")

        # --- 呼叫 API ---
        try:
            if tools_config:
                # 如果成功配置了 proto tool，則使用它
                response = model.generate_content(prompt, tools=tools_config)
            else:
                # 否則，不帶 tools 參數呼叫
                print("未成功配置網路搜尋工具，將不啟用網路搜尋功能進行 API 呼叫。")
                response = model.generate_content(prompt)

            print("Gemini 模型回應完成。")

        except Exception as api_call_err:
             # 捕獲 API 調用本身可能因為 tools 配置或其他原因產生的錯誤
             print(f"呼叫 generate_content 時發生錯誤 (可能與 tools 有關): {api_call_err}")
             traceback.print_exc()
             # 如果帶 tools 的調用失敗，嘗試不帶 tools 再調用一次
             print("嘗試不使用網路搜尋重新呼叫 API...")
             response = model.generate_content(prompt)
             print("不使用網路搜尋的 API 呼叫完成。")


        answer = response.text

        # 處理 sources 的回傳 (保持不變)
        sources = []
        for index in top_indices:
            item = regulation_data[index]
            source_info = {
                "metadata": item.get('metadata', {}),
                "text": item.get('text', '')
            }
            sources.append(source_info)

        return jsonify({
            "answer": answer,
            "sources": sources
            })

    except Exception as e:
        print(f"處理請求時發生嚴重錯誤: {e}")
        traceback.print_exc()
        return jsonify({"error": f"處理請求時發生內部錯誤: {e}"}), 500

if __name__ == '__main__':
    # load_and_embed_data() # 已在全域調用
    print("警告：正在使用 Flask 開發伺服器運行。部署時應使用 Gunicorn。")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)), debug=False)

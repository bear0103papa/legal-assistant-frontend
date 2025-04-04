import os
import json
import numpy as np
import google.generativeai as genai
from google.generativeai.types import Tool, GenerateContentConfig, GoogleSearch, safety_types
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
import traceback
from google.api_core import exceptions as api_exceptions

# --- *** 修改路徑定義以使其相對於 app.py *** ---
# 獲取 app.py 所在的目錄的絕對路徑
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR_NAME 仍然是子目錄名稱
DATA_DIR_NAME = "data"
# 構建 data 目錄的絕對路徑
DATA_DIR_PATH = os.path.join(APP_DIR, DATA_DIR_NAME)
# 構建 regulations.json 和 embeddings.npy 的絕對路徑
REGULATIONS_JSON_PATH = os.path.join(DATA_DIR_PATH, "regulations.json")
EMBEDDINGS_NPY_PATH = os.path.join(DATA_DIR_PATH, "embeddings.npy")

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
                 os.makedirs(DATA_DIR_PATH, exist_ok=True)
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

# --- 向量搜尋函數 (確保 n 較小) ---
def find_top_n_similar(query_embedding, doc_embeddings, n=10): # <-- *** 保持 n 值很小 ***
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
        top_indices = find_top_n_similar(question_embedding, chunk_embeddings, n=10) # <-- *** 確保 n 值很小 ***
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

        # --- *** Prompt 保持要求區分來源的版本 *** ---
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

        # --- *** 嘗試使用官方範例的方式配置 Google 搜尋 *** ---
        generation_config_with_tool = None # 先初始化
        try:
            # 1. 創建搜尋工具 (直接使用導入的 GoogleSearch)
            google_search_tool = Tool(google_search=GoogleSearch())
            print("嘗試使用 Tool(google_search=GoogleSearch()) 配置工具。")

            # 2. 創建 GenerationConfig (直接使用導入的 GenerateContentConfig)
            generation_config_with_tool = GenerateContentConfig(
                tools=[google_search_tool]
                # 可以加入其他 config，例如 temperature, max_output_tokens 等
                # temperature=0.4,
                # max_output_tokens=8000
            )
            print("成功創建包含搜尋工具的 GenerationConfig。")

        except NameError: # 如果 GoogleSearch 還是找不到，會是 NameError
            print("警告：當前 SDK 版本似乎不支援 GoogleSearch 類別。將不啟用網路搜尋。")
            traceback.print_exc()
            generation_config_with_tool = None
        except AttributeError: # 以防萬一還是 AttributeError
            print("警告：配置工具時發生 AttributeError。將不啟用網路搜尋。")
            traceback.print_exc()
            generation_config_with_tool = None
        except Exception as config_err:
            print(f"創建 GenerationConfig 時發生其他錯誤: {config_err}。將不啟用網路搜尋。")
            traceback.print_exc()
            generation_config_with_tool = None

        # --- 呼叫 API ---
        try:
            if generation_config_with_tool:
                # 如果成功配置了工具，則傳遞 generation_config
                response = model.generate_content(prompt, generation_config=generation_config_with_tool)
                print("使用 GenerationConfig (含搜尋工具) 呼叫 API。")
            else:
                # 否則，不帶 config 參數呼叫
                print("未成功配置網路搜尋工具，將不啟用網路搜尋功能進行 API 呼叫。")
                response = model.generate_content(prompt)

            print("Gemini 模型回應完成。")

        except api_exceptions.InvalidArgument as api_err:
             # 捕獲特定的 API 400 錯誤
             if "Search Grounding is not supported" in str(api_err):
                 print(f"API 錯誤：模型或配置不支援網路搜尋。錯誤訊息: {api_err}")
                 print("嘗試不使用網路搜尋重新呼叫 API...")
                 response = model.generate_content(prompt)
                 print("不使用網路搜尋的 API 呼叫完成。")
             else:
                 # 如果是其他 400 錯誤，則向上拋出
                 print(f"API 呼叫時發生未預期的 InvalidArgument 錯誤: {api_err}")
                 traceback.print_exc()
                 return jsonify({"error": f"呼叫語言模型時發生參數錯誤: {api_err}"}), 400
        except Exception as api_call_err:
             # 捕獲其他 API 調用錯誤
             print(f"呼叫 generate_content 時發生錯誤: {api_call_err}")
             traceback.print_exc()
             # 也可以考慮在這裡回退
             # print("嘗試不使用網路搜尋重新呼叫 API...")
             # response = model.generate_content(prompt)
             # print("不使用網路搜尋的 API 呼叫完成。")
             return jsonify({"error": f"呼叫語言模型時發生錯誤: {api_call_err}"}), 500


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

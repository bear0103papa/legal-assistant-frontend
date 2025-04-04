# backend/process_txt_laws.py
import os
import json
import re

# --- 設定 ---
BASE_DATA_DIR = "/Users/rogerchen/Documents/17. 法規AI/legal-qa-system/legal_qa_app/backend/data"
TXT_FILES_TO_PROCESS = [
    "各類所得扣繳率標準.txt",
    "所得基本稅額條例.txt",
    "所得稅法.txt",
    "稅捐稽徵法.txt",
    "稅捐稽徵法施行細則.txt",
    "營利事業所得稅查核準則.txt",
    "營業稅法.txt"
]
OUTPUT_JSON_PATH = os.path.join(BASE_DATA_DIR, "regulations.json") # 直接合併到目標檔案

# --- 函數：處理單個 TXT 檔案 ---
def process_law_txt(txt_filepath, law_title):
    """讀取 TXT，按條文切分，返回 chunks 列表"""
    chunks = []
    current_article_number = None
    current_chunk_text = ""
    start_line_number = 1 # 記錄原始行號 (可選)
    current_chunk_start_line = 1

    print(f"開始處理檔案: {txt_filepath} (法規: {law_title})")

    try:
        with open(txt_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line_content = line.strip()
            if not line_content: # 跳過空行
                 continue

            # 使用正則表達式尋找條文標題 (例如 "第 1 條", "第 3-1 條")
            # 考慮到可能有全形空格或半形空格
            match = re.match(r'^\s*第\s*([\w-]+)\s*條.*$', line_content) # \w 包含數字和字母, - 處理 "3-1"

            if match:
                # 如果找到新的條文，保存上一個 chunk (如果存在)
                if current_article_number is not None and current_chunk_text:
                    chunk_id = f"{law_title}_article_{current_article_number}"
                    chunks.append({
                        "id": chunk_id,
                        "text": current_chunk_text.strip(),
                        "metadata": {
                            "title": law_title,
                            "article": str(current_article_number), # 確保是字串
                            "source_file": os.path.basename(txt_filepath),
                            "source_row": current_chunk_start_line # 記錄該條起始行
                        }
                    })
                    # print(f"  - 新增 chunk: {chunk_id}")

                # 開始新的 chunk
                current_article_number = match.group(1) # 提取條號 (例如 "1", "3-1")
                current_chunk_text = line_content + "\n" # 包含條文標題行
                current_chunk_start_line = i + 1
            elif current_article_number is not None:
                # 如果已經在某個條文內，將內容添加到當前 chunk
                current_chunk_text += line_content + "\n"
            else:
                 # 在第一個條文標題出現前的內容，可以選擇忽略或作為開頭的 chunk
                 # 這裡暫時忽略開頭的非條文內容，可根據需要修改
                 # print(f"  - 忽略開頭行: {line_content}")
                 pass

        # 保存最後一個讀到的 chunk
        if current_article_number is not None and current_chunk_text:
            chunk_id = f"{law_title}_article_{current_article_number}"
            chunks.append({
                "id": chunk_id,
                "text": current_chunk_text.strip(),
                "metadata": {
                    "title": law_title,
                    "article": str(current_article_number),
                    "source_file": os.path.basename(txt_filepath),
                    "source_row": current_chunk_start_line
                }
            })
            # print(f"  - 新增 chunk: {chunk_id}")

        print(f"完成處理檔案: {txt_filepath}, 新增 {len(chunks)} 個條文 chunks。")
        return chunks

    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {txt_filepath}")
        return []
    except Exception as e:
        print(f"處理檔案 {txt_filepath} 時發生未預期錯誤：{e}")
        return []

# --- 主執行區塊 ---
if __name__ == "__main__":
    all_regulation_data = []

    # 1. (重要) 嘗試讀取現有的 regulations.json (來自 CSV 的資料)
    existing_data = []
    if os.path.exists(OUTPUT_JSON_PATH):
        try:
            with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"已成功載入現有的 {len(existing_data)} 筆資料從 {OUTPUT_JSON_PATH}")
        except json.JSONDecodeError:
            print(f"警告：現有的 {OUTPUT_JSON_PATH} 格式錯誤，將被覆蓋。")
        except Exception as e:
            print(f"讀取現有的 {OUTPUT_JSON_PATH} 時出錯: {e}。將被覆蓋。")

    # 將現有資料加入到最終列表
    all_regulation_data.extend(existing_data)
    print(f"目前資料總數 (處理 TXT 前): {len(all_regulation_data)}")

    print("\n開始處理 TXT 法規檔案...")

    # 2. 處理所有指定的 TXT 檔案
    for txt_filename in TXT_FILES_TO_PROCESS:
        txt_filepath = os.path.join(BASE_DATA_DIR, txt_filename)
        # 從檔名推斷法規標題 (去除 .txt)
        law_title = os.path.splitext(txt_filename)[0]
        # 處理單個 TXT 檔案並獲取 chunks
        new_chunks = process_law_txt(txt_filepath, law_title)
        # 將新處理的 chunks 添加到總列表
        all_regulation_data.extend(new_chunks)

    # 3. 寫入合併後的資料到 JSON 檔案
    if all_regulation_data:
        print(f"\n處理完成，最終資料總數: {len(all_regulation_data)}。")
        try:
            output_dir = os.path.dirname(OUTPUT_JSON_PATH)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"已建立輸出目錄: {output_dir}")

            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as jsonfile:
                json.dump(all_regulation_data, jsonfile, ensure_ascii=False, indent=2)
            print(f"成功將合併後的資料寫入到: {OUTPUT_JSON_PATH}")
        except IOError as e:
            print(f"寫入 JSON 檔案 {OUTPUT_JSON_PATH} 時發生 IO 錯誤：{e}")
        except Exception as e:
            print(f"寫入 JSON 檔案 {OUTPUT_JSON_PATH} 時發生未預期錯誤：{e}")
    else:
        print("\n未處理任何有效資料。請檢查 TXT 檔案內容和路徑。")
        
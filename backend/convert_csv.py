# convert_csv.py
import csv
import json
import os

def convert_csv_to_json(csv_filepath, output_list, file_prefix):
    """
    讀取指定的 CSV 檔案，將其內容轉換為結構化的字典列表，
    並添加到 output_list 中。

    Args:
        csv_filepath (str): CSV 檔案的路徑。
        output_list (list): 用於儲存所有處理後資料的列表。
        file_prefix (str): 用於生成唯一 ID 的檔案前綴。
    """
    initial_count = len(output_list) # 記錄處理前列表長度
    try:
        with open(csv_filepath, 'r', encoding='utf-8') as csvfile:
            # 使用 DictReader 將每一行讀取為字典
            reader = csv.DictReader(csvfile)

            # 檢查必要的欄位是否存在
            if not reader.fieldnames or not all(col in reader.fieldnames for col in ['title', 'content']):
                 print(f"警告：CSV 檔案 {csv_filepath} 缺少標頭或必要的 'title'/'content' 欄位，將跳過此檔案。")
                 return

            print(f"開始處理檔案: {csv_filepath}")
            processed_rows = 0
            for i, row in enumerate(reader):
                # 獲取內容，去除前後空白
                content = row.get('content', '').strip()
                title = row.get('title', '').strip()
                number = row.get('number', '').strip() # 處理可能不存在的欄位
                date = row.get('date', '').strip()     # 處理可能不存在的欄位

                # 確保 content 不為空才加入
                if content:
                    chunk = {
                        # 生成唯一 ID (檔名前綴 + 行號)
                        "id": f"{file_prefix}_row_{i+1}",
                        # 主要的文本內容，用於向量化
                        "text": content,
                        # 保留原始的相關資訊作為 metadata
                        "metadata": {
                            "title": title,
                            "number": number,
                            "date": date,
                            "source_file": os.path.basename(csv_filepath),
                            "source_row": i + 1 # 記錄原始行號 (1-based)
                        }
                    }
                    output_list.append(chunk)
                    processed_rows += 1

            print(f"成功處理檔案: {csv_filepath}，新增 {processed_rows} 筆有效資料。")

    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {csv_filepath}")
    except Exception as e:
        print(f"處理檔案 {csv_filepath} 時發生未預期錯誤：{e}")

# --- 主執行區塊 ---
if __name__ == "__main__":
    # 定義你的 CSV 檔案路徑
    csv_files_to_process = [
        '/Users/rogerchen/Documents/17. 法規AI/legal-qa-system/legal_qa_app/backend/data/processed_laws.csv',
        '/Users/rogerchen/Documents/17. 法規AI/legal-qa-system/legal_qa_app/backend/data/processed_company_laws.csv'
    ]

    # 定義輸出的 JSON 檔案路徑 (放在 backend/data/ 下)
    output_json_path = '/Users/rogerchen/Documents/17. 法規AI/legal-qa-system/legal_qa_app/backend/data/regulations.json'

    # 用於儲存所有處理結果的列表
    all_regulation_data = []

    print("開始轉換 CSV 到 JSON...")

    # 依序處理每個 CSV 檔案
    # 使用不同的前綴確保 ID 唯一性
    convert_csv_to_json(csv_files_to_process[0], all_regulation_data, "laws")
    convert_csv_to_json(csv_files_to_process[1], all_regulation_data, "company_laws")

    # 檢查是否有成功處理的資料
    if all_regulation_data:
        print(f"\n共處理完成 {len(all_regulation_data)} 筆資料。")
        # 將合併後的資料寫入 JSON 檔案
        try:
            # 確保輸出目錄存在
            output_dir = os.path.dirname(output_json_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"已建立輸出目錄: {output_dir}")

            with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
                # ensure_ascii=False 確保中文正確顯示，indent=2 使 JSON 格式化易讀
                json.dump(all_regulation_data, jsonfile, ensure_ascii=False, indent=2)
            print(f"成功將資料合併並寫入到: {output_json_path}")
        except IOError as e:
            print(f"寫入 JSON 檔案 {output_json_path} 時發生 IO 錯誤：{e}")
        except Exception as e:
            print(f"寫入 JSON 檔案 {output_json_path} 時發生未預期錯誤：{e}")
    else:
        print("\n未處理任何有效資料，沒有生成 JSON 檔案。請檢查 CSV 檔案內容和路徑。")

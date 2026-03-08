import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

def main():
    # ---------------------------------------------------------
    # 1. 設定模型路徑
    # 如果你有下載微調模型，請改成資料夾路徑，例如: "C:/Users/bhbor/Downloads/finetuned_senBERT_train_v2"
    # 如果沒有，就保留 'all-MiniLM-L6-v2' (會自動下載基礎模型)
    model_name_or_path = r"C:\Users\bhbor\OneDrive - UW\咪咪\LLM\Mini Project\P2\finetuned_senBERT_v3_kaggle"
    # ---------------------------------------------------------

    print(f"Loading model: {model_name_or_path} ...")
    try:
        model = SentenceTransformer(model_name_or_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("請確認你的模型路徑是否正確，或是網路是否通暢。")
        return

    # 2. 讀取資料
    print("Reading CSV files...")
    try:
        df_test_queries = pd.read_csv('test_query.csv')
        df_test_docs = pd.read_csv('test_documents.csv')
        df_sample = pd.read_csv('sample_submission.csv')
    except FileNotFoundError as e:
        print(f"Error: 找不到檔案 - {e}")
        print("請確認 test_query.csv, test_documents.csv, sample_submission.csv 都在同一個資料夾下。")
        return

    print(f"Loaded {len(df_test_queries)} queries and {len(df_test_docs)} documents.")

    # 3. 準備資料 (合併標題與內文)
    # fillna('') 防止有空值報錯
    doc_texts = (df_test_docs['Title'].fillna('') + " " + df_test_docs['Text'].fillna('')).tolist()
    doc_ids = df_test_docs['Doc_ID'].tolist()
    
    query_texts = df_test_queries['Query'].tolist()
    query_ids = df_test_queries['Query'].tolist() # 使用 Query 文字當 ID (配合 sample_submission)

    # 4. 編碼 (Encoding)
    print("Encoding documents... (這可能需要一點時間)")
    doc_embeddings = model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True)
    
    print("Encoding queries...")
    query_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=True)

    # 5. 計算相似度與排名
    print("Ranking...")
    # util.cos_sim 會計算所有 query 對所有 doc 的相似度矩陣
    cosine_scores = util.cos_sim(query_embeddings, doc_embeddings)

    output_rows = []
    
    # 遍歷每個 Query
    for i in range(len(query_texts)):
        qid = query_ids[i]
        scores = cosine_scores[i]
        
        # 取得分數最高的 Top 10 索引
        # torch.topk 回傳 (values, indices)
        top_results = torch.topk(scores, k=10)
        top_indices = top_results.indices.tolist()
        
        # 根據索引找到對應的 Doc ID
        top_doc_ids = [doc_ids[idx] for idx in top_indices]
        
        # 轉成字串格式 "MED-xxx MED-xxx ..."
        top_10_str = " ".join(top_doc_ids)
        output_rows.append({"Query": qid, "Doc_ID": top_10_str})

    # 6. 存檔
    submission_df = pd.DataFrame(output_rows)
    output_filename = "submission.csv"
    submission_df.to_csv(output_filename, index=False)
    
    print("-" * 30)
    print(f"✅ Success! Generated {output_filename}")
    print("Preview of the first 5 rows:")
    print(submission_df.head())

if __name__ == "__main__":
    main()
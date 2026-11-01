# RAG 教學影片智慧搜尋（Lecture Video RAG）

從一支英文教學影片出發：用 Whisper 取得語音文字、用 PaddleOCR 取得投影片文字，
切分並對齊後轉成向量存入 Chroma，最後可以用問題搜尋影片片段，並由 Gemini 產生簡短回答。

## 檔案說明

| 檔案 | 功能 | 輸入 → 輸出 |
|---|---|---|
| `extract.py` | 每秒截圖 1 張、Whisper 語音轉文字、PaddleOCR 辨識畫面文字（含過濾 logo、亂碼） | `video.mp4` → `whisper.json`、`ocr.json`、`frames/`、`meta.json` |
| `build_chunks.py` | 清理、分段（約 45 秒）、對齊（語音與投影片依時間配對） | `whisper.json`、`ocr.json` → `chunks.jsonl` |
| `add_captions.py` | （選用）用 BLIP 為每個片段的截圖產生描述，補充 OCR 看不到的圖片內容 | `chunks.jsonl` → 更新後的 `chunks.jsonl` |
| `build_db.py` | 用 BGE-M3 轉成 1024 維向量，存入 Chroma | `chunks.jsonl` → `output/chroma_db/` |
| `search.py` | 輸入問題，顯示最接近的 3 個影片片段（時間、距離、截圖） | 問題 → 片段 |
| `answer.py` | 先檢索，再把片段交給 Gemini，產生附時間點的簡短回答 | 問題 → 回答與來源 |
| `.gitignore` | 排除 `venv/`、`output/`、影片與金鑰檔，避免上傳 | — |

`extract.py` 的選項：

- `--skip-whisper`：重新截圖並做 OCR，保留舊的 `whisper.json`
- `--ocr-only`：只對現有截圖重做 OCR

## 使用的技術

| 任務 | 技術 | 說明 |
|---|---|---|
| 分離音訊、截圖 | FFmpeg | 每秒 1 張截圖 |
| 語音轉文字 | faster-whisper（`small.en`） | int8，輸出短句與時間 |
| 畫面文字辨識 | PaddleOCR | 信心度 ≥ 0.7、至少 4 字元、排除底部 6% 的 logo 區域 |
| 減少 OCR 次數 | Pillow | 畫面幾乎沒變就沿用上一次結果 |
| 分段 | Python | 約 45 秒（上限 180 字）、不切斷句子、重疊 1 句 |
| 對齊 | Python | 時間重疊 ≥ 3 秒的投影片文字附加到片段 |
| 圖片描述（選用） | BLIP（`blip-image-captioning-base`） | 補充 OCR 看不到的插圖內容 |
| 向量化 | BAAI/bge-m3 | 1024 維、多語言、最長 8192 tokens |
| 向量資料庫 | Chroma | 內部以 SQLite 儲存，另含 id、文字、metadata |
| 回答生成 | Gemini Flash-Lite（`google-genai`） | 只閱讀檢索到的片段，找不到就回答找不到 |

向量化用的文字格式：`Slide: … Visual: … Speech: …`

## 環境

- 作業系統：Windows
- 無 GPU，全部使用 CPU 執行
- Python 3.11（建議）。太新的版本可能裝不到 PaddlePaddle（我們在 3.15 遇到此問題）
- 使用虛擬環境 `venv`
- FFmpeg：`winget install Gyan.FFmpeg`

## 安裝

```
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 執行順序

把影片命名為 `video.mp4`，放在專案資料夾，然後依序執行：

```
python extract.py video.mp4
python build_chunks.py
python add_captions.py      （選用，需要時才執行）
python build_db.py
python search.py
python answer.py
```

第一次執行時會下載模型（Whisper、PaddleOCR、BGE-M3 約 2 GB、BLIP 約 1 GB），需要網路。

## 使用 answer.py 前

到 https://aistudio.google.com/apikey 取得 API key，並設定環境變數：

```
set GEMINI_API_KEY=你的金鑰
```

**不要把金鑰寫進程式碼或上傳到 GitHub。**

## 注意事項

- 影片與 `output/` 資料夾（截圖、JSON、資料庫）不在這個儲存庫裡，需要自己執行程式產生。
- 修改 `extract.py` 或 `build_chunks.py` 之後，後面的步驟都要重新執行。

## 目前限制

- OCR 與 BLIP 都無法判斷「哪個數字對應哪個圖示」。
- 只用一支約 6 分鐘的影片測試，分段長度與距離門檻尚未系統性比較。
- 測驗題生成尚未實作。

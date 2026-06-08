# FastAPI + React RAG Demo

Demo UI cho chatbot RAG và RAGAS evaluation.

## Chạy backend

Từ thư mục gốc project:

```bash
pip install -r requirements.txt
uvicorn group_project.web_demo.backend.main:app --reload --host 0.0.0.0 --port 8000
```

Backend API:

- `GET /api/health`
- `GET /api/config`
- `POST /api/chat`
- `POST /api/evaluate`

## Chạy frontend

Terminal khác:

```bash
cd group_project/web_demo/frontend
npm install
npm run dev
```

Mở: <http://localhost:5173>

## Các nút/cấu hình trên giao diện

- Chọn generation model.
- Chỉnh `temperature`, `top_p`.
- Bật/tắt reranking.
- Chỉnh `top_k` và fallback threshold.
- Chọn số golden cases để chạy RAGAS.
- Bấm `Hỏi RAG` để xem câu trả lời và source chunks.
- Bấm `Chạy RAGAS` để xem bảng `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`.

## Lưu ý demo

- Cần `.env` có `OPENAI_API_KEY` để generation và RAGAS judge chạy thật.
- Cần có dữ liệu trong `data/standardized/` và index Weaviate/PageIndex đã chuẩn bị trước để retrieval có context tốt.
- Nếu thiếu OpenAI key, backend vẫn trả lời bằng extractive fallback khi có chunks.

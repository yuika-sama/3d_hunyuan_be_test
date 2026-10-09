# Kế hoạch triển khai dự án lên Runpod Serverless

> Tài liệu này lưu thiết kế triển khai ban đầu. API contract hiện hành và các mặc định shape/mesh mới nằm trong mục **API Reference** của [`README.md`](../README.md#-4-api-reference).

## 1. Mục tiêu và phạm vi

Đưa toàn bộ bốn chức năng hiện tại vào **một Runpod Serverless endpoint**:

1. Kiểm tra ảnh NSFW.
2. Sinh mô tả ảnh bằng BLIP.
3. Sinh/chỉnh mô tả bằng Ollama + LLaVA 7B.
4. Sinh mô hình 3D bằng Hunyuan3D, giữ nguyên thiết lập chất lượng hiện tại.

Các yêu cầu đã chốt:

- Tải thấp: khoảng 10–20 request/ngày.
- Không cần xử lý đồng thời.
- Ưu tiên chi phí thấp nhất nhưng không giảm chất lượng đầu ra 3D.
- File GLB được trả về qua API.
- Repository GitHub đang private.
- Docker image lưu trên GHCR.
- Mỗi commit vào nhánh `main` tự động build và cập nhật Runpod.
- Frontend chỉ dùng để kiểm thử cục bộ, không được commit lên GitHub.
- Quy mô đồ án, chưa cần kiến trúc nhiều service hoặc khả năng chịu tải lớn.

## 2. Kiến trúc đề xuất

Sử dụng một Docker image và một Runpod queue-based Serverless worker. Không tiếp tục chạy năm HTTP server nội bộ như cách chạy local hiện tại.

```text
Client / Frontend local
        |
        | POST /run
        v
Runpod Serverless Endpoint
        |
        v
runpod_handler.py
        |
        +-- action=analyze          -> NSFW model, CPU
        +-- action=caption          -> BLIP model, CPU
        +-- action=custom_describe  -> Ollama + LLaVA 7B, GPU
        +-- action=generate3d       -> Hunyuan3D, GPU
        |
        +-- /stream/{job_id} -> tiến độ, runtime và dữ liệu GLB
```

Một handler trực tiếp sẽ đơn giản hơn việc giữ `gateway.py` cùng các Flask server và gọi qua `localhost`. Các server local cũ có thể được giữ để phục vụ phát triển trên Windows, nhưng không dùng làm entrypoint của container Runpod.

## 3. API dự kiến

### 3.1. Request

```json
{
  "input": {
    "action": "generate3d",
    "image_base64": "...",
    "prompt": "optional"
  }
}
```

Các giá trị `action` hợp lệ:

- `analyze`
- `caption`
- `custom_describe`
- `generate3d`

Handler phải kiểm tra action, định dạng ảnh và kích thước payload trước khi chạy model.

### 3.2. Response dạng stream

Dùng endpoint bất đồng bộ `/run`, sau đó frontend đọc `/stream/{job_id}`. Không dùng `/runsync` cho GLB vì payload base64 có thể lớn hơn giới hạn response đồng bộ.

Các event mẫu:

```json
{"type":"progress","stage":"preprocess","elapsed_ms":438}
{"type":"progress","stage":"shape_generation","elapsed_ms":17432}
{"type":"progress","stage":"texture_generation","elapsed_ms":28110}
{"type":"file_meta","name":"model.glb","size":4656916,"chunks":9,"sha256":"..."}
{"type":"file_chunk","index":0,"data":"...base64..."}
{"type":"done","timings":{"total_ms":52140}}
```

Chia file theo block khoảng **512 KiB dữ liệu thô**. Sau khi mã hóa base64, mỗi event vẫn thấp hơn giới hạn 1 MB cho một stream chunk. Frontend ghép các block theo `index`, kiểm tra SHA-256 rồi tạo file GLB để tải xuống hoặc hiển thị.

Với các action không sinh file, event cuối trả JSON thông thường.

## 4. Quản lý vòng đời model

Model được lazy-load theo action để request nhẹ không phải nạp tất cả model:

- NSFW và BLIP chạy trên CPU.
- Hunyuan3D chỉ nạp khi gọi `generate3d`.
- Ollama daemon khởi động cùng container, nhưng `llava:7b` chỉ nạp khi cần.
- Sau khi LLaVA hoàn thành, gọi unload bằng `keep_alive=0` để giải phóng VRAM trước khi chạy Hunyuan3D.
- Runpod cấu hình concurrency bằng 1, nên chưa cần hàng đợi hoặc lock riêng trong ứng dụng.

Giữ nguyên các tham số Hunyuan hiện tại:

- Texture: bật.
- Octree resolution: `256`.
- Inference steps: `5`.
- Guidance scale: `5`.
- Face reduction: mặc định tối đa `200000` để giữ chi tiết hình học.

Các đường dẫn Windows cố định trong mã hiện tại phải được đổi thành đường dẫn Linux hoặc thư mục tạm do Python tạo.

## 5. Lưu trữ model: Network Volume hay Hugging Face

### 5.1. Network Volume

Ưu điểm:

- Kiểm soát hoàn toàn file model và phiên bản.
- Có thể dùng cho model riêng hoặc model không có trên Hugging Face.
- Model tồn tại ngoài vòng đời worker.

Nhược điểm:

- Có chi phí lưu trữ cố định dù tải chỉ 10–20 request/ngày.
- Endpoint bị giới hạn theo vùng có volume, làm giảm tập GPU khả dụng và tăng khả năng thiếu GPU.
- Cần quy trình tải, đồng bộ và quản lý model trên volume.
- Không đem lại nhiều lợi ích khi model đã có sẵn trên Hugging Face.

### 5.2. Hugging Face + Runpod Cached Models

Ưu điểm:

- Không phải tự quản lý volume.
- Không khóa endpoint vào một vùng chứa volume.
- Model được tải/cached gần worker và có thể dùng khi scale từ 0.
- Phù hợp với tải thấp và các model public hiện tại.

Nhược điểm:

- Cold start đầu tiên vẫn có thể lâu nếu cache chưa sẵn sàng.
- Model public có thể thay đổi nếu tham chiếu `main`; vì vậy phải pin commit SHA/revision.
- Model private hoặc gated cần cấu hình Hugging Face token.
- Layout model của Ollama không tương thích trực tiếp với Hugging Face cache.

### 5.3. Quyết định

Chọn **Hugging Face + Runpod Cached Models**, không dùng Network Volume trong giai đoạn đồ án.

- Hunyuan3D dùng repository chính chủ Tencent, không dùng bản cộng đồng: mặc định `tencent/Hunyuan3D-2.1` / `hunyuan3d-dit-v2-1`.
- BLIP dùng `Salesforce/blip-image-captioning-large`.
- NSFW dùng `strangerguardhf/nsfw_image_detection`.
- Tất cả tham chiếu phải pin theo commit SHA sau khi xác nhận phiên bản đang chạy tốt.
- Ollama binary và model `llava:7b` được đóng sẵn trong Docker image vì định dạng lưu trữ của Ollama khác Hugging Face cache.

Không cần upload lại Hunyuan lên tài khoản Hugging Face cá nhân. Chỉ cần token nếu Runpod yêu cầu truy cập model private/gated; các model public có thể cấu hình bằng model reference trực tiếp.

## 6. Phần cứng và cấu hình Serverless

### Cấu hình ban đầu

- GPU tier: **24 GB VRAM, Standard**.
- GPU ưu tiên: **RTX 3090**, sau đó **L4**, rồi **RTX A5000** tùy khả dụng.
- `workersMin = 0` để scale về 0 khi không dùng.
- `workersMax = 1`.
- Concurrency mỗi worker: `1`.
- Idle timeout: khoảng `5` giây.
- Execution timeout: `20` phút để đủ thời gian cold start và sinh texture.
- Không gắn Network Volume.

Không chọn GPU 16 GB ngay từ đầu. Hunyuan3D shape + texture cần vùng nhớ sát giới hạn này và container còn phải có dư địa cho runtime, rasterizer và phân mảnh VRAM. GPU 24 GB là điểm cân bằng chi phí/rủi ro hợp lý.

Chỉ nâng lên tier 48 GB nếu benchmark thực tế cho thấy 24 GB bị OOM sau khi đã xác nhận LLaVA được unload đúng cách.

### Ước tính chi phí compute

Theo mức tham khảo khoảng **0,69 USD/giờ** cho Serverless Standard 24 GB:

- 5 phút xử lý: khoảng 0,058 USD/job.
- 10 phút xử lý: khoảng 0,115 USD/job.
- 20 job/ngày, mỗi job 5 phút: khoảng 34,5 USD/tháng.

Đây chỉ là ước tính; chi phí thật phụ thuộc cold start, thời gian sinh texture và giá tại thời điểm triển khai.

## 7. Docker image

Các file tối thiểu cần thêm:

```text
Dockerfile
.dockerignore
start.sh
requirements-runpod.txt
runpod_handler.py
test_handler.py
```

Yêu cầu cho image:

1. Base image Linux có CUDA/PyTorch tương thích với Hunyuan3D.
2. Build theo kiến trúc `linux/amd64`.
3. Cài dependency Python đã pin phiên bản.
4. Build/cài rasterizer native của Hunyuan3D.
5. Cài Ollama.
6. Tải `llava:7b` vào image trong giai đoạn build.
7. `start.sh` khởi động Ollama daemon, chờ health check rồi chạy Runpod worker.
8. Không copy model Hugging Face lớn vào image.
9. Không copy file output, virtual environment, cache, ảnh mẫu hoặc secret vào image.

Trước khi push, chạy local handler test và build image để bắt lỗi import/path. Việc kiểm thử GPU cuối cùng phải thực hiện bằng một request thật trên Runpod.

## 8. Các giai đoạn triển khai

### Giai đoạn 1 — Chuẩn hóa handler

- Thêm `runpod_handler.py` làm entrypoint duy nhất.
- Route bốn action vào code/model hiện có, không dựng lại logic model.
- Thay đường dẫn Windows bằng đường dẫn Linux/tạm.
- Chuẩn hóa JSON error và validation đầu vào.
- Viết một test nhỏ cho route action, validation và chunk GLB.

### Giai đoạn 2 — Tối ưu bộ nhớ

- Lazy-load từng model.
- Chạy NSFW và BLIP trên CPU.
- Đảm bảo LLaVA unload khỏi GPU sau request.
- Chỉ giữ Hunyuan trên GPU khi `generate3d` đang chạy.
- Log VRAM trước/sau mỗi giai đoạn để xác nhận 24 GB đủ dùng.

### Giai đoạn 3 — Progress và runtime

Backend phát event cho các giai đoạn có thật:

- Worker/model initialization.
- Decode input.
- Preprocess.
- NSFW/caption/LLaVA inference tùy action.
- Shape generation.
- Texture generation.
- Mesh cleanup/reduction.
- GLB export.
- Chunking/streaming.
- Total.

Frontend chỉ hiển thị timing backend gửi về; đồng thời có thể hiển thị thêm queue delay, Runpod execution time và tổng thời gian phía client.

### Giai đoạn 4 — Docker và kiểm thử local

- Hoàn thiện Dockerfile và `.dockerignore`.
- Build `linux/amd64`.
- Chạy test handler không GPU.
- Kiểm tra container khởi động được và Ollama health check thành công.
- Push image thử nghiệm lên GHCR.

### Giai đoạn 5 — Tạo tài nguyên Runpod lần đầu

- Tạo credential để Runpod đọc private GHCR image.
- Tạo Runpod template trỏ đến image bằng immutable tag.
- Cấu hình Hugging Face cached model references.
- Tạo một Serverless endpoint dùng template đó.
- Áp dụng cấu hình GPU và autoscaling ở mục 6.
- Gọi thử cả bốn action.
- Mở GLB trả về để xác nhận file không hỏng.
- Ghi lại cold start, thời gian inference, VRAM và chi phí thực tế.

### Giai đoạn 6 — CI/CD qua GitHub Actions

Workflow chạy khi push vào `main`:

1. Checkout repository.
2. Chạy test nhẹ.
3. Đăng nhập GHCR bằng `GITHUB_TOKEN`.
4. Build image `linux/amd64` bằng Docker Buildx.
5. Push immutable tag:

   ```text
   ghcr.io/yuika-sama/3d_hunyuan_be_test:sha-<commit_sha>
   ```

6. Đọc image hiện tại của Runpod template để lưu thông tin rollback.
7. Gọi Runpod REST API cập nhật `imageName` của template sang tag mới; Runpod thực hiện rolling release cho endpoint dùng template.
8. Gửi một smoke test rẻ bằng action `analyze`.
9. Nếu smoke test thất bại, PATCH template về image tag trước đó và làm workflow thất bại.

Không dùng tag `latest` làm nguồn deploy chính vì khó xác định chính xác phiên bản và rollback.

### GitHub secrets/variables

Secrets:

- `RUNPOD_API_KEY`

Variables:

- `RUNPOD_TEMPLATE_ID`
- `RUNPOD_ENDPOINT_ID`

GitHub Actions cần quyền:

```yaml
permissions:
  contents: read
  packages: write
```

Runpod cần một GitHub Personal Access Token chỉ có quyền tối thiểu `read:packages` để pull image private từ GHCR. Token này cấu hình trong Runpod Container Registry Authentication, không đưa vào repository hoặc Docker image.

## 9. Frontend kiểm thử local-only

Tạo frontend HTML/JS đơn giản từ file test hiện có, với các chức năng:

- Chọn action.
- Chọn ảnh.
- Gửi request lên Runpod `/run`.
- Poll status và đọc `/stream/{job_id}`.
- Hiển thị progress và runtime từng giai đoạn.
- Ghép các GLB chunk.
- Kiểm tra SHA-256.
- Xem trước GLB và tải file.

Frontend không được commit. Thêm đường dẫn file/thư mục frontend vào:

```text
.git/info/exclude
```

Không thêm vào `.gitignore`, vì thay đổi `.gitignore` vẫn bị commit và làm lộ việc tồn tại của file local. Sau khi tạo frontend, kiểm tra bằng `git status` và `git ls-files` để chắc chắn file không được track.

Không lưu Runpod API key vào source, `localStorage` hoặc query string. Chỉ nhập key tạm trong UI khi test hoặc dùng một proxy local nếu cần dùng lâu dài.

## 10. Bảo mật và vận hành

- Không commit Runpod API key, Hugging Face token hoặc GHCR PAT.
- Không in secret trong log GitHub Actions hoặc log worker.
- Giới hạn kích thước ảnh đầu vào trước khi decode.
- Chỉ chấp nhận các MIME ảnh dự kiến.
- Xóa file tạm sau khi stream xong hoặc khi job thất bại.
- Trả lỗi có `stage` và thông báo đủ để debug, không trả stack trace cho client.
- Pin Docker base image, Python dependency và Hugging Face revision.
- Giữ `workersMin=0`; chỉ tăng lên 1 nếu cold start ảnh hưởng buổi demo.

## 11. Tiêu chí nghiệm thu

- Một Runpod Serverless endpoint xử lý đủ bốn action.
- Endpoint scale về 0 khi không có request.
- Cấu hình 24 GB hoàn thành `generate3d` mà không OOM.
- Chất lượng Hunyuan3D giữ nguyên thiết lập hiện tại.
- GLB trả hoàn toàn qua API, kiểm tra SHA-256 đúng và mở được.
- Frontend hiển thị runtime từng giai đoạn.
- Push vào `main` tự động build, push GHCR, cập nhật template và smoke test.
- Có rollback tự động nếu smoke test thất bại.
- Frontend test không xuất hiện trong Git history.
- Không có secret trong repository, image hoặc log.

## 12. Thứ tự thực hiện khuyến nghị

1. Viết handler và test nhỏ.
2. Chuẩn hóa đường dẫn/model loading.
3. Thêm streaming GLB và timing.
4. Đóng Docker, kiểm tra local.
5. Push image đầu tiên lên GHCR.
6. Tạo template và endpoint Runpod thủ công lần đầu.
7. Benchmark request thật trên GPU 24 GB.
8. Thêm GitHub Actions sau khi deployment thủ công đã chạy ổn định.
9. Tạo frontend local-only và kiểm thử end-to-end.

Thứ tự này tránh debug đồng thời code, Docker, Runpod và CI/CD.

## 13. Tài liệu tham khảo

- [Runpod Serverless model caching](https://docs.runpod.io/serverless/endpoints/model-caching)
- [Runpod pricing](https://www.runpod.io/pricing)
- [Runpod REST API specification](https://rest.runpod.io/v1/openapi.json)
- [Tencent Hunyuan3D-2 repository](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
- [Tencent Hunyuan3D-2.1 on Hugging Face](https://huggingface.co/tencent/Hunyuan3D-2.1)


# Hunyuan image-to-3D worker

Ảnh vật thể → NSFW (action `pipeline`) → tách nền → shape → texture → GLB.
Repo chạy bằng Runpod Serverless hoặc gateway local dành cho thử nghiệm.
Không có bước rig/sinh animation nhân vật.

**Triển khai và kiểm thử:** [Hướng dẫn Runpod và bàn giao Bookaria](docs/runpod-serverless-deployment-plan.md).

## Cấu hình thực tế

| Thành phần | Mặc định |
|---|---|
| Shape | `tencent/Hunyuan3D-2.1`, `hunyuan3d-dit-v2-1` |
| Paint | `tencent/Hunyuan3D-2`, `hunyuan3d-paint-v2-0` |
| Shape steps / octree / faces | 50 / 384 / 40.000 |
| Texture | 2048 px, delight tắt |
| Turbo shape | Chỉ bật FlashVDM khi subfolder shape có `turbo`; mặc định 5 steps khi đó |
| Runtime | Python 3.10, CUDA 12.1, torch 2.5.1, torchvision 0.20.1, Runpod SDK 2.3.0 |

GLB từ Paint dùng `PBRMaterial` với metallic `0`, roughness `0.9`, giữ texture và UV. Đây là mặc định cho vật liệu màu đơn, không sinh thêm bản đồ metallic/roughness. Kiểm thử CPU: `python -m unittest test_material_export -v`.

Pipeline lai shape 2.1 + paint 2.0; không phải full PBR 2.1. Model được chuyển giữa CPU/GPU
để giảm VRAM. Cần benchmark cả RAM, VRAM và cold start trên GPU thực tế;
24 GB VRAM là cấu hình thử nghiệm, không phải cam kết đủ cho mọi ảnh.

PNG đã có alpha được giữ lại; ảnh đục chạy rembg. Mesh cleanup hiện **có**
FloaterRemover, DegenerateFaceRemover và FaceReducer. FloaterRemover có thể bỏ chi tiết
tách rời nhỏ, cần kiểm tra ảnh có tai/móng/đuôi. Khi texture được yêu cầu, pipeline lỗi
hoặc không có texture phải báo lỗi, không trả mesh trắng như một kết quả thành công.

## API Runpod

Base URL: `https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}`.
Header: `Authorization: Bearer {RUNPOD_API_KEY}`. Key chỉ ở server hoặc môi trường test,
không đưa vào browser/Studio.

| Action | Chức năng |
|---|---|
| `analyze` | Kiểm tra NSFW; trả event `result` |
| `pipeline` | Kiểm tra ảnh nguồn và mọi ảnh texture được dùng, rồi sinh GLB có texture |
| `generate3d` | Primitive nội bộ cho input đã được kiểm duyệt; không tự chạy NSFW |

Dùng `pipeline` cho tác giả Bookaria. Gửi một ảnh dùng chung cho shape và texture:

```json
{
  "input": {
    "action": "pipeline",
    "image_base64": "<base64 ảnh>",
    "texture_resolution": 1024,
    "face_count": 40000,
    "seed": 1234
  },
  "policy": { "executionTimeout": 1200000, "ttl": 1800000 }
}
```

`POST /run` trả job ID; đọc `GET /stream/{id}` và trạng thái terminal.
Đây là API JSON polling, không phải một kết nối SSE.
Không tự POST lại khi submit timeout: job trước có thể đã được nhận và tính phí.

### Input và giới hạn

- Ảnh nguồn: `image_base64` hoặc alias `image`; Base64 chuẩn hoặc data URI.
- Texture tùy chọn: `texture_image_base64` / `texture_image`, hoặc list
  `texture_images_base64` / `texture_images` gồm 1–6 ảnh. Không gửi đồng thời ảnh đơn và list.
  Bỏ texture input thì dùng ảnh nguồn. Multiview cần kiểm chứng chất lượng trên GPU.
- Mỗi ảnh giải mã tối đa **6 MiB, 16 megapixel**. Tổng JSON `input` tối đa **9.5 MB**.
  Runpod `/run` giới hạn toàn body 10 MB; Base64 làm tăng khoảng 4/3.
  Kiểm tra toàn envelope trước khi gửi, không nhân đôi ảnh khi dùng chung shape/texture.
- Các field lạ, alias mâu thuẫn, chuỗi thay cho số/boolean đều bị từ chối.
- `seed`: integer 0–4294967295; `num_inference_steps`: integer 1–100.
- `octree_resolution`: integer 128–512; `face_count`: integer 1000–200000.
- `guidance_scale`: số hữu hạn 0–20; `bake_exp`: integer 1–16.
- `texture_resolution` (alias `texture_size`): 256, 512, 1024 hoặc 2048.
- `texture`, `enable_delight`: JSON boolean. `pipeline` luôn bật texture.
- `texture_subfolder` (alias `subfolder`): `hunyuan3d-paint-v2-0` hoặc
  `hunyuan3d-paint-v2-0-turbo`. Nếu paint thường không load, loader hiện có fallback turbo.

### Stream output

1. `progress`: stage, elapsed_ms, thông tin VRAM nếu có.
2. `result`: kết quả kiểm duyệt; `texture_moderation` là object với một ảnh,
   là list khi có nhiều ảnh texture.
3. `file_meta`: name, size, chunks, sha256.
4. `file_chunk`: index, data Base64; mỗi chunk gốc 512 KiB.
5. `done`: timings và VRAM.

Lỗi có `type: "error"`, `stage`, `error`. SDK Runpod 2.3.0 đã khóa phiên bản
nhận trường `error` của generator làm lỗi job; client phải xử lý cả terminal
FAILED/CANCELLED/TIMED_OUT và lỗi ở stream. Có test đi qua SDK thực với HTTP mock.
Thành công phải đủ chunk, đúng size/hash, GLB đọc được và có texture khi yêu cầu.

## Local gateway

`run.bat` chỉ bind loopback, không tự mở ngrok. Đặt `GATEWAY_API_KEY` trước khi chạy;
gửi `Authorization: Bearer <key>` cho mọi API ngoài `/health`.
Đây là khóa dịch vụ nội bộ, không thay thế tài khoản/quyền sở hữu/quota của Bookaria.

| API | Output |
|---|---|
| GET /health | Trạng thái tiến trình gateway, không kiểm chứng GPU |
| POST /analyze | multipart image → NSFW JSON |
| POST /generate3d | multipart image + texture_image tùy chọn → GLB sau kiểm duyệt |
| POST /pipeline/process-all | Cùng kiểm duyệt → lưu outputs/*.glb, trả task_id/glb_url |
| GET /pipeline/result/{task_id}.glb | Tải file; cần cùng khóa dịch vụ |

Local chỉ nhận PNG/JPEG, mỗi ảnh tối đa 6 MiB/16 MP. Các model local dùng cấu hình
mini-turbo trong api_server.py nên chất lượng/thời gian có thể khác Runpod.
Không expose trực tiếp service port 5001/5003. Local outputs cần dọn định kỳ;
khi tích hợp phải nhập file vào storage Bookaria.

## Test

```bash
python -m pytest test_handler.py test_gateway.py test_runpod_smoke.py -v
```

Các unit/mock test không tải trọng số hay gọi GPU. Hướng dẫn cài môi trường test CPU,
triển khai, chạy smoke GPU, rollback và tích hợp nằm trong [tài liệu bàn giao](docs/runpod-serverless-deployment-plan.md).

# Triển khai Runpod và bàn giao Bookaria

Cập nhật 18/11/2026. Tài liệu này thay thế kế hoạch cũ còn nhắc các dịch vụ đã bỏ.
Phạm vi bản sửa: worker, gateway thử nghiệm, tests và quy trình kiểm chứng.
**Chưa triển khai hoặc nối code vào Bookaria.** Thành viên chạy test GPU sau khi push.

## 1. Những thay đổi cần biết khi nhận bản sửa

- Chuẩn hóa field/alias, kiểm tra kiểu và giới hạn tài nguyên trước khi load model.
- Mọi ảnh trong list texture của `pipeline` đều được kiểm duyệt. Response moderation lỗi/
  thiếu boolean không được xem là an toàn.
- Override checkpoint paint được truyền đúng vào loader; giải phóng checkpoint cũ trước khi thay.
- Shape được offload trong `finally`, kể cả đóng generator giữa chừng.
- Local paint không load hoặc trả model không có texture đều báo lỗi.
- Gateway kiểm tra ảnh thật, đọc có giới hạn, kiểm duyệt cả hai route sinh GLB và kiểm tra header GLB.
- Gateway yêu cầu `GATEWAY_API_KEY`; run.bat chỉ mở loopback, không tự chạy ngrok.
- Local worker khóa việc tạo model để không có hai thread đồng thời di chuyển model CPU/GPU.
- Docker image nhẹ tối đa: tách biệt toàn bộ weights/models ra khỏi build layer. Weights được lưu trữ bền vững trên Network Volume (/runpod-volume) thay vì nướng cứng vào image.
- Khi khởi động, worker tự động phát hiện Network Volume (/runpod-volume) và trỏ các cache directory (HF_HOME, TORCH_HOME, U2NET_HOME, HY3DGEN_MODELS) vào volume.
  Nếu model chưa có trên volume, worker tự động tải về volume ở lần chạy đầu tiên. Ở các lần khởi động tiếp theo, worker tái sử dụng cache từ volume, giảm thiểu Cold Start xuống chỉ vài giây.
  Log GPU có các mốc UV, render, multiview, bake và inpaint để xác định bước đang chạy.
  `PRELOAD_MODELS=0` cho phép bỏ qua bước kiểm tra/tải trước lúc khởi động và chỉ tải khi có yêu cầu đầu tiên.
- Khóa torch/torchvision/Runpod SDK; chưa phải lockfile đầy đủ cho mọi dependency native.
- Giữ error-event contract: SDK 2.3.0 nhận `error` và báo job lỗi; đã test qua SDK với HTTP mock.
- CI không bỏ qua deploy HTTP lỗi và không dùng regex COMPLETED làm bằng chứng inference.
  Smoke chỉ submit một job rồi đọc kết quả, không tạo năm job khi cold start.

**Thay đổi contract:** input lạ/sai kiểu, ảnh >6 MiB hoặc >16 MP, JSON input >9.5 MB
nay bị từ chối; gateway cần Bearer key; `/generate3d` gateway nay có kiểm duyệt.
Worker `generate3d` vẫn là primitive cho backend tin cậy với ảnh đã được kiểm duyệt.
Giữ `pipeline` cho luồng người dùng.

## 2. Kiểm tra CPU trước khi push

Từ thư mục gốc repo, dùng Python 3.10/3.11:

```powershell
python -m venv .venv-test
.\.venv-test\Scripts\python.exe -m pip install --upgrade pip
.\.venv-test\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-test\Scripts\python.exe -m pip install "numpy<2" pillow httpx runpod==2.3.0 pytest fastapi python-multipart
.\.venv-test\Scripts\python.exe -m pytest test_handler.py test_gateway.py test_runpod_smoke.py -v
git diff --check
git status --short
```

Linux: tạo venv tương tự, thay đường dẫn Python bằng `.venv-test/bin/python`.
Không cài requirements-runpod.txt trên máy chỉ chạy unit test: file đó gồm CUDA/native/model stack.
Test mock kiểm tra routing, NSFW, decode, checkpoint, offload, local texture error, auth và
stream/checksum/SDK errors; không phải benchmark AI.

Kiểm chứng local ngày 18/11/2026: **48 tests + 44 subtests pass**, 76 file Python compile,
workflow YAML/Bash syntax hợp lệ, git diff --check sạch. Môi trường Python 3.11 / torch CPU 2.5.1;
chưa chạy Docker CUDA, GitHub Actions hoặc inference Runpod thật.

Chỉ stage các file source/docs/test đã sửa; không stage venv, keys hay file GLB.
Không commit `RUNPOD_API_KEY` hoặc `GATEWAY_API_KEY`.

## 3. Build và cấu hình Runpod

Repo đã có GitHub Actions chạy test → Docker build → push GHCR → cập nhật Runpod.
Workflow kích hoạt khi push nhánh `main` hoặc Run workflow thủ công.

1. Bật Actions và quyền GitHub Packages cho repo. Image sẽ mang tag
   `ghcr.io/<owner>/<repo>:sha-<commit>`. Dùng tag commit để đối chiếu/rollback.
2. Nếu GHCR private, cấu hình registry credentials trong Runpod; nếu public, kiểm tra
   worker kéo được image. Không đưa GitHub token vào image.
3. Dùng **Serverless queue-based endpoint** chạy Dockerfile của repo; không phải HTTP load-balancer.
   Container entrypoint là `start.sh` → `runpod_handler.py`.
4. Ban đầu đặt min workers 0, max workers 1, concurrency 1. Thử tier 24 GB VRAM;
   nếu OOM, xem stage/RAM trước khi đổi GPU. Repo lai/offload chưa có số đo cam kết.
5. Gắn Network Volume (mount tại `/runpod-volume`). Worker sẽ tự động trỏ toàn bộ cache
   (`HF_HOME`, `TORCH_HOME`, `U2NET_HOME`, `HY3DGEN_MODELS`) vào volume. Dung lượng Container Disk chỉ cần
   đặt tối thiểu khoảng 15–20 GB (vì image không còn chứa model nặng); Network Volume nên từ 50–100 GB.
   Tải model lần đầu sẽ lưu vĩnh viễn vào Network Volume, các lần worker sau sẽ khởi động ngay lập tức.
6. Đặt GitHub Secret `RUNPOD_API_KEY` có quyền `serverless:write` và quyền gọi job
   trên endpoint thử nghiệm. Đặt GitHub Variables `RUNPOD_ENDPOINT_ID` trỏ tới endpoint đó.
   Workflow cập nhật trường `image` trực tiếp trên endpoint qua Runpod REST API v2.
7. Khi thiếu key hoặc endpoint ID, workflow chỉ build/push và báo skip deploy.
   Khi cập nhật API thất bại, workflow báo lỗi. Có thể đổi image trực tiếp trong
   Runpod Console rồi chạy smoke riêng.
8. Xác nhận image tag ở dashboard/worker trước khi kiểm thử. Worker nóng cũ có thể vẫn phục vụ
   trong lúc rollout; analysis smoke không chứng minh mọi worker đã dùng image mới.

Biến môi trường model (giữ mặc định khi test lần đầu):

```text
HUNYUAN_SHAPE_MODEL=tencent/Hunyuan3D-2.1
HUNYUAN_SHAPE_SUBFOLDER=hunyuan3d-dit-v2-1
HUNYUAN_TEX_MODEL=tencent/Hunyuan3D-2
HUNYUAN_TEX_SUBFOLDER=hunyuan3d-paint-v2-0
DEFAULT_OCTREE_RESOLUTION=384
DEFAULT_FACE_COUNT=40000
DEFAULT_TEXTURE_RESOLUTION=2048
HUNYUAN_ENABLE_DELIGHT=0
PRELOAD_MODELS=1
```

Các DEFAULT_* là cấu hình người vận hành tin cậy; giữ trong giới hạn API của README.
Đường CPU test không chứng minh CUDA rasterizer/build image chạy được.
Không bật nhiều job/GPU cùng lúc với mô hình dùng offload.

## 4. Chạy test thật trên Runpod

Đặt key vào môi trường shell riêng; không ghi key trong lệnh đã commit, log hay gửi vào chat.
Ví dụ PowerShell 7 nhập key ẩn:

```powershell
$env:RUNPOD_API_KEY = Read-Host "Runpod API key" -MaskInput
$env:RUNPOD_ENDPOINT_ID = "endpoint-id-cua-ban"
python -m pip install httpx
python runpod_smoke.py 3d_generative/Hunyuan3D-2-main/assets/example_images/004.png --action analyze
python runpod_smoke.py 3d_generative/Hunyuan3D-2-main/assets/example_images/004.png --action pipeline --output outputs/test-004.glb
```

Chạy các lệnh này có thể phát sinh phí Runpod. Analysis smoke kiểm tra kết quả NSFW thực,
**không kiểm tra sinh 3D**. Workflow tự chạy analysis smoke khi có endpoint ID;
thành viên phải chạy thêm lệnh `pipeline` và mở GLB.

Script chỉ gửi một POST /run, in job ID, đọc stream đến terminal và hết chunk.
Nó kiểm tra error/FAILED/CANCELLED/TIMED_OUT, done, đủ chunk, size, SHA-256,
GLB header/JSON, có mesh/texture và không tham chiếu tài nguyên ngoài.
Đây là kiểm tra cấu trúc cơ bản; vẫn phải mở bằng Viewer để kiểm chứng render.
Script kiểm tra GLB nguồn theo giới hạn **50 MiB** của Bookaria, tương đương tối đa
100 chunk 512 KiB. Trần **10 MiB** áp dụng cho GLB sau tối ưu trước khi phân phối. Không ghi đè file output đã có; dùng tên mới cho lần test khác.

Nếu timeout/mất kết nối, xem job ID vừa in ở dashboard; kiểm tra/hủy job cũ trước khi
chạy lại để tránh tính phí trùng. Khi POST timeout chưa có ID, kiểm tra queue Runpod,
không tự retry. Script dừng chờ sau 30 phút; policy job đặt execution 20 phút, ttl 30 phút.
Thời gian khởi động/tải weights có thể cần điều chỉnh theo môi trường thử.

Trong lúc test ghi:
- GPU, VRAM/RAM peak, image tag, checkpoint, seed, preset.
- Cold start và thời gian từng stage trong log; có OOM/fallback hay không.
- GLB trước tối ưu, GLB sau tối ưu Bookaria và số tam giác.
- Texture, mặt sau, chi tiết nhỏ, hướng và tỷ lệ vật thể, kết quả trên điện thoại.
- Một job lỗi rồi một job tốt tiếp theo để kiểm tra giải phóng bộ nhớ.

Dùng 12–20 ảnh đại diện sau khi một ảnh chạy được; không mở rộng worker trước khi kiểm tra
chi phí cho một model được chấp nhận. Người đọc sách dùng lại asset đã lưu, không gọi GPU.

## 5. Cách nối Bookaria sau khi test GPU đạt

Đây là phạm vi triển khai tiếp theo, chưa nằm trong bản sửa repo GPU:

```text
Studio chọn IMAGE → Bookaria kiểm tra quyền/quota → job GENERATE_MODEL
→ Runpod /run action=pipeline → lưu Runpod job ID → đọc stream
→ kiểm tra GLB → optimizeModel/storage hiện có → asset MODEL READY
→ Studio xem trước/chèn scene → Viewer hiện có
```

- Request từ Studio chỉ có sourceAssetId và preset; backend xác minh ảnh thuộc quyền user.
  Key Runpod ở backend. Không nhận URL tùy ý hoặc để browser gọi trực tiếp.
- Thêm DB enum/job schema/result/handler; đồng bộ contracts Studio và nhãn admin/notifications.
  Tái dùng pg-boss đang có, không cần thêm queue framework.
- Lưu provider job ID ngay khi submit thành công. Retry tiếp tục theo dõi job cũ;
  xử lý riêng trường hợp submit timeout chưa biết ID. Quota số lần tạo khác quota lưu file.
- /run giới hạn body 10 MB; encode Base64 làm tăng khoảng 4/3. Resize ảnh và kiểm tra toàn body.
  Một ảnh dùng chung shape/texture chỉ cần gửi một lần.
- Backend từ chối GLB nguồn >50 MiB. Sau tối ưu, GLB tối đa 10 MiB, texture tối đa
  1024 px/cạnh và mục tiêu 50k tam giác. Smoke chọn face_count=40000/texture=1024.
  Nếu file vẫn quá lớn, điều chỉnh preset trước; không tăng upload cap toàn hệ thống mù quáng.
- Backend đọc ảnh từ R2 bằng object key, chuẩn bị ảnh theo giới hạn Runpod rồi gửi Base64.
  Sau khi xác minh GLB, lưu bằng adapter R2 hiện có với MIME `model/gltf-binary` và
  metadata định dạng `glb`. Database giữ object key, Studio/Viewer dùng URL từ `R2_PUBLIC_URL`.
  Credentials R2 ở backend. Cấu hình CORS R2 cho origin thực tế của Studio và Viewer.
- Chỉ READY khi optimize và upload storage thành công; thất bại cần đánh FAILED/cleanup/quota.
  Theo dõi timeout pg-boss đang mặc định 900 giây, không để job GPU dài bị retry tạo trùng.
- Studio thêm chọn ảnh/tạo/progress/lỗi/preview/chèn. Viewer đọc MODEL chuẩn, không cần gọi GPU.
- Tests tích hợp cần chứng minh ownership, giới hạn lượt, retry không duplicate, stream hỏng,
  asset chỉ READY sau storage thành công, tải lại trang vẫn theo dõi job.

## 6. Local và rollback

Local gateway cần `GATEWAY_API_KEY`; request có Bearer header. Service nội bộ 5001/5003
chỉ loopback; không expose chúng trực tiếp. Gateway là công cụ thử cho nhóm,
chưa có user ownership/quota/retention của Bookaria. Nếu dùng tunnel/reverse proxy,
key vẫn phải giữ phía server; không gửi vào frontend công khai.

Rollback bằng image tag commit trước đó trong Runpod/template. Kiểm tra lại worker image,
analysis và một pipeline job; không xóa storage hay sửa database Bookaria để rollback worker.

Tài liệu tham chiếu:
- [Runpod operations: payload, stream, cancel](https://docs.runpod.io/serverless/endpoints/operation-reference)
- [Runpod SDK xử lý stream/error](https://github.com/runpod/runpod-python/blob/main/runpod/serverless/modules/rp_job.py)
- [Hunyuan 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
- [Hunyuan 2.0 license](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE)
- [Hunyuan 2.1 license](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/LICENSE)

Cần rà giấy phép cả shape 2.1 và paint 2.0 trước phát hành; giới hạn lãnh thổ có thể áp dụng
cho output. Chưa có benchmark GPU hoặc cam kết chất lượng/chi phí từ bộ test CPU.


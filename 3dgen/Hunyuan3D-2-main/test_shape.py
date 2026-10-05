from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# tải model từ Hugging Face
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")

# dùng ảnh demo có sẵn trong repo
mesh = pipe(image="assets/demo.png")[0]

# lưu mesh ra file glb
mesh.export("output_shape.glb")
print("Saved mesh to output_shape.glb")

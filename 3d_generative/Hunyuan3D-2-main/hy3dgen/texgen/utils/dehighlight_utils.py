# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


class Light_Shadow_Remover():
    def __init__(self, config):
        self.device = getattr(config, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg_image = 1.5
        self.cfg_text = 1.0

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            config.light_remover_ckpt_path,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        pipeline.set_progress_bar_config(disable=True)

        self.pipeline = pipeline
        self.to(self.device)

    def to(self, device):
        self.device = device
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            dtype = torch.float16 if torch.device(device).type == 'cuda' else torch.float32
            self.pipeline.to(device, dtype)
        return self
    
    def recorrect_rgb(self, src_image, target_image, alpha_channel, scale=0.95):
        
        def flat_and_mask(bgr, a):
            mask = torch.where(a > 0.5, True, False)
            bgr_flat = bgr.reshape(-1, bgr.shape[-1])
            mask_flat = mask.reshape(-1)
            bgr_flat_masked = bgr_flat[mask_flat, :]
            return bgr_flat_masked
        
        src_flat = flat_and_mask(src_image, alpha_channel)
        target_flat = flat_and_mask(target_image, alpha_channel)
        corrected_bgr = torch.zeros_like(src_image)

        for i in range(3): 
            src_mean, src_stddev = torch.mean(src_flat[:, i]), torch.std(src_flat[:, i])
            target_mean, target_stddev = torch.mean(target_flat[:, i]), torch.std(target_flat[:, i])
            corrected_bgr[:, :, i] = torch.clamp(
                (src_image[:, :, i] - scale * src_mean) * 
                (target_stddev / src_stddev) + scale * target_mean, 
                0, 1)

        src_mse = torch.mean((src_image - target_image) ** 2)
        modify_mse = torch.mean((corrected_bgr - target_image) ** 2)
        if src_mse < modify_mse:
            corrected_bgr = torch.cat([src_image, alpha_channel], dim=-1)
        else: 
            corrected_bgr = torch.cat([corrected_bgr, alpha_channel], dim=-1)

        return corrected_bgr

    @torch.no_grad()
    def __call__(self, image):
        try:
            device = getattr(self.pipeline, 'device', torch.device(self.device if isinstance(self.device, str) else 'cpu'))
            resized_image = image.resize((512, 512))

            if resized_image.mode == 'RGBA':
                image_array = np.array(resized_image)
                alpha_channel = image_array[:, :, 3]
                erosion_size = 3
                kernel = np.ones((erosion_size, erosion_size), np.uint8)
                alpha_channel = cv2.erode(alpha_channel, kernel, iterations=1)
                image_array[alpha_channel == 0, :3] = 255
                image_array[:, :, 3] = alpha_channel
                proc_image = Image.fromarray(image_array)

                image_tensor = torch.tensor(np.array(proc_image) / 255.0, device=device)
                alpha = image_tensor[:, :, 3:]
                rgb_target = image_tensor[:, :, :3]
            else:
                image_tensor = torch.tensor(np.array(resized_image) / 255.0, device=device)
                alpha = torch.ones_like(image_tensor)[:, :, :1]
                rgb_target = image_tensor[:, :, :3]

            prompt_image = resized_image.convert('RGB')
            gen_device = device if isinstance(device, (torch.device, str)) else 'cpu'
            generator = torch.Generator(device=gen_device).manual_seed(42)

            out_image = self.pipeline(
                prompt="",
                image=prompt_image,
                generator=generator,
                height=512,
                width=512,
                num_inference_steps=50,
                image_guidance_scale=self.cfg_image,
                guidance_scale=self.cfg_text,
            ).images[0]

            image_tensor = torch.tensor(np.array(out_image) / 255.0, device=device)
            rgb_src = image_tensor[:, :, :3]
            corrected = self.recorrect_rgb(rgb_src, rgb_target, alpha)
            final_img = corrected[:, :, :3] * corrected[:, :, 3:] + torch.ones_like(corrected[:, :, :3]) * (1.0 - corrected[:, :, 3:])
            return Image.fromarray((final_img.cpu().numpy() * 255).astype(np.uint8))
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Light_Shadow_Remover failed ({e}), using original image as fallback.")
            return image.convert('RGB') if image.mode != 'RGB' else image

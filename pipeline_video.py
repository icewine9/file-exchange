import inspect
import os
from typing import Union

import PIL
import numpy as np
import torch
#import tqdm
from tqdm import tqdm
from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from transformers import CLIPImageProcessor

from model.attn_processor import SkipAttnProcessor
from model.utils import get_trainable_module, init_adapter
from utils import (compute_vae_encodings, compute_vae_encodings_video2, numpy_to_pil, prepare_image, prepare_video,
                   prepare_mask_image, prepare_mask_video,
                   resize_and_crop, resize_and_padding)

from model.unet_3d import UNet3DConditionModel
from einops import rearrange

class CatVTON_Video_Pipeline:
    def __init__(
        self, 
        base_ckpt, 
        attn_ckpt,
        vae_ckpt,
        motion_ckpt,
        unet_config,
        attn_ckpt_version="mix",
        weight_dtype=torch.float32,
        device='cuda',
        compile=False,
        skip_safety_check=False,
        use_tf32=True,
    ):
        self.device = device
        self.weight_dtype = weight_dtype
        self.skip_safety_check = skip_safety_check

        self.noise_scheduler = DDIMScheduler.from_pretrained(base_ckpt, subfolder="scheduler")
        #self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device, dtype=weight_dtype)
        self.vae = AutoencoderKL.from_pretrained(vae_ckpt).to(device, dtype=weight_dtype)
        # print(f'scaling_factor:{self.vae.scaling_factor}')
        # print(f'latents_std:{self.vae.latents_std}')
        # print(f'latents_mean:{self.vae.latents_mean}')
        # print(f'use_post_quant_conv:{self.vae.use_post_quant_conv}')
        # print(f'use_quant_conv:{self.use_quant_conv}')
        # print(f'force_upcast:{self.force_upcast}')
        # print(f'shift_factor:{self.vae.shift_factor}')
        # print(sdfsdf)

        if not skip_safety_check:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(base_ckpt, subfolder="feature_extractor")
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(base_ckpt, subfolder="safety_checker").to(device, dtype=weight_dtype)

        self.unet = UNet3DConditionModel.from_pretrained_2d(
            base_ckpt,
            motion_ckpt,
            subfolder="unet",
            unet_additional_kwargs=unet_config.unet_additional_kwargs,
        ).to(device, dtype=weight_dtype)

        init_adapter(self.unet, cross_attn_cls=SkipAttnProcessor)  # Skip Cross-Attention
        # init attention process, set all cross-attn process=Skipattnprocessor

        self.attn_modules = get_trainable_module(self.unet, "attention")
        self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)
        # Pytorch 2.0 Compile
        if compile:
            self.unet = torch.compile(self.unet)
            self.vae = torch.compile(self.vae, mode="reduce-overhead")
            
        # Enable TF32 for faster training on Ampere GPUs (A100 and RTX 30 series).
        if use_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
        #self.check_motion_module_params()
        #self.compare_motion_module_with_checkpoint(motion_ckpt)
        #print(f'ok')

    def auto_attn_ckpt_load(self, attn_ckpt, version):
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
            "vivid": "mix-48k-1024",
        }[version]

        # print(f'1check attn-param before load')
        # for name, param in self.attn_modules.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: mean={param.data.mean()}, std={param.data.std()}")

        if os.path.exists(attn_ckpt):
            load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, sub_folder, 'attention'))
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, sub_folder, 'attention'))
            
        # print(f'2check attn-param after load')
        # for name, param in self.attn_modules.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: mean={param.data.mean()}, std={param.data.std()}")

    def run_safety_checker(self, image):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(self.weight_dtype)
            )
        return image, has_nsfw_concept
    
    def decode_latents(self, latents):
        video_length = latents.shape[2]
        #latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        # 这里是第二个进度条，第一个是去噪，第二个是decode出来的frame进行concat
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        #video = video.cpu().float().numpy()
        return video
    
    # def check_inputs(self, image, condition_image, mask, width, height):
    #     if isinstance(image, torch.Tensor) and isinstance(condition_image, torch.Tensor) and isinstance(mask, torch.Tensor):
    #         return image, condition_image, mask
    #     assert image.size == mask.size, "Image and mask must have the same size"
    #     image = resize_and_crop(image, (width, height))
    #     mask = resize_and_crop(mask, (width, height))
    #     condition_image = resize_and_padding(condition_image, (width, height))
    #     return image, condition_image, mask

    def check_inputs(self, video, condition_video, video_mask, width, height):
        if isinstance(video, torch.Tensor) and isinstance(condition_video, torch.Tensor) and isinstance(video_mask, torch.Tensor):
            return video, condition_video, video_mask

        assert video.size(2) == video_mask.size(2), "video and video_mask must have the same frame numbers"
        assert video.size(3) == video_mask.size(3), "video and video_mask must have the same frame width"
        assert video.size(4) == video_mask.size(4), "video and video_mask must have the same frame height"

        video = [resize_and_crop(frame, (width, height)) for frame in video]
        video_mask = [resize_and_crop(mask_frame, (width, height)) for mask_frame in video_mask]
        condition_video = [resize_and_padding(cond_frame, (width, height)) for cond_frame in condition_video]
        print(f'i came here')

        return torch.stack(video), torch.stack(condition_video), torch.stack(video_mask)
    # [batch_size, num_channels, num_frames, height, width]

    def compare_motion_module_with_checkpoint(self, motion_ckpt):
        # 载入 checkpoint 中的参数
        checkpoint = torch.load(motion_ckpt, map_location=self.device)
        
        # 遍历 unet 的 down_blocks, up_blocks 和 mid_block，检查 motion_modules
        for block_type in ['down_blocks', 'up_blocks', 'mid_block']:
            if hasattr(self.unet, block_type):
                blocks = getattr(self.unet, block_type)
                if isinstance(blocks, torch.nn.ModuleList):
                    for block_idx, block in enumerate(blocks):
                        if hasattr(block, 'motion_modules'):
                            print(f"\nComparing motion modules in {block_type} block {block_idx}")
                            for motion_module_idx, motion_module in enumerate(block.motion_modules):
                                for name, param in motion_module.named_parameters():
                                    # 在 checkpoint 中寻找对应的参数
                                    checkpoint_param_name = f"{block_type}.{block_idx}.motion_modules.{motion_module_idx}.{name}"
                                    if checkpoint_param_name in checkpoint:
                                        checkpoint_param = checkpoint[checkpoint_param_name]

                                        # 将参数转换为相同的数据类型
                                        param_data = param.data.float()  # 转换为 float32
                                        checkpoint_data = checkpoint_param.float()  # 转换为 float32

                                        if torch.allclose(param_data, checkpoint_data, atol=1e-5):
                                            print(f"{checkpoint_param_name}: matches the checkpoint.")
                                        else:
                                            print(f"{checkpoint_param_name}: does NOT match the checkpoint.")
                                    else:
                                        print(f"{checkpoint_param_name} not found in checkpoint.")
                        else:
                            print(f"No motion modules in {block_type} block {block_idx}")
                else:
                    # For mid_block case as it's a single module
                    block = blocks
                    if hasattr(block, 'motion_modules'):
                        print(f"\nComparing motion modules in {block_type}")
                        for motion_module_idx, motion_module in enumerate(block.motion_modules):
                            for name, param in motion_module.named_parameters():
                                checkpoint_param_name = f"{block_type}.motion_modules.{motion_module_idx}.{name}"
                                if checkpoint_param_name in checkpoint:
                                    checkpoint_param = checkpoint[checkpoint_param_name]

                                    # 将参数转换为相同的数据类型
                                    param_data = param.data.float()  # 转换为 float32
                                    checkpoint_data = checkpoint_param.float()  # 转换为 float32

                                    if torch.allclose(param_data, checkpoint_data, atol=1e-4):
                                        print(f"{checkpoint_param_name}: matches the checkpoint.")
                                    else:
                                        print(f"{checkpoint_param_name}: does NOT match the checkpoint.")
                                else:
                                    print(f"{checkpoint_param_name} not found in checkpoint.")
                    else:
                        print(f"No motion modules in {block_type}")
            else:
                print(f"{block_type} not found in UNet.")


    def check_motion_module_params(self):
        print("Checking Motion Module parameters initialization...\n")
        
        # 遍历 unet 的 down_blocks, up_blocks 和 mid_block，检查 motion_modules
        for block_type in ['down_blocks', 'up_blocks', 'mid_block']:
            if hasattr(self.unet, block_type):
                blocks = getattr(self.unet, block_type)
                if isinstance(blocks, torch.nn.ModuleList):
                    for block_idx, block in enumerate(blocks):
                        if hasattr(block, 'motion_modules'):
                            print(f"\nChecking motion modules in {block_type} block {block_idx}")
                            for motion_module_idx, motion_module in enumerate(block.motion_modules):
                                for name, param in motion_module.named_parameters():
                                    if param.requires_grad:
                                        print(f"{name}: requires_grad={param.requires_grad}, mean={param.data.mean().item()}, std={param.data.std().item()}")
                                    else:
                                        print(f"{name}: requires_grad={param.requires_grad}, (Frozen Parameter)")
                        else:
                            print(f"No motion modules in {block_type} block {block_idx}")
                else:
                    # For mid_block case as it's a single module
                    block = blocks
                    if hasattr(block, 'motion_modules'):
                        print(f"\nChecking motion modules in {block_type}")
                        for motion_module_idx, motion_module in enumerate(block.motion_modules):
                            for name, param in motion_module.named_parameters():
                                if param.requires_grad:
                                    print(f"{name}: requires_grad={param.requires_grad}, mean={param.data.mean().item()}, std={param.data.std().item()}")
                                else:
                                    print(f"{name}: requires_grad={param.requires_grad}, (Frozen Parameter)")
                    else:
                        print(f"No motion modules in {block_type}")
            else:
                print(f"{block_type} not found in UNet.")



    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self, 
        video, # person videos
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        video_mask, # mask videos
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta=1.0,
        **kwargs
    ):
           
        concat_dim = -2  # FIXME: y axis concat
        video, condition_image, video_mask = self.check_inputs(video, condition_image, video_mask, width, height)
        # [batch_size, num_channels, num_frames, height, width]

        # to check dimension
        video = prepare_video(video).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        video_mask = prepare_mask_video(video_mask).to(self.device, dtype=self.weight_dtype)

        # Mask video
        masked_video = video * (video_mask < 0.5)

        # VAE encoding
        masked_latent = compute_vae_encodings_video2(masked_video, self.vae).permute(0, 2, 1, 3, 4)
        condition_latent = compute_vae_encodings_video2(condition_image, self.vae).permute(0, 2, 1, 3, 4)
        #print(f'mask latent shape:{masked_latent.shape}') # [b,c,f,h,w]
        #print(f'condition_latent:{condition_latent.shape}') # [b,c,1,h,w]

        num_frames = masked_latent.size(2)
        condition_latent_repeated = condition_latent.repeat(1, 1, num_frames, 1, 1)

        # Reshape mask_video
        interpolated_masks = []
        for t in range(num_frames):
            interpolated_frame = torch.nn.functional.interpolate(video_mask[:, t], size=masked_latent.shape[-2:], mode="nearest")
            interpolated_masks.append(interpolated_frame)
        interpolated_masks_latent_video = torch.stack(interpolated_masks, dim=1).permute(0, 2, 1, 3, 4)
        
        del video, video_mask, condition_image

        # Concatenate latents
        masked_latent_concat = torch.cat([masked_latent, condition_latent_repeated], dim=concat_dim)
        mask_latent_concat = torch.cat([interpolated_masks_latent_video, torch.zeros_like(interpolated_masks_latent_video)], dim=concat_dim)
        # Prepare noise
        latents = randn_tensor(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.weight_dtype,
        )

        # Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma

        # Classifier-Free Guidance
        if do_classifier_free_guidance := (guidance_scale > 1.0):
            masked_latent_concat = torch.cat(
                [
                    torch.cat([masked_latent, torch.zeros_like(condition_latent_repeated)], dim=concat_dim),
                    masked_latent_concat,
                ]
            )
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)

        # Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (len(timesteps) - num_inference_steps * self.noise_scheduler.order)
        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                non_inpainting_latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
                non_inpainting_latent_model_input = self.noise_scheduler.scale_model_input(non_inpainting_latent_model_input, t)
                # prepare the input for the inpainting model
                inpainting_latent_model_input = torch.cat([non_inpainting_latent_model_input, mask_latent_concat, masked_latent_concat], dim=1)
                
                # cfg:
                # non_inpainting_latent_model_input(noise) : *2
                # mask_latent: *2
                # masked_latent_concat: no condition + image_condition
        
                
                # predict the noise residual
                noise_pred= self.unet(
                    inpainting_latent_model_input,
                    t.to(self.device),
                    encoder_hidden_states=None, # FIXME
                    return_dict=False,
                )[0]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents # [b,c,f,h,w]
        images = self.decode_latents(latents) 
        
        # delete satefy-check here
        return images

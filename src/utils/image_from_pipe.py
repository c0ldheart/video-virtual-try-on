import os
import sys
from pathlib import Path
from typing import List
from utils.colorvisual import cat2rgb, getshow
from torchvision import utils
import torch
import torchvision.transforms as transforms
from src.utils.data_utils import mask_features
import numpy as np
import cv2
PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from diffusers import StableDiffusionInpaintPipeline
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange, repeat

import torchvision




@torch.no_grad()
def visual_dataset_from_pipe_frames(pipe,
                            test_dataloader: torch.utils.data.DataLoader, output_dir: str,
                            save_name: str, vision_encoder,
                            cloth_input_type: str, cloth_cond_rate: int = 1,
                            seed: int = 1234, num_inference_steps: int = 50,
                            guidance_scale: int = 1,
                            add_warped_cloth_agnostic: bool = False,):

    # Set seed
    generator = torch.Generator("cuda").manual_seed(seed)
    num_samples = 1
    b, f, _, h, w = next(iter(test_dataloader))['target_cloth_frames'].shape
    latent_shape = (b, 4, f, h//8, w//8)
    default_latents = randn_tensor(latent_shape, generator, pipe.device, torch.float16)
    # default_latents = None
    # Generate images
    for idx, batch in enumerate(tqdm(test_dataloader)):
        one_batch_frames_name = [frame_name[0] for frame_name in batch['source_file_name_frames']]
        cur_frame_name = batch['source_image_name'][0]
        frame_num = len(one_batch_frames_name)
        is_last_frame = batch['is_last_frame'][0]
        next_pose_path = batch['next_pose_path'][0]
        # tqdm.write(f"checking {cur_frame_name=}, adj={one_batch_frames_name}, {is_last_frame=}, {next_pose_path=}")
        is_all_new_flag = True
        target_path_list = []
        cur_target_path = os.path.join(f"{output_dir}/{save_name}", cur_frame_name[:-4] + ".jpg")
        for frame_name in one_batch_frames_name:
            target_path = os.path.join(f"{output_dir}/{save_name}", frame_name[:-4] + ".jpg")
            target_path_list.append(target_path)
            # 滑动窗口有重合且当前帧没到最后一帧
            if os.path.exists(target_path):
                if is_last_frame:
                    if not os.path.exists(cur_target_path):
                        # tqdm.write(f"!!!!!!!NOTE: this is last but {cur_target_path=} not exists, so still sample")
                        continue
                    else:
                        pass
                        # tqdm.write(f"-------NOTE: this is last but {cur_target_path=} exists, skip this batch")
                # tqdm.write(f"in this batch {target_path=} exists")
                is_all_new_flag = False
                break
        if not is_all_new_flag:
            continue
        else:
            pass
            # print(f"this batch will simple {target_path_list}")
        if add_warped_cloth_agnostic:
            model_img = batch.get("source_warped_cloth_agnostic_image_frames").to(pipe.device, torch.float16)
            mask_input = batch.get("source_cloth_agnostic_mask_frames").to(pipe.device, torch.float16)
            unet_cloth_input = batch.get("target_cloth_frames").to(pipe.device, torch.float16)
        else:
            mask_input = batch.get("inpaint_mask").to(pipe.device, torch.float16)
            unet_cloth_input = batch.get('warp_feat').to(pipe.device, torch.float16)
            model_img = batch.get("source_image").to(pipe.device, torch.float16)

        # if mask_input is not None:
        #     mask_input = mask_input.type(torch.float32)
        pose_map = batch.get("source_densepose_one_hot_frames").to(pipe.device, torch.float16)
        
        cloth = batch.get("target_cloth_img").to(pipe.device, torch.float16)

        source_frames = batch.get("source_image_frames").to(pipe.device, torch.float16)

        input_image = torchvision.transforms.functional.resize((cloth + 1) / 2, (224, 224),
                                                                antialias=True).clamp(0, 1)
        # processed_images = processor(images=input_image, return_tensors="pt")
        # clip_cloth_features = vision_encoder(
        #     processed_images.pixel_values.to(model_img.device)).last_hidden_state
        # encoder_hidden_states = clip_cloth_features.to(pipe.device, torch.float16)
        #dino

        encoder_hidden_states = vision_encoder(input_image).last_hidden_state.to(pipe.device, torch.float16)
        # Generate images

        generated_images = pipe(
            image=model_img,
            mask_image=mask_input,
            pose_map=pose_map,
            warped_cloth=unet_cloth_input,
            prompt_embeds=encoder_hidden_states,
            height=256,
            width=256,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_samples,
            generator=generator,
            cloth_input_type=cloth_input_type,
            cloth_cond_rate=cloth_cond_rate,
            num_inference_steps=num_inference_steps,
            latents=default_latents
        ).images

        for i in range(len(batch['source_image'])):
            source_image = batch['source_image'].to(pipe.device)
            # target_imgage = batch['target_image'].to(pipe.device)
            source_parsing = batch['source_parsing'].to(pipe.device)
            source_densepose = batch['source_densepose'].to(pipe.device)
            source_pose_forshow = batch['source_pose_forshow'].to(pipe.device)
            source_cloth = batch['source_cloth_img'].to(pipe.device)
            source_cloth_mask = batch['source_cloth_mask'].to(pipe.device)
            source_preserve_mask_forshow = batch['source_preserve_mask_forshow'].to(pipe.device)
            target_cloth = batch['target_cloth_img'].to(pipe.device)
            target_cloth_mask = batch['target_cloth_mask'].to(pipe.device)
            warped_cloth = batch['warp_feat'].to(pipe.device)
            # inpaint_image = batch['inpaint_image'].to(pipe.device)
            inpaint_image = batch['inpaint_image_with_warped'].to(pipe.device)
            source_img_agnostic = batch['source_img_agnostic'].to(pipe.device)
            source_mask_agnostic = batch['source_mask_agnostic'].to(pipe.device)

            generated_images = generated_images.clamp(-1, 1) 


            for j, target_path in enumerate(target_path_list):
                # combine = torch.cat([
                # getshow(source_parsing, 20),
                # # getshow(source_densepose, 25),
                # getshow(pose_map[i:i+1,j , :, :, :],25),
                # cat2rgb(mask_input[i:i+1,j , :, :, :]),
                # inpaint_image[i:i+1, :, :, :],
                # cat2rgb(source_cloth_mask[i:i+1, :, :, :]),
                # cat2rgb(target_cloth_mask[i:i+1, :, :, :]),
                # warped_cloth[i:i+1, :, :, :],
                # model_img[i:i+1,j , :, :, :],
                # generated_images[i:i+1,j ,:, :, :],
                # source_frames[i:i+1,j ,:, :, :],
                # unet_cloth_input[i:i+1,j, :, :, :],
                # ], 0).squeeze()
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                # utils.save_image(combine, target_path, nrow=int(4), normalize=True, range=(-1, 1), )
                frame_img = generated_images[i,j ,:, :, :]
                # frame_img = frame_img * 0.5 + 0.5
                # frame_img = frame_img.squeeze()
                # frame_img = frame_img.permute(1, 2, 0).cpu().numpy()
                # frame_img = (frame_img * 255).astype('uint8')
 
                cv_img = (frame_img.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
                rgb = (cv_img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(target_path, bgr)

                tqdm.write(f"saved at {target_path}")



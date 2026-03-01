import cv2
import torch
import os
import json 
import numpy as np
from PIL import Image

from diffusers.utils import load_image
from diffusers.models import ControlNetModel

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


if __name__ == "__main__":

    base_image_dir = "/data1/joo/pai_bench/data/benchmark/main"
    prompt_json_path = "/data1/joo/pai_bench/data/prompts.json"
    output_root = "/data1/joo/pai_bench/data/generation/instantID"
    os.makedirs(output_root, exist_ok=True)
    
    # Load face encoder
    app = FaceAnalysis(name='antelopev2', root='/data1/joo/pai_bench/model/instantID', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Path to InstantID models
    face_adapter = f'/data1/joo/pai_bench/model/instantID/ip-adapter.bin'
    controlnet_path = f'/data1/joo/pai_bench/model/instantID/ControlNetModel'

    # Load pipeline
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

    base_model_path = 'stabilityai/stable-diffusion-xl-base-1.0'

    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.cuda()
    pipe.load_ip_adapter_instantid(face_adapter)

    with open(prompt_json_path, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    prompt_dict = {item['id']: item['prompt'] for item in prompts_data}

    for i in range(1, 51):
        file_id = f"{i:03d}"
        image_path = os.path.join(base_image_dir, f"{file_id}.jpg")
        
        if not os.path.exists(image_path) or file_id not in prompt_dict:
            continue

        print(f"[{file_id}] 처리 중...")
        text_prompt = prompt_dict[file_id]

        face_image = Image.open(image_path).convert("RGB")
        face_image = resize_img(face_image)

        face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        
        if len(face_info) == 0:
            print(f"  - {file_id}: 얼굴을 검출하지 못해 건너뜁니다.")
            continue
            
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        face_emb = face_info['embedding']
        face_kps = draw_kps(face_image, face_info['kps'])

        num_samples = 4
        save_folder = os.path.join(output_root, file_id)
        os.makedirs(save_folder, exist_ok=True)
        
        for idx in range(num_samples):
            output_image = pipe(
                prompt=text_prompt,
                negative_prompt="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, cartoon, anime",
                image_embeds=face_emb,
                image=face_kps,
                controlnet_conditioning_scale=0.8,
                ip_adapter_scale=0.8,
                num_inference_steps=30,
                guidance_scale=5,
                num_images_per_prompt=1, # 1개씩 생성 설정
            ).images[0]

            # 샘플 저장
            save_path = os.path.join(save_folder, f"{idx}.jpg")
            output_image.save(save_path)
            print(f"  - 샘플 {idx} 저장 완료: {save_path}")

        torch.cuda.empty_cache()
        
    print("전체 작업 완료!")
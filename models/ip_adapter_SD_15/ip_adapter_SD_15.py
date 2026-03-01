import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
import os
import json
from ip_adapter import IPAdapterPlus

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "/data1/joo/pai_bench/model/ia_SD_15/models/image_encoder"
ip_ckpt = "/data1/joo/pai_bench/model/ia_SD_15/models/ip-adapter-plus-face_sd15.bin"
device = "cuda:0"
image_dir = "/data1/joo/pai_bench/data/benchmark"
prompts_dir = "/data1/joo/pai_bench/data/prompts/prompts.json"
output_dir = "/data1/joo/pai_bench/data/generation/ablation/ip_adapter_15_SD"

os.makedirs(output_dir, exist_ok=True)

with open(prompts_dir, 'r', encoding='utf-8') as f:
    prompts_text = json.load(f)

prompt_dict = {item['id']: item['description'] for item in prompts_text}

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

# load SD pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
).to(device)


# load ip-adapter
ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

# for i in range(1, 51):
#     file_id = f"{i:03d}"  
#     image_path = os.path.join(image_dir, f"{file_id}.jpg")
    
#     if not os.path.exists(image_path):
#         print(f"이미지 없음: {image_path}, 건너뜁니다.")
#         continue
    
#     if file_id not in prompt_dict:
#         print(f"ID {file_id}에 해당하는 프롬프트 없음, 건너뜁니다.")
#         continue

#     text_prompt = prompt_dict[file_id]
#     print(f"[{file_id}] 처리 중... Prompt: {text_prompt}")

#     input_image = Image.open(image_path).convert("RGB")
#     input_image = input_image.resize((256, 256))

#     images = ip_model.generate(
#         pil_image=input_image, 
#         num_samples=4, 
#         num_inference_steps=50, 
#         seed=420,
#         prompt=text_prompt
#     )
#     save_folder = os.path.join(output_dir, file_id)
#     os.makedirs(save_folder, exist_ok=True)
    
#     for idx, img in enumerate(images):
#         save_file_name = f"{idx}.jpg"
#         save_path = os.path.join(save_folder, save_file_name)
        
#         img.save(save_path)
#         print(f"  - 샘플 {idx} 저장 완료: {save_path}")

# print("모든 작업이 완료되었습니다.")

for i in range(1, 51):
    folder_name = f"{i:03d}"
    folder_path = os.path.join(image_dir, folder_name)

    if not os.path.isdir(folder_path):
        print(f"폴더 없음: {folder_path}, 건너뜁니다.")
        continue

    if folder_name not in prompt_dict:
        print(f"ID {folder_name}에 해당하는 프롬프트 없음, 건너뜁니다.")
        continue

    # 폴더 내 이미지 파일 정렬 후 처음 3장 선택
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ])

    if len(image_files) < 3:
        print(f"[{folder_name}] 이미지가 3장 미만 ({len(image_files)}장), 건너뜁니다.")
        continue

    # 1, 2, 3번째 이미지 로드
    input_images = []
    for img_file in image_files[:3]:
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path).convert("RGB").resize((256, 256))
        input_images.append(img)

    text_prompt = prompt_dict[folder_name]
    print(f"[{folder_name}] 처리 중... Prompt: {text_prompt}, 입력 이미지: {image_files[:3]}")

    # pil_image에 리스트로 전달하여 다중 참조 이미지 사용
    images = ip_model.generate(
        pil_image=input_images,
        num_samples=4,
        num_inference_steps=50,
        seed=420,
        prompt=text_prompt
    )

    save_folder = os.path.join(output_dir, folder_name)
    os.makedirs(save_folder, exist_ok=True)

    for idx, img in enumerate(images):
        save_file_name = f"{idx}.jpg"
        save_path = os.path.join(save_folder, save_file_name)
        img.save(save_path)
        print(f"  - 샘플 {idx} 저장 완료: {save_path}")

print("모든 작업이 완료되었습니다.")
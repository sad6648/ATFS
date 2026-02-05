import argparse
import copy
import hashlib
import itertools
import logging
import os
import sys
import gc
from pathlib import Path
from colorama import Fore, Style, init, Back
import random, time
import matplotlib.pyplot as plt

'''some system level settings'''
init(autoreset=True)
sys.path.insert(0, sys.path[0]+"/../")

from model import Generator, Discriminator
import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, PretrainedConfig
from torch import autograd
from typing import Optional, Tuple
import pynvml
from utils import *

logger = get_logger(__name__)

# =========================================================================
# 1. 辅助函数：通用自适应梯度投影 (Multi-Gradient Adaptive PCGrad)
# =========================================================================
def apply_adaptive_pcgrad(grads, weights=None):
    """
    通用版自适应 PCGrad，支持任意数量的任务梯度。
    解决多个模型之间的梯度冲突。
    """
    if len(grads) == 0:
        return None
    if len(grads) == 1:
        return grads[0] * (weights[0] if weights else 1.0)

    # 1. 展平所有梯度以便计算点积
    flatten_grads = [g.view(-1) for g in grads]
    
    # 2. 初始化修正后的梯度
    pc_grads = [g.clone() for g in flatten_grads]
    
    # 3. 随机打乱任务顺序进行投影 (PCGrad 核心)
    num_tasks = len(grads)
    indices = list(range(num_tasks))
    random.shuffle(indices) 
    
    for i in indices:
        for j in indices:
            if i == j:
                continue
            
            g_i = pc_grads[i]
            g_j = flatten_grads[j] # 对比的是原始梯度
            
            inner_product = torch.dot(g_i, g_j)
            
            # 如果冲突 (夹角 > 90度)
            if inner_product < 0:
                norm_j_sq = torch.dot(g_j, g_j)
                if norm_j_sq > 1e-8:
                    # 投影: g_i = g_i - proj
                    proj = (inner_product / norm_j_sq) * g_j
                    pc_grads[i] = g_i - proj
    
    # 4. 恢复形状并加权融合
    final_grad = torch.zeros_like(grads[0])
    
    for i, g_flat in enumerate(pc_grads):
        g_restored = g_flat.view_as(grads[0])
        w = weights[i] if weights else 1.0
        final_grad += w * g_restored
        
    return final_grad

# =========================================================================
# 2. 辅助函数：加载目标图像
# =========================================================================
def load_target_image_tensor(path, size=512, device='cuda', dtype=torch.float32):
    """加载单张目标图像（Target Image）并转为Tensor"""
    if not os.path.exists(path):
        print(f"{Fore.RED}[警告] 找不到目标图片: {path}，将生成随机噪声纹理作为目标。{Style.RESET_ALL}")
        return torch.randn(1, 3, size, size, device=device, dtype=dtype)
    
    image = Image.open(path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor.to(device, dtype=dtype)

# =========================================================================
# 3. 核心攻击逻辑：三模型联合特征攻击
# =========================================================================
def pgd_attack_combined(
    args,
    accelerator,
    models_dm,
    tokenizer_dm,
    noise_scheduler_dm,
    vae_dm,
    generator_gan,
    vae_vq,  # [新增] VQ-VAE 模型
    data_tensor,
    original_images,
    weight_dtype
):
    """
    联合攻击: SD (VAE Latent) + StarGAN (Encoder Feature) + VQ-VAE (Latent)
    """
    unet_dm, text_encoder_dm = models_dm
    device = accelerator.device
    
    perturbed_images = data_tensor.detach().clone().to(device, dtype=weight_dtype)
    original_images = original_images.to(device, dtype=weight_dtype)

    # -----------------------------------------------------------
    # Step A: 提取 "目标图像" 的特征 (Target Features)
    # -----------------------------------------------------------
    target_img_tensor = load_target_image_tensor(
        args.target_image_path, 
        size=args.resolution, 
        device=device, 
        dtype=weight_dtype
    )

    with torch.no_grad():
        # [Target 1] SD VAE Latent
        target_feature_dm = vae_dm.encode(target_img_tensor).latent_dist.mode() * vae_dm.config.scaling_factor
        target_feature_dm = target_feature_dm.detach()

        # [Target 2] StarGAN Encoder Feature (第8层)
        c_trg_gan_dummy = torch.zeros(1, args.stargan_c_dim, device=device, dtype=weight_dtype)
        _, target_features_gan_list = generator_gan(target_img_tensor, c_trg_gan_dummy)
        target_feature_gan = target_features_gan_list[8].detach()

        # [Target 3] VQ-VAE Latent (新增)
        # 根据你提供的代码，VQ-VAE latent = mode() * scaling
        target_feature_vq = vae_vq.encode(target_img_tensor).latent_dist.mode() * args.vqvae_scaling_factor
        target_feature_vq = target_feature_vq.detach()

    # -----------------------------------------------------------
    # Step B: PGD 攻击循环
    # -----------------------------------------------------------
    batch_size = perturbed_images.shape[0]
    
    # 扩展目标特征到 Batch 维度
    batch_target_dm = target_feature_dm.repeat(batch_size, 1, 1, 1)
    batch_target_gan = target_feature_gan.repeat(batch_size, 1, 1, 1)
    batch_target_vq = target_feature_vq.repeat(batch_size, 1, 1, 1)
    batch_c_gan = c_trg_gan_dummy.repeat(batch_size, 1)

    tbar = tqdm(range(args.max_adv_train_steps), desc="三重联合攻击", leave=False)
    
    for step in tbar:
        perturbed_images.requires_grad = True

        # --- Loss 1: SD VAE 攻击 ---
        latents_dm = vae_dm.encode(perturbed_images).latent_dist.mode() * vae_dm.config.scaling_factor
        loss_dm = F.mse_loss(latents_dm, batch_target_dm)
        grad_dm = autograd.grad(loss_dm, perturbed_images, retain_graph=True)[0]

        # --- Loss 2: StarGAN 攻击 ---
        _, current_features_gan_list = generator_gan(perturbed_images, batch_c_gan)
        current_feature_gan = current_features_gan_list[8] 
        loss_gan = F.mse_loss(current_feature_gan, batch_target_gan)
        grad_gan = autograd.grad(loss_gan, perturbed_images, retain_graph=True)[0]

        # --- Loss 3: VQ-VAE 攻击 (新增) ---
        latents_vq = vae_vq.encode(perturbed_images).latent_dist.mode() * args.vqvae_scaling_factor
        loss_vq = F.mse_loss(latents_vq, batch_target_vq)
        grad_vq = autograd.grad(loss_vq, perturbed_images)[0]

        # --- 梯度融合 (3路 PCGrad) ---
        # 权重设置建议: GAN 特征层较深通常给大权重，VAE 类模型给中等权重
        # 这里的权重你可以根据效果微调：[w_sd, w_gan, w_vqvae]
        weights = [1.0, 5.0, 5.0] 
        combined_grad = apply_adaptive_pcgrad([grad_dm, grad_gan, grad_vq], weights=weights)

        # --- PGD 更新 ---
        alpha = args.pgd_alpha
        eps = args.pgd_eps
        
        with torch.no_grad():
            adv_images = perturbed_images - alpha * combined_grad.sign()
            eta = torch.clamp(adv_images - original_images, min=-eps, max=eps)
            perturbed_images = torch.clamp(original_images + eta, min=-1, max=1).detach()
        
        tbar.set_postfix(
            L_dm=f"{loss_dm.item():.3f}", 
            L_gan=f"{loss_gan.item():.3f}",
            L_vq=f"{loss_vq.item():.3f}"
        )

    return perturbed_images

# =========================================================================
# 4. 数据加载
# =========================================================================
def load_data(data_dir, size=512) -> torch.Tensor:
    image_transforms = transforms.Compose([
        transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    images = []
    all_files = os.listdir(data_dir)
    image_filenames = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    try:
        image_filenames.sort(key=lambda f: int(os.path.splitext(f)[0]))
    except ValueError:
        image_filenames.sort()

    print(f"检测到 {len(image_filenames)} 张图片，正在加载...")
    
    for filename in image_filenames:
        file_path = os.path.join(data_dir, filename)
        images.append(Image.open(file_path).convert("RGB"))

    images = [image_transforms(img) for img in images]
    images = torch.stack(images)
    print(f"== 数据张量形状: {images.shape} ==")
    return images

# =========================================================================
# 5. 参数解析
# =========================================================================
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="MIST + StarGAN + VQVAE Combined Attack")

    # 基础配置
    parser.add_argument("--cuda", action='store_true', default=True)
    parser.add_argument("--output_dir", type=str, default="./output_adv")
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--target_image_path", type=str, default="data/MIST.png")
    
    # 模型路径
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="./stable-diffusion", help="SD模型路径")
    parser.add_argument("--stargan_model_path", type=str, default="./models/stargan_G.ckpt", help="StarGAN权重")
    parser.add_argument("--vqvae_model_path", type=str, default="./", help="[新增] VQ-VAE模型路径")
    
    # 攻击参数
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_adv_train_steps", type=int, default=50)
    parser.add_argument("--pgd_alpha", type=float, default=0.005)
    parser.add_argument("--pgd_eps", type=float, default=0.05)
    parser.add_argument("--vqvae_scaling_factor", type=float, default=1.0, help="[新增] VQ-VAE的缩放因子")

    # StarGAN 配置
    parser.add_argument('--stargan_c_dim', type=int, default=5)
    parser.add_argument('--stargan_g_conv_dim', type=int, default=64)
    parser.add_argument('--stargan_g_repeat_num', type=int, default=6)

    # 兼容参数
    parser.add_argument("--revision", type=str, default="")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    
    # 允许 tf32, low_vram_mode 等参数（如果之前脚本有用到）
    parser.add_argument("--low_vram_mode", action='store_true')
    parser.add_argument("--instance_prompt", type=str, default="a person")

    args = parser.parse_args(input_args)
    return args

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision,
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    else:
        return transformers.CLIPTextModel

# =========================================================================
# 6. Main
# =========================================================================
def main(args):
    print(f"\n{Fore.GREEN}🚀 启动联合对抗攻击 (SD + StarGAN + VQVAE)...{Style.RESET_ALL}")
    
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=Path(args.output_dir, args.logging_dir),
        cpu=not args.cuda
    )
    
    weight_dtype = torch.float32
    if args.cuda and args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.cuda and args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    print(f"使用精度: {weight_dtype}")

    # 1. 加载 Stable Diffusion
    print("\n⏳ 加载 Stable Diffusion 模型...")
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
    vae_dm = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False)
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # 2. 加载 StarGAN
    print("⏳ 加载 StarGAN 模型...")
    generator_gan = Generator(args.stargan_g_conv_dim, args.stargan_c_dim, args.stargan_g_repeat_num)
    if os.path.exists(args.stargan_model_path):
        generator_gan.load_state_dict(torch.load(args.stargan_model_path, map_location="cpu"))
    else:
        print(f"{Fore.RED}[Error] 找不到 StarGAN 模型文件: {args.stargan_model_path}{Style.RESET_ALL}")
        return

    # 3. 加载 VQ-VAE (新增)
    print("⏳ 加载 VQ-VAE 模型...")
    try:
        vae_vq = AutoencoderKL.from_pretrained(args.vqvae_model_path, local_files_only=True)
    except Exception as e:
        print(f"{Fore.RED}[Error] 加载 VQ-VAE 失败: {e}{Style.RESET_ALL}")
        print("请检查 --vqvae_model_path 路径是否包含 model_index.json 或 config.json")
        return

    # 4. 移动模型到设备
    models_to_move = [vae_dm, text_encoder, unet, generator_gan, vae_vq]
    for model in models_to_move:
        model.to(device=accelerator.device, dtype=weight_dtype).eval()
        model.requires_grad_(False) 

    # 5. 加载数据
    print(f"\n📂 读取数据: {args.instance_data_dir}")
    all_original_data = load_data(args.instance_data_dir, size=args.resolution)
    dataset = torch.utils.data.TensorDataset(all_original_data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 6. 批处理攻击
    print(f"\n⚔️ 开始生成对抗样本...")
    all_perturbed_data_list = []

    for i, (batch_imgs,) in enumerate(data_loader):
        print(f"\n[Batch {i+1}/{len(data_loader)}]")
        
        adv_batch = pgd_attack_combined(
            args=args,
            accelerator=accelerator,
            models_dm=(unet, text_encoder),
            tokenizer_dm=tokenizer,
            noise_scheduler_dm=noise_scheduler,
            vae_dm=vae_dm,
            generator_gan=generator_gan,
            vae_vq=vae_vq, # 传入第三个模型
            data_tensor=batch_imgs,
            original_images=batch_imgs,
            weight_dtype=weight_dtype
        )
        
        all_perturbed_data_list.append(adv_batch.cpu())

    # 7. 保存结果
    print(f"\n💾 正在保存结果到: {args.output_dir}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    final_data = torch.cat(all_perturbed_data_list, dim=0)
    
    for i, img_tensor in enumerate(final_data):
        img_np = (img_tensor.permute(1, 2, 0) * 127.5 + 127.5).clamp(0, 255).to(torch.uint8).numpy()
        Image.fromarray(img_np).save(os.path.join(args.output_dir, f"{i}.png"))

    print(f"\n✅ 全部完成！")

if __name__ == "__main__":
    args = parse_args()
    main(args)
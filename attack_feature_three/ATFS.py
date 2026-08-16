import argparse
import logging
import os
import sys
from pathlib import Path
from colorama import Fore, Style, init

init(autoreset=True)
sys.path.insert(0, sys.path[0] + "/../")

from model import Generator, Discriminator
try:
    from stgan_model import STGANGenerator
    STGAN_AVAILABLE = True
except ImportError:
    STGAN_AVAILABLE = False

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm.auto import tqdm
from torch import autograd
from transformers import AutoTokenizer, PretrainedConfig

logger = get_logger(__name__)


# =========================================================================
# 1. Helper: gradient L2 normalization (ATFS: g_hat_k = g_k / (||g_k|| + xi))
# =========================================================================
def normalize_grad(g, xi=1e-8):
    g = g.float()
    return g / (g.norm() + xi)


# =========================================================================
# 2. Helper: load target image
# =========================================================================
def load_target_image_tensor(path, size=512, device='cuda', dtype=torch.float32):
    if not os.path.exists(path):
        print(f"{Fore.RED}[Warning] Target image not found: {path}; using random noise.{Style.RESET_ALL}")
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
# 3. ATFS attack (Algorithm 1)
# =========================================================================
def pgd_attack_combined(
    args,
    accelerator,
    models_dm,
    tokenizer_dm,
    noise_scheduler_dm,
    vae_dm,
    generator_gan,
    vae_vq,
    data_tensor,
    original_images,
    weight_dtype
):
    """
    ATFS attack following Algorithm 1.
    Supports selective model activation via args.active_models.
    Available models: sd, gan, stgan, vqvae.
    """
    unet_dm, text_encoder_dm = models_dm
    device = accelerator.device
    active = set(args.active_models.split(","))

    perturbed_images = data_tensor.detach().clone().to(device, dtype=weight_dtype)
    original_images = original_images.to(device, dtype=weight_dtype)

    # ---- Phase 1: Precompute Target Features ----
    target_img_tensor = load_target_image_tensor(
        args.target_image_path,
        size=args.resolution,
        device=device,
        dtype=weight_dtype
    )

    with torch.no_grad():
        target_feature_dm = None
        target_feature_gan = None
        target_feature_vq = None
        c_trg_gan_dummy = None

        if "sd" in active:
            target_feature_dm = vae_dm.encode(target_img_tensor).latent_dist.mode() * vae_dm.config.scaling_factor
            target_feature_dm = target_feature_dm.detach()

        if generator_gan is not None and ("gan" in active or "stgan" in active):
            c_trg_gan_dummy = torch.zeros(1, args.stargan_c_dim, device=device, dtype=weight_dtype)
            _, target_features_gan_list = generator_gan(target_img_tensor, c_trg_gan_dummy)
            target_feature_gan = target_features_gan_list[8].detach()

        if vae_vq is not None and "vqvae" in active:
            if hasattr(vae_vq, 'latent_dist'):
                target_feature_vq = vae_vq.encode(target_img_tensor).latent_dist.mode() * args.vqvae_scaling_factor
            elif hasattr(vae_vq, 'encode'):
                enc_out = vae_vq.encode(target_img_tensor)
                if hasattr(enc_out, 'latents'):
                    target_feature_vq = enc_out.latents * args.vqvae_scaling_factor
                else:
                    target_feature_vq = enc_out * args.vqvae_scaling_factor
            target_feature_vq = target_feature_vq.detach()

    # ---- Phase 2: Synergistic Optimization Loop ----
    batch_size = perturbed_images.shape[0]

    batch_target_dm = target_feature_dm.repeat(batch_size, 1, 1, 1) if target_feature_dm is not None else None
    batch_target_gan = target_feature_gan.repeat(batch_size, 1, 1, 1) if target_feature_gan is not None else None
    batch_target_vq = target_feature_vq.repeat(batch_size, 1, 1, 1) if target_feature_vq is not None else None
    batch_c_gan = c_trg_gan_dummy.repeat(batch_size, 1) if c_trg_gan_dummy is not None else None

    weights = [float(w) for w in args.ensemble_weights.split(",")]
    w_idx = 0
    omega_dm = omega_gan = omega_vq = 0.0
    if "sd" in active:
        omega_dm = weights[w_idx]; w_idx += 1
    if generator_gan is not None and ("gan" in active or "stgan" in active):
        omega_gan = weights[w_idx]; w_idx += 1
    if vae_vq is not None and "vqvae" in active:
        omega_vq = weights[w_idx]; w_idx += 1
    xi = 1e-8

    tbar = tqdm(range(args.max_adv_train_steps), desc="ATFS optimization", leave=False)

    for step in tbar:
        perturbed_images.requires_grad = True
        grads = []

        # --- Loss 1: SD VAE ---
        if "sd" in active:
            latents_dm = vae_dm.encode(perturbed_images).latent_dist.mode() * vae_dm.config.scaling_factor
            loss_dm = (latents_dm.float() - batch_target_dm.float()).pow(2).sum()
            grad_dm = normalize_grad(autograd.grad(loss_dm, perturbed_images)[0], xi)
            grads.append(omega_dm * grad_dm)

        # --- Loss 2: GAN (StarGAN or STGAN) ---
        if generator_gan is not None and ("gan" in active or "stgan" in active):
            _, current_features_gan_list = generator_gan(perturbed_images, batch_c_gan)
            current_feature_gan = current_features_gan_list[8]
            loss_gan = (current_feature_gan.float() - batch_target_gan.float()).pow(2).sum()
            grad_gan = normalize_grad(autograd.grad(loss_gan, perturbed_images)[0], xi)
            grads.append(omega_gan * grad_gan)

        # --- Loss 3: VQ-VAE ---
        if vae_vq is not None and "vqvae" in active:
            if hasattr(vae_vq, 'latent_dist'):
                latents_vq = vae_vq.encode(perturbed_images).latent_dist.mode() * args.vqvae_scaling_factor
            elif hasattr(vae_vq, 'encode'):
                enc_out = vae_vq.encode(perturbed_images)
                if hasattr(enc_out, 'latents'):
                    latents_vq = enc_out.latents * args.vqvae_scaling_factor
                else:
                    latents_vq = enc_out * args.vqvae_scaling_factor
            loss_vq = (latents_vq.float() - batch_target_vq.float()).pow(2).sum()
            grad_vq = normalize_grad(autograd.grad(loss_vq, perturbed_images)[0], xi)
            grads.append(omega_vq * grad_vq)

        # --- Synergistic gradient: g_syn = sum_k omega_k * g_hat_k ---
        g_syn = sum(grads).to(weight_dtype)

        # --- PGD update ---
        alpha = args.pgd_alpha
        eps = args.pgd_eps
        with torch.no_grad():
            delta = perturbed_images - original_images
            delta = torch.clamp(delta - alpha * g_syn.sign(), min=-eps, max=eps)
            perturbed_images = (original_images + delta).detach()

        postfix = {}
        if "sd" in active:
            postfix["L_dm"] = f"{loss_dm.item():.4f}"
        if generator_gan is not None and ("gan" in active or "stgan" in active):
            postfix["L_gan"] = f"{loss_gan.item():.4f}"
        if vae_vq is not None and "vqvae" in active:
            postfix["L_vq"] = f"{loss_vq.item():.4f}"
        tbar.set_postfix(**postfix)

    x_adv = torch.clamp(perturbed_images, min=-1, max=1)
    return x_adv


# =========================================================================
# 4. Data loading
# =========================================================================
def load_data(data_dir, size=512) -> torch.Tensor:
    image_transforms = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
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

    print(f"Detected {len(image_filenames)} images, loading...")

    for filename in image_filenames:
        file_path = os.path.join(data_dir, filename)
        images.append(Image.open(file_path).convert("RGB"))

    images = [image_transforms(img) for img in images]
    images = torch.stack(images)
    print(f"== Data tensor shape: {images.shape} ==")
    return images


# =========================================================================
# 5. Argument parsing
# =========================================================================
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="ATFS: Architecture-Agnostic Targeted Feature Synergy")

    parser.add_argument("--cuda", action='store_true', default=True)
    parser.add_argument("--output_dir", type=str, default="./output_adv")
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--target_image_path", type=str, default="data/ATFS.png")

    # Model paths
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="./stable-diffusion")
    parser.add_argument("--stargan_model_path", type=str, default="./attack_feature_three/stargan_celeba_256/models/200000-G.ckpt")
    parser.add_argument("--stgan_model_path", type=str, default="./attack_feature_three/stgan_G.pth")
    parser.add_argument("--vqvae_model_path", type=str, default="./attack_feature_three/VQ-VAE/")

    # Attack parameters (aligned with paper: eps=6/255, T=100, alpha=eps/10)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_adv_train_steps", type=int, default=100)
    parser.add_argument("--pgd_alpha", type=float, default=0.00235, help="Step size alpha = eps/10")
    parser.add_argument("--pgd_eps", type=float, default=0.02353, help="Perturbation budget eps = 6/255")
    parser.add_argument("--vqvae_scaling_factor", type=float, default=1.0)
    parser.add_argument("--ensemble_weights", type=str, default="1.0,5.0",
                        help="Weights omega_k for active models, comma-separated. Order follows active_models.")

    # Model selection
    parser.add_argument("--active_models", type=str, default="sd,gan",
                        help="Comma-separated: sd, gan, stgan, vqvae. Default: sd,gan (two-model, Table III).")
    parser.add_argument("--gan_type", type=str, default="stargan", choices=["stargan", "stgan"],
                        help="GAN architecture: stargan or stgan")

    # StarGAN config
    parser.add_argument('--stargan_c_dim', type=int, default=5)
    parser.add_argument('--stargan_g_conv_dim', type=int, default=64)
    parser.add_argument('--stargan_g_repeat_num', type=int, default=6)

    # Compatibility
    parser.add_argument("--revision", type=str, default="")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
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
    active = set(args.active_models.split(","))
    gan_label = "stgan" if args.gan_type == "stgan" else "gan"
    models_str = ",".join(sorted(active))
    print(f"\n{Fore.GREEN}[Start] ATFS attack | active models: {models_str}{Style.RESET_ALL}")

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

    print(f"Precision: {weight_dtype}")

    # 1. Load Stable Diffusion (always needed)
    print("\n[Loading] Stable Diffusion...")
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
    vae_dm = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False)
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # 2. Load GAN (StarGAN or STGAN)
    generator_gan = None
    if gan_label in active or "gan" in active:
        if args.gan_type == "stgan":
            print("[Loading] STGAN ...")
            if not STGAN_AVAILABLE:
                print(f"{Fore.RED}[Error] stgan_model.py not found. Falling back to StarGAN.{Style.RESET_ALL}")
                args.gan_type = "stargan"
            else:
                generator_gan = STGANGenerator(args.stargan_g_conv_dim, args.stargan_c_dim, args.stargan_g_repeat_num)
                if os.path.exists(args.stgan_model_path):
                    generator_gan.load_state_dict(torch.load(args.stgan_model_path, map_location="cpu"))
                else:
                    print(f"{Fore.YELLOW}[Warning] STGAN weights not found: {args.stgan_model_path}. Using random init.{Style.RESET_ALL}")

        if args.gan_type == "stargan" or generator_gan is None:
            print("[Loading] StarGAN (v1, Choi 2018)...")
            generator_gan = Generator(args.stargan_g_conv_dim, args.stargan_c_dim, args.stargan_g_repeat_num)
            if os.path.exists(args.stargan_model_path):
                generator_gan.load_state_dict(torch.load(args.stargan_model_path, map_location="cpu"))
            else:
                print(f"{Fore.RED}[Error] StarGAN model not found: {args.stargan_model_path}{Style.RESET_ALL}")
                return

    # 3. Load VQ-VAE
    vae_vq = None
    if "vqvae" in active:
        print("[Loading] VQ-VAE...")
        try:
            vae_vq = AutoencoderKL.from_pretrained(args.vqvae_model_path, local_files_only=True)
        except Exception:
            try:
                from huggingface_hub import PyTorchModelHubMixin
                vae_vq = PyTorchModelHubMixin.from_pretrained(args.vqvae_model_path)
                print(f"{Fore.YELLOW}[Info] VQ-VAE loaded via PyTorchModelHubMixin (non-diffusers format).{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}[Error] Failed to load VQ-VAE from {args.vqvae_model_path}: {e}{Style.RESET_ALL}")
                vae_vq = None

    # 4. Move models to device
    models_to_move = [vae_dm, text_encoder, unet]
    if generator_gan is not None:
        models_to_move.append(generator_gan)
    if vae_vq is not None:
        models_to_move.append(vae_vq)
    for model in models_to_move:
        model.to(device=accelerator.device, dtype=weight_dtype).eval()
        model.requires_grad_(False)

    # 5. Load data
    print(f"\n[Data] Loading data: {args.instance_data_dir}")
    all_original_data = load_data(args.instance_data_dir, size=args.resolution)
    dataset = torch.utils.data.TensorDataset(all_original_data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 6. Batch attack
    print(f"\n[Attack] Generating adversarial samples...")
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
            vae_vq=vae_vq,
            data_tensor=batch_imgs,
            original_images=batch_imgs,
            weight_dtype=weight_dtype
        )

        all_perturbed_data_list.append(adv_batch.cpu())

    # 7. Save results
    print(f"\n[Save] Saving to: {args.output_dir}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    final_data = torch.cat(all_perturbed_data_list, dim=0)

    for i, img_tensor in enumerate(final_data):
        img_np = (img_tensor.permute(1, 2, 0) * 127.5 + 127.5).clamp(0, 255).to(torch.uint8).numpy()
        Image.fromarray(img_np).save(os.path.join(args.output_dir, f"{i}.png"))

    print(f"\n[Done] All adversarial samples generated.")


if __name__ == "__main__":
    args = parse_args()
    main(args)

# Breaking the Defense Silos: A Universal Feature Anchoring Framework (ATFS)

Official PyTorch Implementation for the paper:"**Breaking the Defense Silos: A Universal Feature Anchoring Framework Against Heterogeneous Generative Threats**"
The ubiquitous deployment of Generative AI has created "Defense Silos"—where protection methods optimized for one architecture (e.g., Diffusion Models) fail against others (e.g., GANs). This paper proposes ATFS (Architecture-Agnostic Targeted Feature Synergy), a framework that breaks these silos by shifting the defense paradigm from pixel-level conflict to feature-level consensus.
**Key Features:**
1) Universal Protection: Simultaneously defends against Diffusion Models (SD v1.5), GANs (StarGAN v2), and VQ-VAEs.
2) High Efficiency: Converges in ~100 steps (approx. 10.0s) using linear gradient aggregation, bypassing expensive gradient surgery like PCGrad.
3) Robustness: Resilient to JPEG compression, scaling, and Gaussian noise.

# Architecture & Method

ATFS solves the gradient conflict problem by aligning gradients in the feature space.

## The ATFS Workflow (Algorithm 1)

1) Semantic Anchoring: Extract features of a target anchor image $x_{tgt}$ using the encoders of all target models ($\Phi_{k}$).
2) Synergistic Optimization: Optimize the perturbation $\delta$ to minimize the distance between the protected image's features and the anchor's features across all models.
3) Adaptive Graient Equalization: Normalize gradients from heterogeneous models to ensure balanced contribution.
The implementation of this logic can be found in ATFS.py inside the pgd_attack_combined function.

# Quick Start

## 1.Requirements

The environment requires PyTorch and Diffusers.

```bah
conda create -n atfs python=3.9
conda activate atfs
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate matplotlib scikit-image colorama pynvml
```

## 2.Model Preparatio

Please download the pre-trained weights and place them in the `models/` directory:

**Stable Diffusion v1.5**: https://huggingface.co/docs/diffusers/using-diffusers/img2img

**StarGAN v2**: Pre-trained on CelebA-HQ (`stargan_G.ckpt`).

**VQ-VAE**: Standard VQ-VAE checkpoints.

## 3.Usage

To generate adversarial examples that are effective against Stable Diffusion, StarGAN, and VQ-VAE simultaneously, run `ATFS.py`.

```bash
accelerate launch ./mist.py \
    --cuda \
    --low_vram_mode \
    --instance_data_dir ./data/img \
    --output_dir ./output \
    --instance_prompt "a person" \
    --mixed_precision bf16 \
    --max_adv_train_steps 100 \
    --stargan_model_path './stargan_celeba_256/models/200000-G.ckpt' \
    --stargan_c_dim 5 \
    --target_image_path "./data/AGC-LDW.png" \
    --pgd_eps 0.0314 \
    --batch_size 1 \
    --vqvae_model_path "./VQ-VAE/" \
    --vqvae_scaling_factor 1.0
```

Key Arguments:

- `--instance_data_dir`: Path to the clean images you want to protect.
- `--target_image_path`: **(Critical)** The **Semantic Anchor ($x_{tgt}$)**. As per the paper, using a structured noise image or a style-distinct image works best.
- `--max_adv_train_steps`: Default is 50. The paper suggests **40 steps** for the optimal efficiency trade-off.
- `--pgd_eps`: Perturbation budget ($\epsilon$). Default is 0.05 (approx 12/255). For stricter invisibility, use smaller values.


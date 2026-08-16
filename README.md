# ATFS: Architecture-Agnostic Targeted Feature Synergy

Official PyTorch implementation for the paper: **"ATFS: A Feature Synergy Framework for Proactive Data Protection Against Heterogeneous Generative Threats"**

ATFS breaks "Defense Silos" by shifting adversarial protection from pixel-level conflict to feature-level consensus. It simultaneously protects personal data against Diffusion Models (Stable Diffusion v1.5), GANs (StarGAN v1 / STGAN), and VQ-VAEs through a unified target-aligned feature synergy mechanism.

## Key Features

1. Universal Protection: Defends against Diffusion, GAN, and VQ-VAE architectures simultaneously or in any combination.
2. High Efficiency: Converges in ~100 PGD steps using L2-normalized weighted gradient aggregation.
3. Robustness: Resilient to JPEG compression, spatial scaling, and Gaussian noise.

## Method

ATFS solves the gradient conflict problem by aligning gradients in the feature space. The implementation follows Algorithm 1:

1. Semantic Anchoring: Extract features of a target image x_tgt using the encoders of all target models.
2. Synergistic Optimization: Optimize the perturbation delta to minimize the squared L2 distance between the protected image's features and the anchor's features across all models.
3. Gradient Equalization: L2-normalize each model's gradient before weighted summation.

## Requirements

```bash
conda create -n atfs python=3.9
conda activate atfs
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate matplotlib scikit-image colorama pynvml
```

## Model Preparation

Download the pre-trained weights and place them in the appropriate directories:

- Stable Diffusion v1.5: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- StarGAN v1 (Choi et al., 2018): Pre-trained on CelebA-HQ (`200000-G.ckpt`). https://huggingface.co/spaces/vipul2412/starGAN
- VQ-VAE (Switti tokenizer): https://huggingface.co/yresearch/VQVAE-Switti

## Usage

### Two-model: SD + StarGAN (Table III)

```bash
accelerate launch ./attack_feature_three/ATFS.py \
    --cuda \
    --instance_data_dir ./data/img \
    --output_dir ./output \
    --mixed_precision bf16 \
    --max_adv_train_steps 100 \
    --pgd_eps 0.02353 \
    --pgd_alpha 0.00235 \
    --active_models sd,gan \
    --gan_type stargan \
    --stargan_model_path './attack_feature_three/stargan_celeba_256/models/200000-G.ckpt' \
    --stargan_c_dim 5 \
    --ensemble_weights "1.0,5.0" \
    --target_image_path "./data/ATFS.png" \
    --batch_size 1
```

### Two-model: SD + STGAN (Table IV)

```bash
accelerate launch ./attack_feature_three/ATFS.py \
    --cuda \
    --instance_data_dir ./data/img \
    --output_dir ./output \
    --mixed_precision bf16 \
    --max_adv_train_steps 100 \
    --pgd_eps 0.02353 \
    --pgd_alpha 0.00235 \
    --active_models sd,stgan \
    --gan_type stgan \
    --stgan_model_path './attack_feature_three/stgan/stgan_G.pth' \
    --stargan_c_dim 5 \
    --ensemble_weights "1.0,5.0" \
    --target_image_path "./data/ATFS.png" \
    --batch_size 1
```

### Two-model: SD + VQ-VAE (Table V)

```bash
accelerate launch ./attack_feature_three/ATFS.py \
    --cuda \
    --instance_data_dir ./data/img \
    --output_dir ./output \
    --mixed_precision bf16 \
    --max_adv_train_steps 100 \
    --pgd_eps 0.02353 \
    --pgd_alpha 0.00235 \
    --active_models sd,vqvae \
    --vqvae_model_path "./attack_feature_three/VQ-VAE/" \
    --vqvae_scaling_factor 1.0 \
    --ensemble_weights "1.0,5.0" \
    --target_image_path "./data/ATFS.png" \
    --batch_size 1
```

## Key Arguments

- `--active_models`: Comma-separated model list. Options: `sd`, `gan` (StarGAN), `stgan` (STGAN/SPADE), `vqvae`. Default: `sd,gan` (Table III).
- `--gan_type`: GAN architecture. `stargan` (Choi 2018) or `stgan` (SPADE). Default: `stargan`.
- `--pgd_eps`: Perturbation budget (epsilon). Default: 0.02353 (6/255), matching the paper.
- `--pgd_alpha`: Step size (alpha). Default: 0.00235 (eps/10), matching the paper.
- `--max_adv_train_steps`: Number of PGD iterations (T). Default: 100, matching the paper.
- `--ensemble_weights`: Weights omega_k for gradient fusion, one per active model. Default: "1.0,5.0".
- `--target_image_path`: Semantic anchor image x_tgt. Using a structured noise image works best.

### Utility Test: Face Detection and Recognition (Reviewer Response)

Evaluate whether ATFS-protected images retain benign utility for standard
discriminative face tasks. Uses MTCNN for detection and FaceNet (VGGFace2)
for recognition, which operate on different feature spaces than the generative
models ATFS targets.

```bash
python ./attack_feature_three/utility_test.py \
    --original_dir ./data/img \
    --protected_dir ./output \
    --device cpu \
    --output_file utility_results.json
```

## File Structure

- `attack_feature_three/ATFS.py`: Main attack script implementing Algorithm 1.
- `attack_feature_three/model.py`: StarGAN v1 Generator and Discriminator.
- `attack_feature_three/stgan_model.py`: STGAN Generator (SPADE) and Discriminator.
- `attack_feature_three/utils.py`: Utility functions.
- `attack_feature_three/utility_test.py`: Benign utility test (MTCNN detection + FaceNet recognition).

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{atfs2026,
  title={ATFS: A Feature Synergy Framework for Proactive Data Protection Against Heterogeneous Generative Threats},
  year={2026}
}
```

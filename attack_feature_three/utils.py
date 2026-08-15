import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from color_space import *

def load_model_weights(model, path):
    pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'preprocessing' not in k}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict, strict=False)

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

# def create_labels(c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
#     """Generate target domain labels for debugging and testing."""
#     # Get hair color indices.
#     if dataset == 'CelebA':
#         hair_color_indices = []
#         for i, attr_name in enumerate(selected_attrs):
#             if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
#                 hair_color_indices.append(i)

#     c_trg_list = []
#     for i in range(c_dim):
#         if dataset == 'CelebA':
#             c_trg = c_org.clone()
#             if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
#                 c_trg[:, i] = 1
#                 for j in hair_color_indices:
#                     if j != i:
#                         c_trg[:, j] = 0
#             else:
#                 c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
#         elif dataset == 'RaFD':
#             c_trg = label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

#         c_trg_list.append(c_trg.cuda())
#     return c_trg_list

def create_labels(c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
    """
    Improved create_labels: only modify one dimension, keep others unchanged.
    """
    hair_color_indices = [i for i, attr in enumerate(selected_attrs) if attr in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']]
    
    c_trg_list = []
    for i in range(c_dim):
        c_trg = c_org.clone()

        # For hair color: keep only current color, set others to 0
        if i in hair_color_indices:
            for j in hair_color_indices:
                c_trg[:, j] = 1 if j == i else 0
        else:
            # Reverse current dimension only, keep other attributes unchanged
            c_trg[:, i] = (c_trg[:, i] == 0)

        c_trg_list.append(c_trg.cuda())

    return c_trg_list

def random_transform(img):
    T_list = [
        T.RandomHorizontalFlip(p=0.5),
        #T.RandomErasing(p=1, scale=(0.03, 0.10)),
        T.RandomRotation(degrees=(-15, 15)),
        T.RandomVerticalFlip(p=0.5),
        T.RandomCrop((192,192)),
    ]

    T_compose = T.Compose([
        T.RandomChoice(T_list),
        T.Resize((256, 256)),
    ])

    return T_compose(img)

# def compare(img1,img2):
#     """input tensor, translate to np.array"""
#     img1_np = img1.squeeze(0).cpu().numpy()
#     img2_np = img2.squeeze(0).cpu().numpy()
#     img1_np = np.transpose(img1_np, (1, 2, 0))
#     img2_np = np.transpose(img2_np, (1, 2, 0))

#     ssim = structural_similarity(img1_np, img2_np, win_size=5, channel_axis=-1, data_range=img1_np.max() - img1_np.min())
#     # -----------------<<<<<<<<<<<<<<<<  source code below
#     # ssim = structural_similarity(img1_np,img2_np,multichannel=True)
#     psnr = peak_signal_noise_ratio(img1_np,img2_np)

#     return ssim, psnr

def compare(img1, img2):
    """Input tensors [1,C,H,W] -> compute SSIM & PSNR"""
    img1_np = img1.squeeze(0).cpu().numpy()  # shape [C, H, W]
    img2_np = img2.squeeze(0).cpu().numpy()

    img1_np = np.transpose(img1_np, (1, 2, 0))  # shape [H, W, C]
    img2_np = np.transpose(img2_np, (1, 2, 0))

    # SSIM with window size check and channel_axis
    ssim_val = structural_similarity(
        img1_np,
        img2_np,
        win_size=5,  # Make sure H, W >= 5
        channel_axis=-1,
        data_range=img1_np.max() - img1_np.min()
    )

    # PSNR with MSE=0 check
    mse = np.mean((img1_np - img2_np) ** 2)
    if mse == 0:
        psnr_val = float('inf')
    else:
        psnr_val = peak_signal_noise_ratio(img1_np, img2_np, data_range=img1_np.max() - img1_np.min())

    return ssim_val, psnr_val



def lab_attack(X_nat, c_trg, model, epsilon=0.05, iter=100):
    criterion = nn.MSELoss().cuda()
    pert_a = torch.zeros(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]).cuda().requires_grad_()
    optimizer = torch.optim.Adam([pert_a], lr=5e-4, betas=(0.9, 0.999))

    X = denorm(X_nat.clone())

    for i in range(iter):
        X_lab = rgb2lab(X).cuda()

        # Add perturbation, clip ab channels to [-1.2, 1.2]
        pert = torch.clamp(pert_a, -epsilon, epsilon)
        X_lab[:, 1:, :, :] = torch.clamp(X_lab[:, 1:, :, :] + pert, -1.2, 1.2)

        # Convert to RGB + normalize
        X_new = lab2rgb(X_lab)
        X_new = torch.nan_to_num(X_new, nan=0.0, posinf=1.0, neginf=0.0)
        X_new = T.Normalize([0.5]*3, [0.5]*3)(X_new)

        gen_noattack, _ = model(X_nat, c_trg[i % len(c_trg)])
        gen_adv, _ = model(X_new, c_trg[i % len(c_trg)])

        loss = -criterion(gen_adv, gen_noattack)

        if i % 10 == 0:
            print(f"[lab_attack] Iter {i}, Loss: {loss.item():.6f}, ||pert|| = {torch.norm(pert).item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return X_new, X_new - X


def pixel_attack(X_nat, c_trg, model, epsilon=0.05, iter=100):
    """
    Add perturbation in RGB pixel space, optimize to maximize difference
    between G(x_adv, c_trg) and G(x_nat, c_trg).
    Input image range: [-1, 1]
    """
    criterion = nn.MSELoss().cuda()
    
    # Step 1: Convert input from [-1, 1] to [0, 1]
    X_nat_denorm = denorm(X_nat.clone()).detach()
    
    # Step 2: Initialize perturbation tensor
    pert = torch.zeros_like(X_nat_denorm, device=X_nat.device).requires_grad_()

    optimizer = torch.optim.Adam([pert], lr=1e-2)

    for i in range(iter):
        # Step 3: Add perturbation + clip to [0, 1]
        X_adv = X_nat_denorm + torch.clamp(pert, -epsilon, epsilon)
        X_adv = X_adv.clamp(0, 1)

        # Step 4: Re-normalize to [-1, 1] for G
        X_input = (X_adv - 0.5) * 2.0

        # Step 5: Original generation vs adversarial generation
        gen_noattack, _ = model(X_nat, c_trg[i % len(c_trg)])
        gen_attack, _ = model(X_input, c_trg[i % len(c_trg)])

        loss = -criterion(gen_attack, gen_noattack)

        if i % 10 == 0:
            print(f"[pixel_attack] Iter {i}, Loss: {loss.item():.6f}, ||pert|| = {torch.norm(pert).item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Step 6: Return perturbed adversarial image
    X_adv_final = (X_nat_denorm + torch.clamp(pert, -epsilon, epsilon)).clamp(0, 1)
    return (X_adv_final - 0.5) * 2.0, X_adv_final - X_nat_denorm



prompt_dataset = [
    "Portrait of an astronaut in space, detailed starry background, reflective helmet,",
    "Painting of a floating island with giant clock gears, populated with mythical creatures,",
    "Landscape of a Japanese garden in autumn, with a bridge over a koi pond,",
    "Painting representing the sound of jazz music, using vibrant colors and erratic shapes,",
    "Painting of a modern smartphone with classic art pieces appearing on the screen,",
    "Battle scene with futuristic robots and a golden palace in the background,",
    "Scene of a bustling city market with different perspectives of people and stalls,",
    "Scene of a ship sailing in a stormy sea, with dramatic lighting and powerful waves,",
    "Portraint of a female botanist surrounded by exotic plants in a greenhouse,",
    "Painting of an ancient castle at night, with a full moon, gargoyles, and shadows,",
]

style_dataset = [
    "Art Nouveau",
    "Romantic",
    "Cubist",
    "Baroque",
    "Pop Art",
    "Abstract",
    "Impressionist",
    "Surrealist",
    "Renaissance",
    "Pointillism",
]



class attack_mixin:
    def __call__(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        unet: torch.nn.Module,
        target_tensor: torch.Tensor,
        noise_scheduler
    ):
        raise NotImplementedError
    
class AdvDM(attack_mixin):
    """
    This attack aims to maximize the training loss of diffusion model
    """
    def __call__(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        unet: torch.nn.Module,
        text_encoder: torch.nn.Module,
        input_ids,
        target_tensor: torch.Tensor,
        noise_scheduler
    ):
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(input_ids)[0]

        # Predict the noise residual
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        unet.zero_grad()
        text_encoder.zero_grad()
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # target-shift loss
        if target_tensor is not None:
            xtm1_pred = torch.cat(
                [
                    noise_scheduler.step(
                        model_pred[idx : idx + 1],
                        timesteps[idx : idx + 1],
                        noisy_latents[idx : idx + 1],
                    ).prev_sample
                    for idx in range(len(model_pred))
                ]
            )
            xtm1_target = noise_scheduler.add_noise(target_tensor, noise, timesteps - 1)
            loss = loss - F.mse_loss(xtm1_pred, xtm1_target)

        return loss
    
class LatentAttack(attack_mixin):
    """
    This attack aims to minimize the l2 distance between latent and target_tensor
    """
    def __call__(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor=None,
        encoder_hidden_states: torch.Tensor=None,
        unet: torch.nn.Module=None,
        target_tensor: torch.Tensor=None,
        noise_scheduler=None
    ):
        if target_tensor == None:
            raise ValueError("Need a target tensor for pre-attack")
        loss = - F.mse_loss(latents, target_tensor, reduction="mean")
        return loss

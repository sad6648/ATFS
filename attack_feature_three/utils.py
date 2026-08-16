import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

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
        T.RandomRotation(degrees=(-15, 15)),
        T.RandomVerticalFlip(p=0.5),
        T.RandomCrop((192,192)),
    ]

    T_compose = T.Compose([
        T.RandomChoice(T_list),
        T.Resize((256, 256)),
    ])

    return T_compose(img)

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

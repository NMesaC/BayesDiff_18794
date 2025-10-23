import os
import torch
from torchvision import transforms
import torchvision.utils as tvu
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

from tqdm import tqdm

to_pil = transforms.ToPILImage()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load model
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:", m)
    if len(u) > 0 and verbose:
        print("unexpected keys:", u)
    model.eval()
    return model.to(device)

config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
model = load_model_from_config(config, "./checkpoints/v1-5-pruned-emaonly.ckpt")

# Directory containing latent z_exp and z_var
exp_dir = "./ddim_exp/skipUQ/cfg3.0_a cat eating pizza_train1000_step50_S10"
os.makedirs(f'{exp_dir}/x_dev', exist_ok=True)

# Sample N times from latent distribution and compute per-pixel std
def get_dev_x_from_z(dev, exp, N):
    z_list = [exp + torch.rand_like(exp) * dev for _ in range(N)]
    Z = torch.stack(z_list, dim=0)
    X = model.decode_first_stage(Z.to(device))
    var_x = torch.var(X, dim=0)
    dev_x = var_x**0.5
    return dev_x

# Get list of latent IDs based on saved files
latent_files = sorted(os.listdir(f'{exp_dir}/z_var'))
latent_ids = [int(f.split(".")[0]) for f in latent_files]

# Process each latent sample
N = 15
for id in tqdm(latent_ids):
    z_var_i = torch.load(f'{exp_dir}/z_var/{id}.pth')
    z_exp_i = torch.load(f'{exp_dir}/z_exp/{id}.pth')
    z_dev_i = torch.clamp(z_var_i, min=0)**0.5

    dev_x = get_dev_x_from_z(z_dev_i, z_exp_i, N)
    tvu.save_image(dev_x * 100, f'{exp_dir}/x_dev/{id}.jpg')

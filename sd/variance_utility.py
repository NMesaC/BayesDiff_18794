import argparse, os
import torch
import math
from itertools import islice
from omegaconf import OmegaConf
from tqdm import tqdm
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from custom_ld import CustomLD
from dataset import laion_dataset
import torchvision.utils as tvu
from ldm.util import instantiate_from_config
from utils import NoiseScheduleVP, get_model_input_time
from ddimUQ_utils import compute_alpha, singlestep_ddim_sample, var_iteration, exp_iteration, \
    sample_from_gaussion

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def conditioned_exp_iteration(model, exp_xt, seq, timestep, pre_wuq, mc_eps_exp_t=None, acc_eps_t = None):
    if pre_wuq == True:
        return exp_iteration(model, exp_xt, seq, timestep, mc_eps_exp_t)
    else:
        return exp_iteration(model, exp_xt, seq, timestep, acc_eps_t)
    
def conditioned_var_iteration(model, var_xt, cov_xt_epst, var_epst, seq, timestep, pre_wuq):

    if pre_wuq == True:
        return var_iteration(model, var_xt, cov_xt_epst, var_epst, seq, timestep)
    else:
        n = var_xt.size(0)
        t = (torch.ones(n)*seq[timestep]).to(var_xt.device)
        next_t = (torch.ones(n)*seq[(timestep-1)]).to(var_xt.device)
        at = compute_alpha(model.betas, t.long())
        at_next = compute_alpha(model.betas, next_t.long())
        var_xt_next = (at_next/at) * var_xt

        return var_xt_next

def get_scaled_var_eps(scale, var_eps_c, var_eps_uc):
    return pow(1-scale, 2)* var_eps_uc + pow(scale, 2)* var_eps_c
def get_scaled_exp_eps(scale, exp_eps_c, exp_eps_uc):
    return (1-scale)* exp_eps_uc + scale* exp_eps_c

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())
 
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a cheetah drinking an espresso",
        help="the prompt to render"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="time_uniform",
        help="skip according to ('uniform' or 'quadratic' for DDIM/DDPM; 'logSNR' or 'time_uniform' or 'time_quadratic' for DPM-Solver)",
    )
    parser.add_argument(
        "--from_file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--laion_art_path", 
        type=str, 
        required=True,
        help='Path to the LAION parquet file for art data'
    )
    parser.add_argument(
        "--local_image_path",
        type=str, 
        required=True,
        help='Path to the images for LLLA Approximation'
    )
    parser.add_argument("--output_dir", type=str, default="./outputs",
                    help="Root directory to store experiment results")
    parser.add_argument("--exp_name", type=str, default="ddim_skipUQ",
                    help="Name of this experiment subfolder")
    parser.add_argument("--mc_size", type=int, default=10)
    parser.add_argument("--sample_batch_size", type=int, default=8)
    parser.add_argument("--train_la_batch_size", type=int, default=4)
    parser.add_argument("--train_la_data_size", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default= 50)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--total_n_samples', type=int, default=80)
    parser.add_argument('--cut', type=int, default=40)
    opt = parser.parse_args()
    print(opt)
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    # print(model.model.diffusion_model.out[2])
    # Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    train_dataset= laion_dataset(model, opt)
    train_dataloader= torch.utils.data.DataLoader(train_dataset, batch_size=opt.train_la_batch_size, shuffle=False)
    custom_ld = CustomLD(model, train_dataloader)

    fixed_xT = torch.randn([opt.sample_batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
##########   get t sequence (note that t is different from timestep)  ########## 

    skip = model.num_timesteps // opt.timesteps
    seq = range(0, model.num_timesteps, skip)

#########   get skip UQ rules  ##########  
# if uq_array[i] == False, then we use origin_dpmsolver_update from t_seq[i] to t_seq[i-1]
    uq_array = [False] * (opt.timesteps)
    cut = opt.cut
    for i in range(opt.timesteps-1, cut, -2):
        uq_array[i] = True
    
#########   get prompt  ##########  
    if opt.from_file:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
    else:
        c = model.get_learned_conditioning(opt.prompt)
        c = torch.concat(opt.sample_batch_size * [c], dim=0)
        uc = model.get_learned_conditioning(opt.sample_batch_size * [""])
        # exp_dir = f'/home///ddim_skipUQ/var_use'
        # os.makedirs(exp_dir, exist_ok=True)
        # Determine experiment directory
        if getattr(opt, "output_dir", None) is None:
            opt.output_dir = "./outputs"

        # Default experiment name if missing
        if not hasattr(opt, "exp_name") or not opt.exp_name:
            opt.exp_name = "ddim_skipUQ"

        exp_dir = os.path.join(opt.output_dir, opt.exp_name)
        os.makedirs(exp_dir, exist_ok=True)

#########   start sample  ########## 
    # --- helper to make conditioning match the batch size of tensor z ---
    def _match_cond_batch(cond, x_target):
        """
        Expand or repeat the conditioning tensor `cond`
        so it matches the batch dimension of x_target.
        Handles dict/list wrappers and preserves correct shape [B, L, D] if present.
        """
        b_target = x_target.shape[0]

        # unwrap nested
        if isinstance(cond, (list, tuple)):
            cond = cond[0]
        elif isinstance(cond, dict):
            if 'c_crossattn' in cond:
                cond = cond['c_crossattn'][0]
            elif 'crossattn' in cond:
                cond = cond['crossattn'][0]
            else:
                for v in cond.values():
                    if torch.is_tensor(v):
                        cond = v
                        break

        if not torch.is_tensor(cond):
            raise TypeError(f"Expected cond Tensor, got {type(cond)}")

        # --- preserve [B, L, D] if present ---
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)              # [1, D]
        elif cond.dim() == 2:
            pass                                  # [B, D]
        elif cond.dim() == 3:
            pass                                  # [B, L, D]
        else:
            # Higher dims unexpected — flatten spatially only if not SD text
            cond = cond.view(cond.shape[0], -1)

        b_cond = cond.shape[0]

        # --- match batch size ---
        if b_cond == b_target:
            return cond
        elif b_cond == 1:
            return cond.repeat(b_target, *([1] * (cond.dim() - 1)))
        elif b_target % b_cond == 0:
            repeats = b_target // b_cond
            return cond.repeat(repeats, *([1] * (cond.dim() - 1)))
        else:
            repeats = (b_target + b_cond - 1) // b_cond
            cond_expanded = cond.repeat(repeats, *([1] * (cond.dim() - 1)))
            return cond_expanded[:b_target]
        
    def _sanitize_filename(s, max_len=100):
        # keep alphanumeric, spaces, hyphens and underscores; collapse spaces to underscore
        import re
        if not isinstance(s, str):
            s = str(s)
        s = s.strip()
        s = s[:max_len]
        s = re.sub(r"[\\/]+", "_", s)           # remove slashes
        s = re.sub(r"[^A-Za-z0-9 _\-\.]", "", s)  # allow limited set
        s = re.sub(r"\s+", "_", s)              # spaces -> underscores
        return s or "prompt"


    # --- main loop (only the relevant parts shown, integrate into your code) ---
    data = laion_dataset(model, opt)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for raw_prompts in tqdm(data):
                    # Normalize prompt(s) to list[str]
                    if isinstance(raw_prompts, str):
                        prompt_list = [raw_prompts]
                    elif isinstance(raw_prompts, (list, tuple)):
                        prompt_list = [str(p) for p in raw_prompts]
                    elif isinstance(raw_prompts, dict):
                        if "text" in raw_prompts:
                            prompt_list = [str(raw_prompts["text"])]
                        elif "caption" in raw_prompts:
                            prompt_list = [str(raw_prompts["caption"])]
                        else:
                            found = None
                            for v in raw_prompts.values():
                                if isinstance(v, str):
                                    found = v
                                    break
                            if found is None:
                                raise ValueError(f"Cannot extract text from dataset sample: keys={list(raw_prompts.keys())}")
                            prompt_list = [str(found)]
                    else:
                        prompt_list = [str(raw_prompts)]

                    # Build exp_dir safely (use your existing exp_dir logic)
                    # Define root directory for experiment outputs
                    output_root = os.path.join(os.getcwd(), "outputs")  # or change to your desired path
                    os.makedirs(output_root, exist_ok=True)

                    exp_name = "variance_utility"  # or use opt.exp_name if your parser defines one
                    safe_prompt = opt.prompt.replace(" ", "_").replace("/", "_")

                    exp_dir = os.path.join(
                        output_root,
                        exp_name,
                        f'prior_precision{1}_train_data_size{opt.train_la_data_size}_cut{opt.cut}_{safe_prompt}'
                    )
                    os.makedirs(exp_dir, exist_ok=True)

                    for j in range(opt.sample_batch_size):
                        img_id = 1000000

                        # Get base conditioning for the prompt(s)
                        base_c = model.get_learned_conditioning(prompt_list)
                        base_uc = model.get_learned_conditioning([""] * len(prompt_list))

                        # Prepare starting xT for this j and move to device
                        xT = fixed_xT[j, :, :, :].unsqueeze(0).to(device)  # (1, C, H, W)
                        timestep = opt.timesteps - 1
                        T = seq[timestep]
                        mc_sample_size = opt.mc_size

                        # Ensure conditioning matches xT batch size
                        c = _match_cond_batch(base_c, xT)   # shape (B=1, D)
                        uc = _match_cond_batch(base_uc, xT)

                        # First call into custom_ld with correctly batched conds
                        if uq_array[timestep] == True:
                            xt_next = xT
                            exp_xt_next, var_xt_next = xT, torch.zeros_like(xT).to(device)
                            # now pass c/uc that match xT batch
                            eps_mu_t_next_c, eps_var_t_next_c = custom_ld(xT, (torch.ones(1) * T).to(xT.device), c=c)
                            eps_mu_t_next_uc, eps_var_t_next_uc = custom_ld(xT, (torch.ones(1) * T).to(xT.device), c=uc)
                            eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=eps_mu_t_next_c, exp_eps_uc=eps_mu_t_next_uc)
                            eps_var_t_next = get_scaled_var_eps(opt.scale, eps_var_t_next_c, eps_var_t_next_uc)
                            cov_xt_next_epst_next = torch.zeros_like(xT).to(device)
                            list_eps_mu_t_next_i = torch.unsqueeze(eps_mu_t_next, dim=0)
                        else:
                            xt_next = xT
                            exp_xt_next, var_xt_next = xT, torch.zeros_like(xT).to(device)
                            eps_mu_t_next_c = custom_ld.accurate_forward(xT, (torch.ones(1) * T).to(xT.device), c=c)
                            eps_mu_t_next_uc = custom_ld.accurate_forward(xT, (torch.ones(1) * T).to(xT.device), c=uc)
                            eps_mu_t_next = get_scaled_exp_eps(opt.scale, eps_mu_t_next_c, eps_mu_t_next_uc)

                        # Main reverse diffusion loop
                        for timestep in range(opt.timesteps - 1, cut, -1):
                            # (your existing logic unmodified)
                            # ... when you compute eps_mu_t_next_c/uc for xt_next, ALWAYS match cond batch to xt_next:
                            # Example replacement inside both branches (wherever you see custom_ld(xt_next, ..., c=c)):
                            #
                            #   c_local = _match_cond_batch(c, xt_next)
                            #   uc_local = _match_cond_batch(uc, xt_next)
                            #   eps_mu_t_next_c, eps_var_t_next_c = custom_ld(xt_next, (torch.ones(1) * seq[timestep-1]).to(xt.device), c=c_local)
                            #
                            # I'll apply this pattern below in the two relevant places.

                            if uq_array[timestep] == True:
                                xt = xt_next
                                exp_xt, var_xt = exp_xt_next, var_xt_next
                                eps_mu_t, eps_var_t, cov_xt_epst = eps_mu_t_next, eps_var_t_next, cov_xt_next_epst_next
                                mc_eps_exp_t = torch.mean(list_eps_mu_t_next_i, dim=0)
                            else:
                                xt = xt_next
                                exp_xt, var_xt = exp_xt_next, var_xt_next
                                eps_mu_t = eps_mu_t_next

                            if uq_array[timestep] == True:
                                eps_t = sample_from_gaussion(eps_mu_t, eps_var_t)
                                xt_next = singlestep_ddim_sample(model, xt, seq, timestep, eps_t)
                                exp_xt_next = conditioned_exp_iteration(model, exp_xt, seq, timestep, pre_wuq=uq_array[timestep], mc_eps_exp_t=mc_eps_exp_t)
                                var_xt_next = conditioned_var_iteration(model, var_xt, cov_xt_epst, var_epst=eps_var_t, seq=seq, timestep=timestep, pre_wuq=uq_array[timestep])
                                if uq_array[timestep - 1] == True:
                                    list_xt_next_i, list_eps_mu_t_next_i = [], []
                                    for _ in range(mc_sample_size):
                                        var_xt_next = torch.clamp(var_xt_next, min=0)
                                        xt_next_i = sample_from_gaussion(exp_xt_next, var_xt_next)
                                        list_xt_next_i.append(xt_next_i)
                                        # Ensure conditioning matches xt_next_i batch
                                        c_local = _match_cond_batch(c, xt_next_i)
                                        uc_local = _match_cond_batch(uc, xt_next_i)
                                        eps_mu_t_next_i_c, _ = custom_ld(xt_next_i, (torch.ones(1) * seq[timestep - 1]).to(xt.device), c=c_local)
                                        eps_mu_t_next_i_uc, _ = custom_ld(xt_next_i, (torch.ones(1) * seq[timestep - 1]).to(xt.device), c=uc_local)
                                        eps_mu_t_next_i = get_scaled_exp_eps(opt.scale, eps_mu_t_next_i_c, eps_mu_t_next_i_uc)
                                        list_eps_mu_t_next_i.append(eps_mu_t_next_i)

                                    # Match conds for the aggregated xt_next
                                    c_local = _match_cond_batch(c, xt_next)
                                    uc_local = _match_cond_batch(uc, xt_next)
                                    eps_mu_t_next_c, eps_var_t_next_c = custom_ld(xt_next, (torch.ones(1) * seq[timestep - 1]).to(xt.device), c=c_local)
                                    eps_mu_t_next_uc, eps_var_t_next_uc = custom_ld(xt_next, (torch.ones(1) * seq[timestep - 1]).to(xt.device), c=uc_local)
                                    eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=eps_mu_t_next_c, exp_eps_uc=eps_mu_t_next_uc)
                                    eps_var_t_next = get_scaled_var_eps(opt.scale, eps_var_t_next_c, eps_var_t_next_uc)
                                    list_xt_next_i = torch.stack(list_xt_next_i, dim=0).to(device)
                                    list_eps_mu_t_next_i = torch.stack(list_eps_mu_t_next_i, dim=0).to(device)
                                    cov_xt_next_epst_next = torch.mean(list_xt_next_i * list_eps_mu_t_next_i, dim=0) - exp_xt_next * torch.mean(list_eps_mu_t_next_i, dim=0)
                                else:
                                    # ensure batch match
                                    c_local = _match_cond_batch(c, xt_next)
                                    uc_local = _match_cond_batch(uc, xt_next)
                                    eps_mu_t_next_c = custom_ld.accurate_forward(xt_next, (torch.ones(1) * seq[timestep - 1]).to(xt.device), c=c_local)
                                    eps_mu_t_next_uc = custom_ld.accurate_forward(xt_next, (torch.ones(1) * seq[timestep - 1]).to(xt.device), c=uc_local)
                                    eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=eps_mu_t_next_c, exp_eps_uc=eps_mu_t_next_uc)
                            else:
                                xt_next = singlestep_ddim_sample(model, xt, seq, timestep, eps_mu_t)
                                exp_xt_next = conditioned_exp_iteration(model, exp_xt, seq, timestep, pre_wuq=uq_array[timestep], acc_eps_t=eps_mu_t)
                                var_xt_next = conditioned_var_iteration(model, var_xt, cov_xt_epst=None, var_epst=None, seq=seq, timestep=timestep, pre_wuq=uq_array[timestep])
                                if uq_array[timestep - 1] == True:
                                    list_xt_next_i, list_eps_mu_t_next_i = [], []
                                    for _ in range(mc_sample_size):
                                        var_xt_next = torch.clamp(var_xt_next, min=0)
                                        xt_next_i = sample_from_gaussion(exp_xt_next, var_xt_next)
                                        list_xt_next_i.append(xt_next_i)
                                        c_local = _match_cond_batch(c, xt_next_i)
                                        uc_local = _match_cond_batch(uc, xt_next_i)
                                        eps_mu_t_next_i_c, _ = custom_ld(xt_next_i, (torch.ones(1) * seq[timestep - 1]).to(xt.device), c=c_local)
                                        eps_mu_t_next_i_uc, _ = custom_ld(xt_next_i, (torch.ones(1) * seq[timestep - 1]).to(xt.device), c=uc_local)
                                        eps_mu_t_next_i = get_scaled_exp_eps(opt.scale, eps_mu_t_next_i_c, eps_mu_t_next_i_uc)
                                        list_eps_mu_t_next_i.append(eps_mu_t_next_i)

                                    c_local = _match_cond_batch(c, xt_next)
                                    uc_local = _match_cond_batch(uc, xt_next)
                                    eps_mu_t_next_c, eps_var_t_next_c = custom_ld(xt_next, (torch.ones(1) * seq[timestep - 1]).to(xt.device), c=c_local)
                                    eps_mu_t_next_uc, eps_var_t_next_uc = custom_ld(xt_next, (torch.ones(1) * seq[timestep - 1]).to(xt.device), c=uc_local)
                                    eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=eps_mu_t_next_c, exp_eps_uc=eps_mu_t_next_uc)
                                    eps_var_t_next = get_scaled_var_eps(opt.scale, eps_var_t_next_c, eps_var_t_next_uc)
                                    list_xt_next_i = torch.stack(list_xt_next_i, dim=0).to(device)
                                    list_eps_mu_t_next_i = torch.stack(list_eps_mu_t_next_i, dim=0).to(device)
                                    cov_xt_next_epst_next = torch.mean(list_xt_next_i * list_eps_mu_t_next_i, dim=0) - exp_xt_next * torch.mean(list_eps_mu_t_next_i, dim=0)
                                else:
                                    c_local = _match_cond_batch(c, xt_next)
                                    uc_local = _match_cond_batch(uc, xt_next)
                                    eps_mu_t_next_c = custom_ld.accurate_forward(xt_next, (torch.ones(1) * seq[timestep - 1]).to(xt.device), c=c_local)
                                    eps_mu_t_next_uc = custom_ld.accurate_forward(xt_next, (torch.ones(1) * seq[timestep - 1]).to(xt.device), c=uc_local)
                                    eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=eps_mu_t_next_c, exp_eps_uc=eps_mu_t_next_uc)

                        # Diversity sampling block: ensure conds match new_xt batch
                        diversity_sample = []
                        for loop in range(6):
                            diversity = 7
                            new_xt = []

                            rep = diversity + 1

                            # Build repeated conds for rep-sized batch
                            base_c = model.get_learned_conditioning(prompt_list)
                            base_uc = model.get_learned_conditioning([""] * len(prompt_list))
                            # we'll match to new_xt after stacking new_xt

                            var_xt_next = torch.clamp(var_xt_next, min=0)
                            print(var_xt_next.sum())

                            new_xt.append(sample_from_gaussion(exp_xt_next, torch.zeros_like(exp_xt_next).to(device)))
                            for i in range(diversity):
                                new_xt.append(sample_from_gaussion(exp_xt_next, var_xt_next))
                            new_xt = torch.cat(new_xt, dim=0).to(device)   # (rep, C, H, W)
                            print(new_xt.shape)

                            # Now match base conds to new_xt batch
                            c_rep = _match_cond_batch(base_c, new_xt)
                            uc_rep = _match_cond_batch(base_uc, new_xt)

                            new_eps_mu_t_next_c = custom_ld.accurate_forward(new_xt, (torch.ones((rep)) * seq[cut]).to(new_xt.device), c=c_rep)
                            new_eps_mu_t_next_uc = custom_ld.accurate_forward(new_xt, (torch.ones((rep)) * seq[cut]).to(new_xt.device), c=uc_rep)
                            new_eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=new_eps_mu_t_next_c, exp_eps_uc=new_eps_mu_t_next_uc)

                            for timestep in range(cut, 0, -1):
                                if timestep == cut:
                                    xt = new_xt
                                    eps_mu_t = new_eps_mu_t_next
                                else:
                                    xt = xt_next
                                    eps_mu_t = eps_mu_t_next
                                xt_next = singlestep_ddim_sample(model, xt, seq, timestep, eps_mu_t)
                                # match conds to xt_next (which may be rep-sized)
                                c_rep2 = _match_cond_batch(c_rep, xt_next)
                                uc_rep2 = _match_cond_batch(uc_rep, xt_next)
                                eps_mu_t_next_c = custom_ld.accurate_forward(xt_next, (torch.ones((rep)) * seq[timestep - 1]).to(xt.device), c=c_rep2)
                                eps_mu_t_next_uc = custom_ld.accurate_forward(xt_next, (torch.ones((rep)) * seq[timestep - 1]).to(xt.device), c=uc_rep2)
                                eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=eps_mu_t_next_c, exp_eps_uc=eps_mu_t_next_uc)

                            x_samples = model.decode_first_stage(xt_next)
                            x = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                            diversity_sample.append(x)

                            os.makedirs(os.path.join(exp_dir, f'{j}'), exist_ok=True)
                            for i in range(diversity):
                                path = os.path.join(exp_dir, f'{j}', f"{img_id}.png")
                                tvu.save_image(x.cpu()[i].float(), path)
                                img_id += 1

                        # finalize diversity grid saving
                        diversity_sample = torch.cat(diversity_sample, dim=0)
                        diversity_sample = tvu.make_grid(diversity_sample, nrow=8, padding=2)
                        tvu.save_image(diversity_sample.cpu().float(), os.path.join(exp_dir, f'{j}', f"diversity_sample_{j}.png"))

if __name__ == "__main__":
    main()

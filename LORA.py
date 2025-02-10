import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision import transforms
import torch
import torchvision
import os
import shutil
from tqdm import tqdm
import zipfile

hf_repo = 'BCMIZB/Libcom_pretrained_models'
ms_repo = 'yujieouo/Libcom_pretrained_models'

def download_pretrained_model(weight_path):
    if os.path.exists(weight_path):
        assert os.path.isfile(weight_path), weight_path
        return weight_path
    else:
        weight_path= os.path.abspath(weight_path)
        model_name = os.path.basename(weight_path)
        save_dir   = os.path.dirname(weight_path)
        download_file_from_network(model_name, save_dir)
        print('Pretrained model has been stored to ', weight_path)
        return weight_path
    
def download_entire_folder(folder_path):
    if os.path.exists(folder_path):
        assert os.path.isdir(folder_path), folder_path
        assert len(os.listdir(folder_path)) > 1, f'{folder_path} is an empty folder'
        return folder_path
    else:
        folder_path = os.path.abspath(folder_path) 
        folder_name = os.path.basename(folder_path)
        file_name   = folder_name + '.zip'
        save_dir    = os.path.dirname(folder_path)
        download_file_from_network(file_name, save_dir)
        zip_file    = zipfile.ZipFile(os.path.join(save_dir, file_name))
        zip_file.extractall(save_dir)
        os.remove(os.path.join(save_dir, file_name))
        print('Folder has been stored to ', folder_path)
        return folder_path
    
def download_file_from_network(file_name, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print('Try to download {} to {}'.format(file_name, save_dir))
    try:
        from huggingface_hub import hf_hub_download
        file_path = hf_hub_download(repo_id=hf_repo, 
                                    filename=file_name, 
                                    cache_dir=save_dir)
    except:
        from modelscope.hub.file_download import model_file_download
        file_path = model_file_download(model_id=ms_repo, 
                                        file_path=file_name, 
                                        cache_dir=save_dir, 
                                        revision='master')
    assert os.path.exists(file_path), 'Download {} failed, please try again'.format(file)
    save_path = os.path.abspath(os.path.join(save_dir, file_name))
    shutil.copyfile(os.path.abspath(file_path), save_path, follow_symlinks=True)
    assert os.path.exists(save_path), 'Move file to {} failed, please try again'.format(save_path)
    os.remove(os.path.realpath(file_path)) # delete the cache
        
if __name__ == '__main__':
    file_list   = ['BargainNet.pth', 'SOPA.pth']
    folder_list = ['openai-clip-vit-large-patch14'] 
    for file in file_list:
        weight_path = './pretrained_models/' + file
        download_pretrained_model(weight_path)
    
    for folder in folder_list:
        folder_path = './pretrained_models/' + folder
        download_entire_folder(folder_path)


import os
import cv2
from PIL import Image
import numpy as np

def opencv_to_pil(input):
    return Image.fromarray(cv2.cvtColor(input, cv2.COLOR_BGR2RGB))

def pil_to_opencv(input):
    return cv2.cvtColor(np.asarray(input), cv2.COLOR_RGB2BGR)

def read_image_opencv(input):
    if isinstance(input, str):
        assert os.path.exists(input), input
        input = cv2.imread(input, cv2.IMREAD_COLOR)        
    elif isinstance(input, Image.Image):
        input = pil_to_opencv(input)
    return input

def read_mask_opencv(input):
    if isinstance(input, str):
        assert os.path.exists(input), input
        input = cv2.imread(input, cv2.IMREAD_GRAYSCALE)        
    elif isinstance(input, Image.Image):
        input = np.asarray(input)
    return input

def read_image_pil(input):
    if isinstance(input, str):
        assert os.path.exists(input), input
        input = Image.open(input).convert('RGB')
    elif isinstance(input, np.ndarray):
        input = opencv_to_pil(input)
    return input    

def read_mask_pil(input):
    if isinstance(input, str):
        assert os.path.exists(input), input
        input = Image.open(input).convert('L')
    elif isinstance(input, np.ndarray):
        input = Image.fromarray(input).convert('L')
    return input

def convert_mask_to_bbox(mask):
    '''
    mask: (h,w) or (h,w,1)
    '''
    if mask.ndim == 3:
        mask = mask[...,0]
    binmask = np.where(mask > 127)
    x1 = int(np.min(binmask[1]))
    x2 = int(np.max(binmask[1]))
    y1 = int(np.min(binmask[0]))
    y2 = int(np.max(binmask[0]))
    return [x1, y1, x2+1, y2+1]

def fill_image_pil(image, mask, fill_pixel=(0,0,0), thresh=127):
    image = fill_image_opencv(image, mask, fill_pixel, thresh)
    return Image.fromarray(image)

def fill_image_opencv(image, mask, fill_pixel=(0,0,0), thresh=127):
    image = np.asarray(image)
    mask  = np.asarray(mask)
    mask  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    image[mask < thresh] = fill_pixel
    return image

def make_image_grid(img_list, text_list=None, resolution=(512,512), cols=None, border_color=255, border_width=5):
    if cols == None:
        cols = len(img_list)
    assert len(img_list) % cols == 0, f'{len(img_list)} % {cols} != 0'
    if isinstance(text_list, (list, tuple)):
        text_list += [''] * max(0, len(img_list) - len(text_list))
    rows = len(img_list) // cols
    hor_border = (np.ones((resolution[0], border_width, 3), dtype=np.float32) * border_color).astype(np.uint8)
    index = 0
    grid_img = []
    for i in range(rows):
        row_img = []
        for j in range(cols):
            img = read_image_opencv(img_list[index])
            img = cv2.resize(img, resolution)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if text_list and len(text_list[index]) > 0:
                cv2.putText(img, text_list[index], (10,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            row_img.append(img)
            if j < cols-1:
                row_img.append(hor_border)
            index += 1
        row_img = np.concatenate(row_img, axis=1)
        grid_img.append(row_img)
        if i < rows-1:
            ver_border = (np.ones((border_width, grid_img[-1].shape[1], 3), dtype=np.float32) * border_color).astype(np.uint8)
            grid_img.append(ver_border)
    grid_img = np.concatenate(grid_img, axis=0)
    return grid_img

def draw_bbox_on_image(input_img, bbox, color=(0,255,255), line_width=5):
    img = read_image_opencv(input_img)
    x1, y1, x2, y2 = bbox
    h,w,_ = img.shape
    x1 = max(x1, line_width)
    y1 = max(y1, line_width)
    x2 = min(x2, w-line_width)
    y2 = min(y2, h-line_width)
    img = cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness=line_width)
    return img

import torch

def check_gpu_device(device):
    assert torch.cuda.is_available(), 'Only GPU are supported'
    if isinstance(device, (int, str)):
        device = 0
        assert 0 <= device < torch.cuda.device_count(), f'invalid device id: {device}'
        device = torch.device(f'cuda:{device}')
    if isinstance(device, torch.device):
        return device
    else:
        raise Exception('invalid device type: type({})={}'.format(device, type(device)))
    
import torch
import os
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
try:
    from lightning_fabric.utilities.seed import log
    log.propagate = False
except:
    pass
from torch import device
import torchvision
import importlib

import torch
import numpy as np
from collections import abc
from einops import rearrange
from functools import partial

import multiprocessing as mp
from threading import Thread
from queue import Queue

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def _do_parallel_data_prefetch(func, Q, data, idx, idx_to_fn=False):
    # create dummy dataset instance

    # run prefetching
    if idx_to_fn:
        res = func(data, worker_id=idx)
    else:
        res = func(data)
    Q.put([idx, res])
    Q.put("Done")


def parallel_data_prefetch(
        func: callable, data, n_proc, target_data_type="ndarray", cpu_intensive=True, use_worker_id=False
):
    # if target_data_type not in ["ndarray", "list"]:
    #     raise ValueError(
    #         "Data, which is passed to parallel_data_prefetch has to be either of type list or ndarray."
    #     )
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print(
                f'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.'
            )
            data = list(data.values())
        if target_data_type == "ndarray":
            data = np.asarray(data)
        else:
            data = list(data)
    else:
        raise TypeError(
            f"The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}."
        )

    if cpu_intensive:
        Q = mp.Queue(1000)
        proc = mp.Process
    else:
        Q = Queue(1000)
        proc = Thread
    # spawn processes
    if target_data_type == "ndarray":
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(np.array_split(data, n_proc))
        ]
    else:
        step = (
            int(len(data) / n_proc + 1)
            if len(data) % n_proc != 0
            else int(len(data) / n_proc)
        )
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(
                [data[i: i + step] for i in range(0, len(data), step)]
            )
        ]
    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    # start processes
    print(f"Start prefetching...")
    import time

    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    try:
        for p in processes:
            p.start()

        k = 0
        while k < n_proc:
            # get result
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()

        raise e
    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    if target_data_type == 'ndarray':
        if not isinstance(gather_res[0], np.ndarray):
            return np.concatenate([np.asarray(r) for r in gather_res], axis=0)

        # order outputs
        return np.concatenate(gather_res, axis=0)
    elif target_data_type == 'list':
        out = []
        for r in gather_res:
            out.extend(r)
        return out
    else:
        return gather_res

"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import torch.nn.functional as F
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        # if conditioning is not None:
        #     if isinstance(conditioning, dict):
        #         cbs = conditioning[list(conditioning.keys())[0]].shape[0]
        #         if cbs != batch_size:
        #             print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
        #     else:
        #         if conditioning.shape[0] != batch_size:
        #             print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    **kwargs
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,**kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device) # (n, 4, 64, 64)
        else:
            img = x_T
        
        if 'test_model_kwargs' in kwargs:
            inputs = kwargs['test_model_kwargs']
        elif 'rest' in kwargs:
            inputs = kwargs['rest']
        else:
            raise Exception("kwargs must contain either 'test_model_kwargs' or 'rest' key")

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        # intermediates = {'x_inter': [img], 'mask_inter': [mask], 'pred_x0': []}
        intermediates = {'x_inter': [img], 'pred_x0': []}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # if mask is not None:
            #     assert x0 is not None
            #     img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
            #     img = img_orig * mask + (1. - mask) * img
            
            outs = self.p_sample_ddim(img, cond, ts, index=index, mask=mask, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,**kwargs)
            # img, pred_mask, pred_x0 = outs
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                # intermediates['mask_inter'].append(pred_mask)
                intermediates['pred_x0'].append(pred_x0)
        # return torch.cat([img, pred_mask], dim=1), intermediates
        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, mask=None, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,**kwargs):
        b, *_, device = *x.shape, x.device
        if 'test_model_kwargs' in kwargs:
            inputs = kwargs['test_model_kwargs']
        elif 'rest' in kwargs:
            inputs = kwargs['rest']
        else:
            raise Exception("kwargs must contain either 'test_model_kwargs' or 'rest' key")
        
        # x_start    = inputs['latent']
        bg_latent  = inputs['bg_latent']
        m = inputs['bg_mask'] if mask is None else mask
        bbox = inputs['bbox']
        x_noisy = x
        x_input = torch.cat([x_noisy, bg_latent, 1-m], dim=1)
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x_input, bbox, t, c)
        else:
            x_in = torch.cat([x_input] * 2)
            t_in = torch.cat([t] * 2)
            bbox_in = torch.cat([bbox] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, bbox_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if x.shape[1]!=4:
            pred_x0 = (x[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(dir_xt.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec
    
"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


class PLMSSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        if ddim_eta != 0:
            raise ValueError('ddim_eta must be 0 for PLMS')
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for PLMS sampling is {size}')

        samples, intermediates = self.plms_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    **kwargs
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def plms_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,**kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = list(reversed(range(0,timesteps))) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running PLMS Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
        old_eps = []

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)
            
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_plms(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      old_eps=old_eps, t_next=ts_next,**kwargs)
            img, pred_x0, e_t = outs
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_plms(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, old_eps=None, t_next=None,**kwargs):
        b, *_, device = *x.shape, x.device
        def get_model_output(x, t):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            return e_t

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0
        kwargs=kwargs['test_model_kwargs']
        x_new=torch.cat([x,kwargs['inpaint_image'],kwargs['inpaint_mask']],dim=1)
        e_t = get_model_output(x_new, t)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            x_prev_new=torch.cat([x_prev,kwargs['inpaint_image'],kwargs['inpaint_mask']],dim=1)
            e_t_next = get_model_output(x_prev_new, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t
    
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import transformers
import shutil
import sys
from tqdm import tqdm
# import bezier
import albumentations as A
from functools import partial
import math
import copy
import torchvision.transforms as T
from torchvision import transforms
import torch.nn.functional as F
import torchvision
import torch
import cv2
import time
import json
import torch.utils.data as data

from audioop import reverse
from cmath import inf
from curses.panel import bottom_panel
from dis import dis
from email.mime import image

import os
from io import BytesIO
import logging
import base64
from sre_parse import State
from sys import prefix
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# from sympy import source
cv2.setNumThreads(0)
proj_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_dir)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from pathlib import Path


def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True, resize=True, image_size=(512, 512)):
    transform_list = []
    if resize:
        transform_list += [torchvision.transforms.Resize(image_size)]
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)


def get_tensor_clip(normalize=True, toTensor=True, resize=True, image_size=(224, 224)):
    transform_list = []
    if resize:
        transform_list += [torchvision.transforms.Resize(image_size)]
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


def scan_all_files():
    bbox_dir = os.path.join(proj_dir, '../../dataset/open-images/bbox_mask')
    assert os.path.exists(bbox_dir), bbox_dir

    bad_files = []
    for split in os.listdir(bbox_dir):
        total_images, total_pairs, bad_masks, bad_images = 0, 0, 0, 0
        subdir = os.path.join(bbox_dir, split)
        if not os.path.isdir(subdir) or split not in ['train', 'test', 'validation']:
            continue
        for file in tqdm(os.listdir(subdir)):
            try:
                with open(os.path.join(subdir, file), 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        info = line.split(' ')
                        mask_file = os.path.join(
                            bbox_dir, '../masks', split, info[-2])
                        if os.path.exists(mask_file):
                            total_pairs += 1
                        else:
                            bad_masks += 1
                total_images += 1
            except:
                bad_files.append(file)
                bad_images += 1
        print('{}, {} images({} bad), {} pairs({} bad)'.format(
            split, total_images, bad_images, total_pairs, bad_masks))

        if len(bad_files) > 0:
            with open(os.path.join(bbox_dir, 'bad_files.txt'), 'w') as f:
                for file in bad_files:
                    f.write(file + '\n')

    print(f'{len(bad_files)} bad_files')


class DataAugmentation:
    def __init__(self, border_mode=0):
        self.blur = A.Blur(p=0.3)
        self.appearance_trans = A.Compose([
            A.ColorJitter(brightness=0.3,
                          contrast=0.3,
                          saturation=0.3,
                          hue=0.05,
                          always_apply=False,
                          p=1)],
            # additional_targets={'image':'image', 'image1':'image', 'image2':'image'}
        )
        self.geometric_trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20,
                     border_mode=border_mode,
                     value=(127, 127, 127),
                     mask_value=0,
                     p=1),
            A.Perspective(scale=(0.01, 0.1),
                          pad_mode=border_mode,
                          pad_val=(127, 127, 127),
                          mask_pad_val=0,
                          fit_output=False,
                          p=0.5)
        ])
        self.bbox_maxlen = 0.8
        self.crop_bg_p = 0.5

    def __call__(self, bg_img, bbox, bg_mask, fg_img, fg_mask):
        # randomly crop background image
        if self.crop_bg_p > 0 and np.random.rand() < self.crop_bg_p:
            trans_bg, trans_bbox, trans_mask = self.random_crop_background(
                bg_img, bbox, bg_mask)
        else:
            trans_bg, trans_bbox, trans_mask = bg_img, bbox, bg_mask

        bbox_mask = bbox2mask(trans_bbox, trans_bg.shape[1], trans_bg.shape[0])
        # perform illumination and pose transformation on foreground
        trans_fg, trans_fgmask = self.augment_foreground(
            fg_img.copy(), fg_mask.copy())
        return {"bg_img":   trans_bg,
                "bg_mask":  trans_mask,
                "bbox":     trans_bbox,
                "bbox_mask": bbox_mask,
                "fg_img":   trans_fg,
                "fg_mask":  trans_fgmask,
                "gt_fg_mask": fg_mask}

    def augment_foreground(self, img, mask):
        # appearance transformed image
        transformed = self.appearance_trans(image=img)
        img = transformed['image']
        transformed = self.geometric_trans(image=img, mask=mask)
        trans_img = transformed['image']
        trans_mask = transformed['mask']
        return trans_img, trans_mask

    def random_crop_background(self, image, bbox, mask):
        width, height = image.shape[1], image.shape[0]
        bbox_w = float(bbox[2] - bbox[0]) / width
        bbox_h = float(bbox[3] - bbox[1]) / height

        left, right, top, down = 0, width, 0, height
        if bbox_w < self.bbox_maxlen:
            maxcrop = width - bbox_w * width / self.bbox_maxlen
            left = int(np.random.rand() * min(maxcrop, bbox[0]))
            right = width - int(np.random.rand() *
                                min(maxcrop, width - bbox[2]))

        if bbox_h < self.bbox_maxlen:
            maxcrop = (height - bbox_h * height / self.bbox_maxlen) / 2
            top = int(np.random.rand() * min(maxcrop, bbox[1]))
            down = height - int(np.random.rand() *
                                min(maxcrop, height - bbox[3]))

        trans_bbox = [bbox[0] - left, bbox[1] -
                      top, bbox[2] - left, bbox[3] - top]
        trans_image = image[top:down, left:right]
        trans_mask = mask[top:down, left:right]
        return trans_image, trans_bbox, trans_mask


def bbox2mask(bbox, mask_w, mask_h):
    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    mask[bbox[1]: bbox[3], bbox[0]: bbox[2]] = 255
    return mask


def mask2bbox(mask):
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=-1)
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [x1, y1, x2, y2]


def constant_pad_bbox(bbox, width, height, value=10):
    # Get reference image
    bbox_pad = copy.deepcopy(bbox)
    left_space = bbox[0]
    up_space = bbox[1]
    right_space = width - bbox[2]
    down_space = height - bbox[3]

    bbox_pad[0] = bbox[0]-min(value, left_space)
    bbox_pad[1] = bbox[1]-min(value, up_space)
    bbox_pad[2] = bbox[2]+min(value, right_space)
    bbox_pad[3] = bbox[3]+min(value, down_space)
    return bbox_pad


def rescale_image_with_bbox(image, bbox=None, long_size=1024):
    src_width, src_height = image.size
    if max(src_width, src_height) <= long_size:
        dst_img = image
        dst_width, dst_height = dst_img.size
    else:
        scale = float(long_size) / max(src_width, src_height)
        dst_width, dst_height = int(scale * src_width), int(scale * src_height)
        dst_img = image.resize((dst_width, dst_height))
    if bbox == None:
        return dst_img
    bbox[0] = int(float(bbox[0]) / src_width * dst_width)
    bbox[1] = int(float(bbox[1]) / src_height * dst_height)
    bbox[2] = int(float(bbox[2]) / src_width * dst_width)
    bbox[3] = int(float(bbox[3]) / src_height * dst_height)
    return dst_img, bbox


def crop_foreground_by_bbox(img, mask, bbox, pad_bbox=10):
    width, height = img.shape[1], img.shape[0]
    bbox_pad = constant_pad_bbox(
        bbox, width, height, pad_bbox) if pad_bbox > 0 else bbox
    img = img[bbox_pad[1]:bbox_pad[3], bbox_pad[0]:bbox_pad[2]]
    if mask is not None:
        mask = mask[bbox_pad[1]:bbox_pad[3], bbox_pad[0]:bbox_pad[2]]
    return img, mask, bbox_pad


def image2inpaint(image, mask):
    if len(mask.shape) == 2:
        mask_f = mask[:, :, np.newaxis]
    else:
        mask_f = mask
    mask_f = mask_f.astype(np.float32) / 255
    inpaint = image.astype(np.float32)
    gray = np.ones_like(inpaint) * 127
    inpaint = inpaint * (1 - mask_f) + mask_f * gray
    inpaint = np.uint8(inpaint)
    return inpaint


def check_dir(dir):
    assert os.path.exists(dir), dir
    return dir


def get_bbox_tensor(bbox, width, height):
    norm_bbox = bbox
    norm_bbox = torch.tensor(norm_bbox).reshape(-1).float()
    norm_bbox[0::2] /= width
    norm_bbox[1::2] /= height
    return norm_bbox


def reverse_image_tensor(tensor, img_size=(256, 256)):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = (tensor.float() + 1) / 2
    tensor = torch.clamp(tensor, min=0.0, max=1.0)
    tensor = torch.permute(tensor, (0, 2, 3, 1)) * 255
    tensor = tensor.detach().cpu().numpy()
    img_nps = np.uint8(tensor)

    def np2bgr(img, img_size=img_size):
        if img.shape[:2] != img_size:
            img = cv2.resize(img, img_size)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_list = [np2bgr(img) for img in img_nps]
    return img_list


def reverse_mask_tensor(tensor, img_size=(256, 256)):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = torch.clamp(tensor, min=0.0, max=1.0)
    tensor = torch.permute(tensor.float(), (0, 2, 3, 1)) * 255
    tensor = tensor.detach().cpu().numpy()
    img_nps = np.uint8(tensor)

    def np2bgr(img, img_size=img_size):
        if img.shape[:2] != img_size:
            img = cv2.resize(img, img_size)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_list = [np2bgr(img) for img in img_nps]
    return img_list


def reverse_clip_tensor(tensor, img_size=(256, 256)):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    MEAN = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073],  dtype=torch.float)
    MEAN = MEAN.reshape(1, 3, 1, 1).to(tensor.device)
    STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float)
    STD = STD.reshape(1, 3, 1, 1).to(tensor.device)
    tensor = (tensor * STD) + MEAN
    tensor = torch.clamp(tensor, min=0.0, max=1.0)
    tensor = torch.permute(tensor.float(), (0, 2, 3, 1)) * 255
    tensor = tensor.detach().cpu().numpy()
    img_nps = np.uint8(tensor)

    def np2bgr(img, img_size=img_size):
        if img.shape[:2] != img_size:
            img = cv2.resize(img, img_size)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_list = [np2bgr(img) for img in img_nps]
    return img_list


def random_crop_image(image, crop_w, crop_h):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    x_space = image.shape[1] - crop_w
    y_space = image.shape[0] - crop_h
    x1 = np.random.randint(0, x_space) if x_space > 0 else 0
    y1 = np.random.randint(0, y_space) if y_space > 0 else 0
    image = image[y1: y1+crop_h, x1: x1+crop_w]
    # assert crop.shape[0] == crop_h and crop.shape[1] == crop_w, (y1, x1, image.shape, crop.shape, crop_w, crop_h)
    return image


def read_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
    return img


def read_mask(image_path):
    with Image.open(image_path) as img:
        img = img.convert('L')
    return img

# poisson blending


def poisson_blending(fg, fg_mask, bg, center=None):
    if center is None:
        height, width, _ = bg.shape
        center = (int(width/2), int(height/2))
    return cv2.seamlessClone(fg, bg, fg_mask, center, cv2.MIXED_CLONE)


class OpenImageDataset(data.Dataset):
    def __init__(self, split, **args):
        self.split = split
        dataset_dir = args['dataset_dir']
        assert os.path.exists(dataset_dir), dataset_dir
        self.bbox_dir = check_dir(os.path.join(
            dataset_dir, 'refine/box', split))
        self.image_dir = check_dir(os.path.join(dataset_dir, 'images', split))
        self.inpaint_dir = check_dir(os.path.join(
            dataset_dir, 'refine/inpaint', split))
        self.mask_dir = check_dir(os.path.join(
            dataset_dir, 'refine/mask', split))
        self.bbox_path_list = np.array(self.load_bbox_path_list())
        self.length = len(self.bbox_path_list)
        self.random_trans = DataAugmentation()
        self.clip_transform = get_tensor_clip(image_size=(224, 224))
        self.image_size = (args['image_size'], args['image_size'])
        self.sd_transform = get_tensor(image_size=self.image_size)
        self.mask_transform = get_tensor(
            normalize=False, image_size=self.image_size)
        self.clip_mask_transform = get_tensor(
            normalize=False, image_size=(224, 224))
        self.bad_images = []

    def load_bbox_path_list(self):
        # cache_dir  = os.path.dirname(os.path.abspath(self.bbox_dir))
        cache_dir = self.bbox_dir  # (dataset_dir/refine/box)
        cache_file = os.path.join(cache_dir, f'{self.split}.json')
        if os.path.exists(cache_file):
            print('load bbox list from ', cache_file)
            with open(cache_file, 'r') as f:
                bbox_path_list = json.load(f)
        else:
            bbox_path_list = os.listdir(self.bbox_dir)
            bbox_path_list.sort()
            print('save bbox list to ', cache_file)
            with open(cache_file, 'w') as f:
                json.dump(bbox_path_list, f)
        return bbox_path_list

    def load_bbox_file(self, bbox_file):
        bbox_list = []
        with open(bbox_file, 'r') as f:
            for line in f.readlines():
                info = line.strip().split(' ')
                bbox = [int(float(f)) for f in info[:4]]
                mask = os.path.join(self.mask_dir, info[-1])
                inpaint = os.path.join(
                    self.inpaint_dir, info[-1].replace('.png', '.jpg'))
                if os.path.exists(mask) and os.path.exists(inpaint):
                    bbox_list.append((bbox, mask, inpaint))
        return bbox_list

    
    def sample_augmented_data(self, source_np, bbox, mask, fg_img, fg_mask):
        transformed = self.random_trans(source_np, bbox, mask, fg_img, fg_mask)
        # get ground-truth composite image and bbox
        gt_mask = Image.fromarray(transformed["bg_mask"])
        img_width, img_height = gt_mask.size
        gt_mask_tensor = self.mask_transform(gt_mask)
        gt_mask_tensor = torch.where(gt_mask_tensor > 0.5, 1, 0).float()

        gt_img_tensor = Image.fromarray(transformed['bg_img'])
        gt_img_tensor = self.sd_transform(gt_img_tensor)

        mask_tensor = Image.fromarray(transformed['bbox_mask'])
        mask_tensor = self.mask_transform(mask_tensor)
        mask_tensor = torch.where(mask_tensor > 0.5, 1, 0).float()
        bbox_tensor = transformed['bbox']
        bbox_tensor = get_bbox_tensor(bbox_tensor, img_width, img_height)
        # get foreground and foreground mask
        fg_mask_tensor = Image.fromarray(transformed['fg_mask'])
        fg_mask_tensor = self.clip_mask_transform(fg_mask_tensor)
        fg_mask_tensor = torch.where(fg_mask_tensor > 0.5, 1, 0)

        fg_img_tensor = transformed['fg_img'] * \
            (transformed['fg_mask'][:, :, None] > 0.5)
        fg_img_tensor = Image.fromarray(fg_img_tensor)
        fg_img_tensor = self.clip_transform(fg_img_tensor)
        inpaint = gt_img_tensor * (mask_tensor < 0.5)

        return {"gt_img":  gt_img_tensor,
                "gt_mask": gt_mask_tensor,
                "bg_img": inpaint,
                "bg_mask": mask_tensor,
                "fg_img":  fg_img_tensor,
                "fg_mask": fg_mask_tensor,
                "bbox": bbox_tensor}

    def __getitem__(self, index):
        try:
            # get bbox and mask
            bbox_file = self.bbox_path_list[index]
            bbox_path = os.path.join(self.bbox_dir, bbox_file)  # 每一个bbox文件的路径
            # 一个list里面包含多个元组，每个元组包含(bbox, mask, inpaint)
            bbox_list = self.load_bbox_file(bbox_path)
            bbox, mask_path, inpaint_path = random.choice(bbox_list)  # 随机选择其中的一个元组，获得bbox和对应的mask_path
            # get source image and mask
            image_path = os.path.join(self.image_dir, os.path.splitext(bbox_file)[0] + '.jpg')  # 获得原始图像
            source_img = read_image(image_path)
            source_img, bbox = rescale_image_with_bbox(source_img, bbox)
            source_np = np.array(source_img)
            mask = read_mask(mask_path)
            mask = mask.resize((source_np.shape[1], source_np.shape[0]))
            mask = np.array(mask)
            # bbox = mask2bbox(mask)
            fg_img, fg_mask, bbox = crop_foreground_by_bbox(source_np, mask, bbox)
            sample = self.sample_augmented_data(source_np, bbox, mask, fg_img, fg_mask)
            sample['image_path'] = image_path
            return sample
        except Exception as e:
            print(os.getpid(), bbox_file, e)
            index = np.random.randint(0, len(self)-1)
            return self[index]

    def __len__(self):
        return self.length

class MureComDataset(data.Dataset):
    def __init__(self,**args):
        self.package_name = args['package_name']
        self.fg_name = args['fg_name']
        self.instance_data_root = os.path.join(args['instance_data_root'], self.package_name, self.fg_name)
        if not os.path.exists(self.instance_data_root):
            print(self.instance_data_root)
            raise ValueError("Instance images root doesn't exists.")
        self.instance_images_path=[]
        self.instance_masks_path=[]
        self.image_num = args['image_num']
        count = 0
        for temp_file_path in Path(self.instance_data_root).glob("*.jpg"):
            self.instance_images_path.append(str(temp_file_path))
            self.instance_masks_path.append(Path(str(temp_file_path).replace('jpg', 'png')))
            count+=1
            if count>=self.image_num:
                break
        self.num_instance_images = len(self.instance_images_path)
        self._length1 = self.num_instance_images
        self.random_trans = DataAugmentation()
        self.clip_transform = get_tensor_clip(image_size=(224, 224))
        self.image_size = (args['image_size'], args['image_size'])
        self.sd_transform = get_tensor(image_size=self.image_size)
        self.mask_transform = get_tensor(
            normalize=False, image_size=self.image_size)
        self.clip_mask_transform = get_tensor(
            normalize=False, image_size=(224, 224))
        self.instance_mask_list, self.bbox_list, self.fg_img_list = [], [], []
        self.instance_image_list, self.fg_mask_list = [], []
        for mask_path in self.instance_masks_path:
            instance_mask = Image.open(mask_path).convert('L')
            mask = np.array(instance_mask)
            m = np.array(mask>0).astype(np.uint8)
            instance_mask = Image.fromarray(m * 255)
            
            self.instance_mask_list.append(instance_mask)
            x_left, x_right, y_bottom, y_top = self.mask_bboxregion_coordinate(np.array(instance_mask))
            H, W = (np.array(instance_mask)).shape[:2]
            x_right=min(x_right,W-1)
            y_bottom=min(y_bottom,H-1)
            bbox = [x_left, y_top, x_right, y_bottom]
            self.bbox_list.append(bbox)
            
        for idx, instance_image_path in enumerate(self.instance_images_path):
            instance_image = Image.open(instance_image_path).convert("RGB")
            instance_image, bbox = rescale_image_with_bbox(instance_image, self.bbox_list[idx])
            instance_image = np.array(instance_image)
            self.instance_mask_list[idx].resize((instance_image.shape[1], instance_image.shape[0]))
            mask = np.array(self.instance_mask_list[idx])
            fg_img, fg_mask, bbox = crop_foreground_by_bbox(instance_image, mask, bbox)
            self.fg_img_list.append(fg_img)
            self.bbox_list[idx] = bbox
            self.instance_image_list.append(instance_image)
            self.fg_mask_list.append(fg_mask)
        
    def __len__(self):
        return self._length1
    def mask_bboxregion_coordinate(self, mask):
        valid_index = np.argwhere(mask == 255)  # [length,2]
        if np.shape(valid_index)[0] < 1:
            x_left = 0
            x_right = 0
            y_bottom = 0
            y_top = 0
        else:
            x_left = np.min(valid_index[:, 1])
            x_right = np.max(valid_index[:, 1])
            y_bottom = np.max(valid_index[:, 0])
            y_top = np.min(valid_index[:, 0])
        return x_left, x_right, y_bottom, y_top

    def sample_augmented_data(self, source_np, bbox, mask, fg_img, fg_mask):
        transformed = self.random_trans(source_np, bbox, mask, fg_img, fg_mask)
        # get ground-truth composite image and bbox
        gt_mask = Image.fromarray(transformed["bg_mask"])
        img_width, img_height = gt_mask.size
        gt_mask_tensor = self.mask_transform(gt_mask)
        gt_mask_tensor = torch.where(gt_mask_tensor > 0.5, 1, 0).float()

        gt_img_tensor = Image.fromarray(transformed['bg_img'])
        gt_img_tensor = self.sd_transform(gt_img_tensor)

        mask_tensor = Image.fromarray(transformed['bbox_mask'])
        mask_tensor = self.mask_transform(mask_tensor)
        mask_tensor = torch.where(mask_tensor > 0.5, 1, 0).float()
        bbox_tensor = transformed['bbox']
        bbox_tensor = get_bbox_tensor(bbox_tensor, img_width, img_height)
        # get foreground and foreground mask
        fg_mask_tensor = Image.fromarray(transformed['fg_mask'])
        fg_mask_tensor = self.clip_mask_transform(fg_mask_tensor)
        fg_mask_tensor = torch.where(fg_mask_tensor > 0.5, 1, 0)

        fg_img_tensor = transformed['fg_img'] * \
            (transformed['fg_mask'][:, :, None] > 0.5)
        fg_img_tensor = Image.fromarray(fg_img_tensor)
        fg_img_tensor = self.clip_transform(fg_img_tensor)
        inpaint = gt_img_tensor * (mask_tensor < 0.5)
        
        fg_img_list_tensor = []
        for idx, fg_img in enumerate(self.fg_img_list):
            transformed = self.random_trans(self.instance_image_list[idx], self.bbox_list[idx], np.array(self.instance_mask_list[idx]), fg_img, self.fg_mask_list[idx])
            fg_img_sample =  transformed['fg_img'] * (transformed['fg_mask'][:, :, None] > 0.5)   
            fg_img_sample = Image.fromarray(fg_img_sample)
            fg_img_sample = self.clip_transform(fg_img_sample)
            fg_img_list_tensor.append(fg_img_sample)
            
        return {"gt_img":  gt_img_tensor,
                "gt_mask": gt_mask_tensor,
                "bg_img": inpaint,
                "bg_mask": mask_tensor,
                "fg_img":  fg_img_tensor,
                "fg_mask": fg_mask_tensor,
                "fg_img_list": fg_img_list_tensor,
                "bbox": bbox_tensor}
        
    def __getitem__(self, index):
        instance_mask = Image.open(self.instance_masks_path[index%self.num_instance_images]).convert('L')
        # mask膨胀处理
        mask = np.asarray(instance_mask)
        m = np.array(mask>0).astype(np.uint8) 
        #m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)
        instance_mask = Image.fromarray(m * 255) ## 得到了一个标准的除了0和255没有其他元素的mask
        x_left, x_right, y_bottom, y_top = self.mask_bboxregion_coordinate(np.array(instance_mask))
        H, W = (np.array(instance_mask)).shape[:2]
        x_right=min(x_right,W-1) 
        y_bottom=min(y_bottom,H-1) 
        bbox = [x_left, y_top, x_right, y_bottom]
        
        instance_image = Image.open(self.instance_images_path[index%self.num_instance_images]).convert("RGB")
        instance_image, bbox = rescale_image_with_bbox(instance_image, bbox)
        instance_image = np.array(instance_image)
        instance_mask = instance_mask.resize((instance_image.shape[1], instance_image.shape[0]))
        mask = np.array(instance_mask)
        fg_img, fg_mask, bbox = crop_foreground_by_bbox(instance_image, mask, bbox)
        sample = self.sample_augmented_data(instance_image, bbox, mask, fg_img, fg_mask)
        sample['image_path'] = self.instance_images_path[index%self.num_instance_images]
        return sample  



    
class COCOEEDataset(data.Dataset):
    def __init__(self, **args):
        dataset_dir = args['dataset_dir']
        self.use_inpaint_background = args['augment_config'].use_inpaint_background if 'augment_config' in args else True
        assert os.path.exists(dataset_dir), dataset_dir
        self.src_dir = check_dir(os.path.join(dataset_dir, "GT_3500"))
        self.ref_dir = check_dir(os.path.join(dataset_dir, 'Ref_3500'))
        self.mask_dir = check_dir(os.path.join(dataset_dir, 'Mask_bbox_3500'))
        self.gt_mask_dir = check_dir(os.path.join(dataset_dir, 'mask'))
        self.inpaint_dir = check_dir(os.path.join(dataset_dir, 'inpaint'))
        self.ref_mask_dir = check_dir(os.path.join(dataset_dir, 'ref_mask'))
        self.image_list = os.listdir(self.src_dir)
        self.image_list.sort()

        self.clip_transform = get_tensor_clip(image_size=(224, 224))
        self.image_size = args['image_size'], args['image_size']
        self.sd_transform = get_tensor(image_size=self.image_size)
        self.mask_transform = get_tensor(
            normalize=False, image_size=self.image_size)
        self.clip_mask_transform = get_tensor(
            normalize=False, image_size=(224, 224))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        try:
            image = self.image_list[index]
            src_path = os.path.join(self.src_dir, image)
            src_img = read_image(src_path)
            src_tensor = self.sd_transform(src_img)
            im_name = os.path.splitext(image)[0].split('_')[0]
            # reference image and object mask
            ref_name = im_name + '_ref.png'

            ref_mask_path = os.path.join(self.ref_mask_dir, ref_name)
            assert os.path.exists(ref_mask_path), ref_mask_path
            ref_mask = read_mask(ref_mask_path)
            ref_mask_np = np.array(ref_mask)
            ref_mask_tensor = self.clip_mask_transform(ref_mask)
            ref_mask_tensor = torch.where(ref_mask_tensor > 0.5, 1, 0)

            ref_path = os.path.join(self.ref_dir, ref_name)
            assert os.path.exists(ref_path), ref_path
            ref_img = read_image(ref_path)
            ref_img_np = np.array(ref_img) * (ref_mask_np[:, :, None] > 0.5)
            ref_img = Image.fromarray(ref_img_np)
            ref_tensor = self.clip_transform(ref_img)

            mask_path = os.path.join(self.mask_dir, im_name + '_mask.png')
            assert os.path.exists(mask_path), mask_path
            mask_img = read_mask(mask_path)
            mask_img = mask_img.resize((src_img.width, src_img.height))
            bbox = mask2bbox(np.array(mask_img))
            bbox_tensor = get_bbox_tensor(bbox, src_img.width, src_img.height)
            mask_tensor = self.mask_transform(mask_img)
            mask_tensor = torch.where(mask_tensor > 0.5, 1, 0).float()
            inpaint_tensor = src_tensor * (1 - mask_tensor)

            return {"image_path": src_path,
                    "gt_img":  src_tensor,
                    "gt_mask": mask_tensor,
                    "bg_img":  inpaint_tensor,
                    "bg_mask": mask_tensor,
                    "fg_img":  ref_tensor,
                    'fg_mask': ref_mask_tensor,
                    "bbox":    bbox_tensor}
        except:
            idx = np.random.randint(0, len(self)-1)
            return self[idx]


class FOSEDataset(data.Dataset):
    def __init__(self, dataset_dir='path-to-FOSCom'):
        data_root = dataset_dir
        self.bg_dir = os.path.join(data_root, 'background')
        self.mask_dir = os.path.join(data_root, 'bbox_mask')
        self.bbox_dir = os.path.join(data_root, 'bbox')
        self.fg_dir = os.path.join(data_root, 'foreground')
        self.fgmask_dir = os.path.join(data_root, 'foreground_mask')
        self.image_list = os.listdir(self.bg_dir)
        self.image_size = (512, 512)
        self.clip_transform = get_tensor_clip(image_size=(224, 224))
        self.sd_transform = get_tensor(image_size=self.image_size)
        self.mask_transform = get_tensor(
            normalize=False, image_size=self.image_size)

    def __len__(self):
        return len(self.image_list)

    def load_bbox_file(self, bbox_file):
        bbox_list = []
        with open(bbox_file, 'r') as f:
            for line in f.readlines():
                info = line.strip().split(' ')
                bbox = [int(float(f)) for f in info[:4]]
                bbox_list.append(bbox)
        return bbox_list[0]

    def __getitem__(self, index):
        image = self.image_list[index]
        bg_path = os.path.join(self.bg_dir, image)
        bg_img = Image.open(bg_path).convert('RGB')
        bg_w, bg_h = bg_img.size
        bg_t = self.sd_transform(bg_img)
        fg_path = os.path.join(self.fg_dir, image)
        fg_img = Image.open(fg_path).convert('RGB')
        fgmask_path = os.path.join(self.fgmask_dir, image)
        fg_mask = Image.open(fgmask_path).convert('L')
        fg_np = np.array(fg_img) * (np.array(fg_mask)[:, :, None] > 0.5)
        fg_img = Image.fromarray(fg_np)

        fg_t = self.clip_transform(fg_img)
        fgmask_t = self.mask_transform(fg_mask)
        mask_path = os.path.join(self.mask_dir, image)
        mask = Image.open(mask_path).convert('L')
        mask_t = self.mask_transform(mask)
        mask_t = torch.where(mask_t > 0.5, 1, 0).float()
        inpaint_t = bg_t * (1 - mask_t)
        bbox_path = os.path.join(self.bbox_dir, image.replace('.png', '.txt'))
        bbox = self.load_bbox_file(bbox_path)
        bbox_t = get_bbox_tensor(bbox, bg_w, bg_h)

        return {"image_path": bg_path,
                "bg_img":  bg_t,
                "inpaint_img":  inpaint_t,
                "bg_mask": mask_t,
                "fg_img":  fg_t,
                "fg_mask": fgmask_t,
                'bbox': bbox_t}


def test_fos_dataset():
    dataset = FOSEDataset()
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=4,
                                 shuffle=False,
                                 num_workers=4)
    for i, batch in enumerate(dataloader):
        print(i, len(dataset),
              batch['inpaint_img'].shape, batch['fg_img'].shape)


def vis_random_augtype(batch):
    file = batch['image_path']
    gt_t = batch['gt_img']
    gtmask_t = batch['gt_mask']
    bg_t = batch['bg_img']
    bgmask_t = batch['bg_mask']
    fg_t = batch['fg_img']
    fgmask_t = batch['fg_mask']

    gt_imgs = reverse_image_tensor(gt_t)
    gt_masks = reverse_mask_tensor(gtmask_t)
    bg_imgs = reverse_image_tensor(bg_t)
    fg_imgs = reverse_clip_tensor(fg_t)
    fg_masks = reverse_mask_tensor(fgmask_t)

    ver_border = np.ones((gt_imgs[0].shape[0], 10, 3),
                         dtype=np.uint8) * np.array([0, 0, 250]).reshape((1, 1, -1))
    img_list = []
    for i in range(len(gt_imgs)):
        im_name = os.path.basename(file[i]) if len(
            file) > 1 else os.path.basename(file[0])
        cat_img = np.concatenate([bg_imgs[i], ver_border, fg_imgs[i], ver_border,
                                 fg_masks[i], ver_border, gt_imgs[i], ver_border, gt_masks[i]], axis=1)
        if i > 0:
            hor_border = np.ones(
                (10, cat_img.shape[1], 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1, 1, -1))
            img_list.append(hor_border)
        img_list.append(cat_img)
    img_batch = np.concatenate(img_list, axis=0)
    return img_batch


def test_cocoee_dataset():
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from torch.utils.data import DataLoader
    cfg_path = os.path.join(proj_dir, 'configs/v1.yaml')
    configs = OmegaConf.load(cfg_path).data.params.validation
    dataset = instantiate_from_config(configs)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=4,
                            shuffle=False,
                            num_workers=4)
    print('{} samples = {} bs x {} batches'.format(
        len(dataset), dataloader.batch_size, len(dataloader)
    ))
    vis_dir = os.path.join(proj_dir, 'outputs/test_samples')
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir, exist_ok=True)
    for i, batch in enumerate(dataloader):
        file = batch['image_path']
        gt_t = batch['gt_img']
        bgmask_t = batch['bg_mask']
        fg_t = batch['fg_img']
        bbox_t = batch['bbox']
        bg_t = batch['bg_img']
        fg_mask = batch['fg_mask']
        im_name = os.path.basename(file[0])
        # test_fill_mask(batch, i)
        print(i, len(dataloader), gt_t.shape, fg_t.shape,
              gt_t.shape, bbox_t.shape, fg_mask.shape)
        batch_img = vis_random_augtype(batch)
        cv2.imwrite(os.path.join(vis_dir, f'batch{i}.jpg'), batch_img)
        if i > 10:
            break


def test_open_images():
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from torch.utils.data import DataLoader
    cfg_path = os.path.join(proj_dir, 'configs/v1.yaml')
    configs = OmegaConf.load(cfg_path).data.params.train
    configs.params.split = 'validation'
    dataset = instantiate_from_config(configs)
    bs = 4
    dataloader = DataLoader(dataset=dataset,
                            batch_size=bs,
                            shuffle=False,
                            num_workers=4)
    print('{} samples = {} bs x {} batches'.format(
        len(dataset), dataloader.batch_size, len(dataloader)
    ))
    vis_dir = os.path.join(proj_dir, 'outputs/train_samples')
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir, exist_ok=True)

    for i, batch in enumerate(dataloader):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor) and batch[k].shape[0] == 1:
                batch[k] = batch[k][0]

        file = batch['image_path']
        gt_t = batch['gt_img']
        gtmask_t = batch['gt_mask']
        bgmask_t = batch['bg_mask']
        fg_t = batch['fg_img']
        bbox_t = batch['bbox']
        im_name = os.path.basename(file[0])
        # test_fill_mask(batch, i)
        print(i, len(dataloader), gt_t.shape, gtmask_t.shape,
              fg_t.shape, gt_t.shape, bbox_t.shape)
        batch_img = vis_random_augtype(batch)
        cv2.imwrite(os.path.join(vis_dir, f'batch{i}.jpg'), batch_img)
        if i > 10:
            break


def test_open_images_efficiency():
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from torch.utils.data import DataLoader
    cfg_path = os.path.join(proj_dir, 'configs/v1.yaml')
    configs = OmegaConf.load(cfg_path).data.params.train
    configs.params.split = 'train'
    dataset = instantiate_from_config(configs)
    bs = 16
    dataloader = DataLoader(dataset=dataset,
                            batch_size=bs,
                            shuffle=False,
                            num_workers=16)
    print('{} samples = {} bs x {} batches'.format(
        len(dataset), dataloader.batch_size, len(dataloader)
    ))
    start = time.time()
    data_len = len(dataloader)
    for i, batch in enumerate(dataloader):
        image = batch['gt_img']
        end = time.time()
        if i % 10 == 0:
            print('{:.2f}, avg time {:.1f}ms'.format(
                float(i) / data_len, (end-start) / (i+1) * 1000
            ))



import copy

cur_dir   = os.path.dirname(os.path.abspath(__file__))
model_dir = os.environ.get('LIBCOM_MODEL_DIR',cur_dir)
model_set = ['ObjectStitch']

class Mure_ObjectStitchModel:
    """
    Unofficial implementation of the paper "ObjectStitch: Object Compositing with Diffusion Model", CVPR 2023.
    Building upon ObjectStitch, we have made improvements to support input of multiple foreground images.

    Args:
        device (str | torch.device): gpu id
        model_type (str): predefined model type.
        kwargs (dict): sampler='ddim' (default) or 'plms', other parameters for building model

    Examples:
        >>> from libcom import Mure_ObjectStitchModel
        >>> from libcom.utils.process_image import make_image_grid, draw_bbox_on_image
        >>> import cv2
        >>> import os
        >>> net    = Mure_ObjectStitchModel(device=0, sampler='plms')
        >>> sample_list = ['000000000003', '000000000004']
        >>> sample_dir  = './tests/mure_objectstitch/'
        >>> bbox_list   = [[623, 1297, 1159, 1564], [363, 205, 476, 276]]
        >>> for i, sample in enumerate(sample_list):
        >>>     bg_img = sample_dir + f'background/{sample}.jpg'
        >>>     fg_img_path = sample_dir + f'foreground/{sample}/'
        >>>     fg_mask_path = sample_dir + f'foreground_mask/{sample}/'
        >>>     fg_img_list = [os.path.join(fg_img_path, f) for f in os.listdir(fg_img_path)]
        >>>     fg_mask_list = [os.path.join(fg_mask_path, f) for f in os.listdir(fg_mask_path)]
        >>>     bbox   = bbox_list[i]
        >>>     comp, show_fg_img = net(bg_img, fg_img_list, fg_mask_list, bbox, sample_steps=25, num_samples=3)
        >>>     bg_img   = draw_bbox_on_image(bg_img, bbox)
        >>>     grid_img = make_image_grid([bg_img, show_fg_img] + [comp[i] for i in range(len(comp))])
        >>>     cv2.imwrite(f'../docs/_static/image/mureobjectstitch_result{i+1}.jpg', grid_img)

    Expected result:

    .. image:: _static/image/mureobjectstitch_result1.jpg
        :scale: 21 %
    .. image:: _static/image/mureobjectstitch_result2.jpg
        :scale: 21 %


    """
    def __init__(self, device=0, model_type='ObjectStitch', **kwargs):
        assert model_type in model_set, f'Not implementation for {model_type}'
        self.model_type = model_type
        self.option = kwargs

        weight_path = os.path.join(cur_dir, 'pretrained_models', f'{self.model_type}.pth')
        download_pretrained_model(weight_path)

        self.device = check_gpu_device(device)
        self.build_pretrained_model(weight_path)
        self.build_data_transformer()

    def build_pretrained_model(self, weight_path):
        pl_sd  = torch.load(weight_path, map_location="cpu")
        sd     = pl_sd["state_dict"]
        config = OmegaConf.load(os.path.join(cur_dir, 'source/ObjectStitch/configs/objectstitch.yaml'))
        clip_path = os.path.join(model_dir, '../shared_pretrained_models', 'openai-clip-vit-large-patch14')
        download_entire_folder(clip_path)
        config.model.params.cond_stage_config.params.version = clip_path
        model  = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)
        self.model   = model.to(self.device).eval()
        if self.option.get('sampler', 'ddim') == 'plms':
            self.sampler = PLMSSampler(self.model)
        else:
            self.sampler = DDIMSampler(self.model)

    def build_data_transformer(self):
        self.image_size       = (512, 512)
        self.clip_transform = get_tensor_clip(image_size=(224, 224))
        self.sd_transform   = get_tensor(image_size=self.image_size)
        self.mask_transform = get_tensor(normalize=False, image_size=self.image_size)
        self.latent_shape   = [4, self.image_size[0] // 8, self.image_size[1] // 8]

    def constant_pad_bbox(self, bbox, width, height, value=10):
        # Get reference image
        bbox_pad = copy.deepcopy(bbox)
        left_space = bbox[0]
        up_space = bbox[1]
        right_space = width - bbox[2]
        down_space = height - bbox[3]

        bbox_pad[0] = bbox[0]-min(value, left_space)
        bbox_pad[1] = bbox[1]-min(value, up_space)
        bbox_pad[2] = bbox[2]+min(value, right_space)
        bbox_pad[3] = bbox[3]+min(value, down_space)
        return bbox_pad

    def crop_foreground_by_bbox(self, img, mask, bbox, pad_bbox=10):
        width, height = img.shape[1], img.shape[0]
        bbox_pad = self.constant_pad_bbox(
            bbox, width, height, pad_bbox) if pad_bbox > 0 else bbox
        img = img[bbox_pad[1]:bbox_pad[3], bbox_pad[0]:bbox_pad[2]]
        if mask is not None:
            mask = mask[bbox_pad[1]:bbox_pad[3], bbox_pad[0]:bbox_pad[2]]
        return img, mask, bbox_pad

    def draw_compose_fg_img(self, fg_img_compose):
        final_img = Image.new('RGB', (512, 512), (255, 255, 255))
        fg_img_nums = len(fg_img_compose)

        if fg_img_nums == 1:
            size = (512, 512)
            positions = [(0, 0)]
        elif fg_img_nums == 2:
            size = (256, 512)
            positions = [(0, 0), (256, 0)]
        elif fg_img_nums == 3:
            size = (256, 512)
            positions = [(0, 0), (256, 0), (0, 256)]
        elif fg_img_nums == 4:
            positions = [(0, 0), (256, 0), (0, 256), (256, 256)]
            size = (256, 256)
        else :
            positions = [(0, 0), (256, 0), (0, 256), (256, 256), (128, 128)]
            size = (256, 256)

        if fg_img_nums>5:
            fg_img_compose = fg_img_compose[:5]

        for idx, img in enumerate(fg_img_compose):
            fg_img = img.resize(size)
            final_img.paste(fg_img, positions[idx])

        return final_img

    def rescale_image_with_bbox(self, image, bbox=None, long_size=1024):
        src_width, src_height = image.size
        if max(src_width, src_height) <= long_size:
            dst_img = image
            dst_width, dst_height = dst_img.size
        else:
            scale = float(long_size) / max(src_width, src_height)
            dst_width, dst_height = int(scale * src_width), int(scale * src_height)
            dst_img = image.resize((dst_width, dst_height))
        if bbox == None:
            return dst_img
        bbox[0] = int(float(bbox[0]) / src_width * dst_width)
        bbox[1] = int(float(bbox[1]) / src_height * dst_height)
        bbox[2] = int(float(bbox[2]) / src_width * dst_width)
        bbox[3] = int(float(bbox[3]) / src_height * dst_height)
        return dst_img, bbox

    def mask_bboxregion_coordinate(self, mask):
        valid_index = np.argwhere(mask == 255)  # [length,2]
        if np.shape(valid_index)[0] < 1:
            x_left = 0
            x_right = 0
            y_bottom = 0
            y_top = 0
        else:
            x_left = np.min(valid_index[:, 1])
            x_right = np.max(valid_index[:, 1])
            y_bottom = np.max(valid_index[:, 0])
            y_top = np.min(valid_index[:, 0])
        return x_left, x_right, y_bottom, y_top

    def generate_multifg(self, fg_list_path, fgmask_list_path):
        fg_list, fg_mask_list, fg_img_list, fg_img_compose = [], [], [], []

        assert len(fg_list_path) < 11, "too many foreground images"
        for fg_img_path in fg_list_path:
            fg_img = Image.open(fg_img_path).convert('RGB')
            fg_list.append(fg_img)
        for fg_mask_name in fgmask_list_path:
            fg_mask = Image.open(fg_mask_name).convert('RGB')
            fg_mask_list.append(fg_mask)

        for idx, fg_mask in enumerate(fg_mask_list):
            fg_mask = fg_mask.convert('L')
            mask = np.asarray(fg_mask)
            m = np.array(mask > 0).astype(np.uint8)
            fg_mask = Image.fromarray(m * 255)
            x_left, x_right, y_bottom, y_top = self.mask_bboxregion_coordinate(np.array(fg_mask))
            H, W = (np.array(fg_mask)).shape[:2]
            x_right=min(x_right, W-1)
            y_bottom=min(y_bottom, H-1)
            fg_bbox = [x_left, y_top, x_right, y_bottom]
            fg_img, fg_bbox = self.rescale_image_with_bbox(fg_list[idx], fg_bbox)
            fg_img = np.array(fg_img)
            fg_mask = fg_mask.resize((fg_img.shape[1], fg_img.shape[0]))
            fg_mask = np.array(fg_mask)
            fg_img, fg_mask, fg_bbox = self.crop_foreground_by_bbox(fg_img, fg_mask, fg_bbox)
            fg_mask = np.array(Image.fromarray(fg_mask).convert('RGB'))
            black = np.zeros_like(fg_mask)
            fg_img = np.where(fg_mask > 127, fg_img, black)
            fg_img = Image.fromarray(fg_img)
            fg_img_compose.append(fg_img)
            fg_t = self.clip_transform(fg_img)
            fg_img_list.append(fg_t.unsqueeze(0))
        fg_img = self.draw_compose_fg_img(fg_img_compose)

        return fg_img_list, fg_img

    def generate_image_batch(self, bg_path, fg_list_path, fgmask_list_path, bbox):

        bg_img     = Image.open(bg_path).convert('RGB')
        bg_w, bg_h = bg_img.size
        bg_t       = self.sd_transform(bg_img)

        ## jiaxuan
        fg_img_list, fg_img = self.generate_multifg(fg_list_path, fgmask_list_path)

        mask       = Image.fromarray(bbox2mask(bbox, bg_w, bg_h))
        mask_t     = self.mask_transform(mask)
        mask_t     = torch.where(mask_t > 0.5, 1, 0).float()
        inpaint_t  = bg_t * (1 - mask_t)
        bbox_t     = get_bbox_tensor(bbox, bg_w, bg_h)

        return {"bg_img":  inpaint_t.unsqueeze(0),
                "bg_mask": mask_t.unsqueeze(0),
                "fg_img":  fg_img,
                "fg_img_list": fg_img_list,
                "bbox":    bbox_t.unsqueeze(0)}

    def prepare_input(self, batch, shape, num_samples):
        if num_samples > 1:
            for k in batch.keys():
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = torch.cat([batch[k]] * num_samples, dim=0)

        test_model_kwargs={}
        bg_img    = batch['bg_img'].to(self.device)
        bg_latent = self.model.encode_first_stage(bg_img)
        bg_latent = self.model.get_first_stage_encoding(bg_latent).detach()
        test_model_kwargs['bg_latent'] = bg_latent
        rs_mask = F.interpolate(batch['bg_mask'].to(self.device), shape[-2:])
        rs_mask = torch.where(rs_mask > 0.5, 1.0, 0.0)
        test_model_kwargs['bg_mask']  = rs_mask
        test_model_kwargs['bbox']  = batch['bbox'].to(self.device)

        condition_list = []
        for fg_img in batch['fg_img_list']:
            fg_img = fg_img.to(self.device)
            condition = self.model.get_learned_conditioning(fg_img)
            condition = self.model.proj_out(condition)
            condition_list.append(condition)
        c = torch.cat(condition_list, dim=1)
        c = torch.cat([c] * num_samples, dim=0)
        uc = self.model.learnable_vector.repeat(c.shape[0], c.shape[1], 1)  # 1,1,768
        return test_model_kwargs, c, uc

    def inputs_preprocess(self, background_image, fg_list_path, fgmask_list_path, bbox, num_samples):

        batch = self.generate_image_batch(background_image, fg_list_path, fgmask_list_path, bbox)
        test_kwargs, c, uc = self.prepare_input(batch, self.latent_shape, num_samples)
        show_fg_img = batch["fg_img"]

        return test_kwargs, c, uc, show_fg_img


    def outputs_postprocess(self, outputs):
        x_samples_ddim = self.model.decode_first_stage(outputs[:,:4]).cpu().float()
        comp_img = tensor2numpy(x_samples_ddim, image_size=self.image_size)
        if len(comp_img) == 1:
            return comp_img[0]
        return comp_img

    @torch.no_grad()
    def __call__(self, background_image, foreground_image, foreground_mask, bbox,
                 num_samples=1, sample_steps=50, guidance_scale=5, seed=321):
        """
        Controllable image composition based on diffusion model.

        Args:
            background_image (str | numpy.ndarray): The path to background image or the background image in ndarray form.
            foreground_image (str | numpy.ndarray): The path to the list of foreground images or the foreground images in ndarray form.
            foreground_mask (None | str | numpy.ndarray): Mask of foreground image which indicates the foreground object region in the foreground image.
            bbox (list): The bounding box which indicates the foreground's location in the background. [x1, y1, x2, y2].
            num_samples (int): Number of images to be generated. default: 1.
            sample_steps (int): Number of denoising steps. The recommended setting is 25 for PLMS sampler and 50 for DDIM sampler. default: 50.
            guidance_scale (int): Scale in classifier-free guidance (minimum: 1; maximum: 20). default: 5.
            seed (int): Random Seed is used to reproduce results and same seed will lead to same results.

        Returns:
            composite_images (numpy.ndarray): Generated images with a shape of 512x512x3 or Nx512x512x3, where N indicates the number of generated images.


        """
        seed_everything(seed)
        test_kwargs, c, uc, show_fg_img = self.inputs_preprocess(background_image, foreground_image,
                                                    foreground_mask, bbox, num_samples)

        start_code = torch.randn([num_samples]+self.latent_shape, device=self.device)
        outputs, _ = self.sampler.sample(S=sample_steps,
                                    conditioning=c,
                                    batch_size=num_samples,
                                    shape=self.latent_shape,
                                    verbose=False,
                                    eta=0.0,
                                    x_T=start_code,
                                    unconditional_guidance_scale=guidance_scale,
                                    unconditional_conditioning=uc,
                                    test_model_kwargs=test_kwargs)
        comp_img   = self.outputs_postprocess(outputs)
        return comp_img, show_fg_img


def resize_with_pad(image, target_size, pad_color=(0, 0, 0)):
    h, w = image.shape[:2]
    scale = float(target_size) / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top_pad = (target_size - new_h) // 2
    bottom_pad = target_size - new_h - top_pad
    left_pad = (target_size - new_w) // 2
    right_pad = target_size - new_w - left_pad
    padded_image = cv2.copyMakeBorder(
        resized,
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )
    return padded_image

def normalize_bbox(bbox, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    return [
        x1 / float(w),
        y1 / float(h),
        x2 / float(w),
        y2 / float(h)
    ]

# ============================
# 自定義資料集 (與之前相同)
# ============================
class CustomObjectDataset(Dataset):
    def __init__(self, root_dir, target_size=512, transform=None):
        self.root_dir = root_dir
        self.target_size = target_size
        self.transform = transform
        self.samples = []
        categories = os.listdir(root_dir)
        for cat in categories:
            cat_path = os.path.join(root_dir, cat)
            if not os.path.isdir(cat_path):
                continue
            bg_path = os.path.join(cat_path, 'bg')
            if not os.path.exists(bg_path):
                continue
            bg_files = glob.glob(os.path.join(bg_path, '*.jpg')) + \
                       glob.glob(os.path.join(bg_path, '*.png'))
            fg_folders = [d for d in os.listdir(cat_path) if d.startswith('fg')]
            for fg_folder in fg_folders:
                fg_folder_path = os.path.join(cat_path, fg_folder)
                if not os.path.isdir(fg_folder_path):
                    continue
                jpg_files = sorted(glob.glob(os.path.join(fg_folder_path, '*.jpg')))
                png_files = sorted(glob.glob(os.path.join(fg_folder_path, '*.png')))
                for jf, pf in zip(jpg_files, png_files):
                    for bgf in bg_files:
                        bbox = [0.3, 0.3, 0.7, 0.7]  # 固定示範
                        self.samples.append((bgf, jf, pf, bbox))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        bg_path, fg_path, mask_path, bbox = self.samples[idx]
        bg_img = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        fg_img = cv2.imread(fg_path, cv2.IMREAD_COLOR)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        bg_resized = resize_with_pad(bg_img, self.target_size)
        fg_resized = resize_with_pad(fg_img, self.target_size)
        mask_resized = resize_with_pad(mask_img, self.target_size)
        norm_bbox = normalize_bbox(bbox, bg_resized.shape)
        bg_tensor = torch.from_numpy(bg_resized).float().permute(2, 0, 1) / 255.0
        fg_tensor = torch.from_numpy(fg_resized).float().permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0) / 255.0
        return bg_tensor, fg_tensor, mask_tensor, norm_bbox

# ============================
# 替換為 LatentDiffusion 的 LoRA 注入
# ============================
from lora_diffusion import inject_trainable_lora
from libcom.objectstitch import Mure_ObjectStitchModel

class LoRA_ObjectStitch(Mure_ObjectStitchModel):
    def __init__(self, device='cuda:0', lora_rank=4, **kwargs):
        super().__init__(device=device, **kwargs)
        self.num_timesteps = 1000
        # 如果檢查後發現 U-Net 其實存放在 self.model.model
        self.diffusion_model = self.model.model  # <--- 假設實際名稱是這個
        inject_trainable_lora(self.diffusion_model, r=lora_rank)
        self._freeze_base_model()

    def _freeze_base_model(self):
        for param in self.diffusion_model.parameters():
            param.requires_grad = False
        for module in self.diffusion_model.modules():
            if hasattr(module, "lora_layer"):
                for param in module.lora_layer.parameters():
                    param.requires_grad = True

    def compute_training_loss(self, background_image, foreground_image, foreground_mask, bbox):
        # 假設 self.num_timesteps 已在 __init__ 中定義, 如 self.num_timesteps=1000
        batch_size = foreground_image.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=foreground_image.device).long()

        # 將單通道遮罩複製成 3 通道
        mask_3ch = foreground_mask.repeat(1, 3, 1, 1)  # shape=(B,3,H,W)

        # 將背景(3ch)、前景(3ch)及複製後遮罩(3ch)串接
        x = torch.cat([background_image, foreground_image, mask_3ch], dim=1)  # shape=(B,9,H,W)

        # 通道滿足9後再 forward
        pred_noise = self.diffusion_model(x, t)

        loss = ((pred_noise - foreground_image) ** 2).mean()
        return loss

def train_lora_objectstitch(
    root_dir,
    batch_size=1,
    epochs=10,
    lr=1e-4,
    lora_rank=8,
    sampler='ddim',
    device='cuda:0'
):
    dataset = CustomObjectDataset(root_dir=root_dir, target_size=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 初始化含 LoRA 的模型
    model = LoRA_ObjectStitch(device=device, lora_rank=lora_rank, sampler=sampler)
    # 這裡的 optimizer 要抓的是 diffusion_model
    optimizer = AdamW(model.diffusion_model.parameters(), lr=lr)
    model.model.to(device)

    for epoch in range(epochs):
        for i, (bg, fg, mask, bbox) in enumerate(dataloader):
            bg = bg.to(device)
            fg = fg.to(device)
            mask = mask.to(device)
            loss = model.compute_training_loss(
                background_image=bg,
                foreground_image=fg,
                foreground_mask=mask,
                bbox=bbox
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch + 1}/{epochs}, Loss = {loss.item()}")
    return model

def inference_lora_objectstitch(
    model_path,
    background_path,
    foreground_paths,
    mask_paths,
    bbox,
    output_dir="output",
    num_samples=3,
    sample_steps=25,
    device='cuda:0'
):
    model = LoRA_ObjectStitch(device=device, lora_rank=8)
    # 載入 LoRA 權重到 diffusion_model
    lora_weights = torch.load(model_path, map_location=device)
    model.diffusion_model.load_state_dict(lora_weights, strict=False)

    bg_img = cv2.imread(background_path, cv2.IMREAD_COLOR)
    fg_imgs = [cv2.imread(fg, cv2.IMREAD_COLOR) for fg in foreground_paths]
    mask_imgs = [cv2.imread(m, cv2.IMREAD_GRAYSCALE) for m in mask_paths]

    # 使用 model(...) 執行推理，實際需依 Mure_ObjectStitchModel 流程
    composite_images = model(
        background_image=bg_img,
        foreground_image=fg_imgs,
        foreground_mask=mask_imgs,
        bbox=bbox,
        num_samples=num_samples,
        sample_steps=sample_steps
    )

    os.makedirs(output_dir, exist_ok=True)
    for i, cimg in enumerate(composite_images):
        out_path = os.path.join(output_dir, f"composite_{i}.png")
        cv2.imwrite(out_path, cimg)
    return composite_images

if __name__ == "__main__":
    trained_model = train_lora_objectstitch(
        root_dir="MureCom",
        batch_size=1,
        epochs=5,
        lr=1e-4,
        lora_rank=8,
        sampler='ddim',
        device='cuda:0'
    )
    # 假設完成後儲存
    torch.save(trained_model.diffusion_model.state_dict(), "lora_weights.bin")

    inference_lora_objectstitch(
        model_path="lora_weights.bin",
        background_path="MureCom/Airplane/bg/airplane_bg1.jpg",
        foreground_paths=["MureCom/Airplane/fg1/0.jpg", "MureCom/Airplane/fg1/1.jpg"],
        mask_paths=["MureCom/Airplane/fg1/0.png", "MureCom/Airplane/fg1/1.png"],
        bbox=[0.3, 0.3, 0.7, 0.7],
        output_dir="output",
        num_samples=3,
        sample_steps=25,
        device='cuda:0'
    )

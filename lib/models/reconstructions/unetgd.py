import os
import math
import copy
from random import random
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm

import torch
from torch import nn,einsum
import torch.nn.functional as F

from einops import rearrange, reduce
import numpy as np

from ..initializer import initialize_from_cfg

# constants
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helper functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def l2norm(t):
    return F.normalize(t, dim = -1)

# helper modules
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, scale = 10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# network
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion funcitons
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# diffusion building
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0.,
        # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.,

        use_input_norm = True,      # [0, 1] <-> [-1, 1]
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        self.use_input_norm = use_input_norm

    # normalization functions
    def normalize_to_neg_one_to_one(self, img):
        if self.use_input_norm:
            return img * 2 - 1
        else:
            return img

    def unnormalize_to_zero_to_one(self, t):
        if self.use_input_norm:
            return (t + 1) * 0.5
        else:
            return t

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, model=None, x_self_cond = None, clip_x_start = False):
        model = self.model if model is None else model
        model_output = model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, model=None, x_self_cond = None, clip_denoised = False):
        preds = self.model_predictions(x, t, model, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, model=None, x_self_cond=None, clip_denoised=False, mask=None, gt=None):
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        if mask is not None:
            weighted_gt = self.q_sample(x_start=gt, t=batched_times)
            x = mask * weighted_gt + (1 - mask) * x
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x = x, t = batched_times, model=model, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, model=None, img=None, total_time=None, mask=None, gt=None,
                      vis_process_step=None, **kwargs):
        model = self.model if model is None else model
        batch, device = shape[0], self.betas.device

        if img is None:
            img = torch.randn(shape, device=device)
        if total_time is None:
            total_time = self.num_timesteps

        x_start = None

        pbar = tqdm(reversed(range(0, total_time)), desc = 'sampling loop time step', total = total_time)

        ret_list = []
        for t in pbar:
            # diffusion process visualize
            if vis_process_step is not None:
                if t + 1 in vis_process_step:
                    ret_list.append(self.unnormalize_to_zero_to_one(img))
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, model=model, x_self_cond=self_cond, mask=mask, gt=gt)

        img = self.unnormalize_to_zero_to_one(img)
        ret_list.insert(0, img)
        # img or tuple of imgs
        return tuple(ret_list)

    @torch.no_grad()
    def ddim_sample(self, shape, model=None, img=None, total_time=None, clip_denoised=False,
                    mask=None, gt=None, **kwargs):
        model = self.model if model is None else model
        if img is None:
            img = torch.randn(shape, device=device)
        if total_time is None:
            total_time = self.num_timesteps
        batch, device, sampling_timesteps, eta = shape[0], self.betas.device, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_time - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):

            if mask is not None:
                mask_q_t = torch.zeros((img.shape[0],), dtype=torch.long, device=img.device) + time
                weighted_gt = self.q_sample(x_start=gt, t=mask_q_t)
                img = mask * weighted_gt + (1 - mask) * img

            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, model=model, x_self_cond=self_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = self.unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, model=None, batch_size = 16, img_size = None, img=None, total_time=None,
                mask=None ,gt=None, vis_process_step=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if img is None:
            print('diffusion random generating..')
            channels = self.channels
            img_size = 32 if img_size is None else img_size
            return sample_fn((batch_size, channels, image_size, image_size), model=model,
                            mask=mask, gt=gt, vis_process_step=vis_process_step)
        else:
            return sample_fn(img.shape, model=model, img=img, total_time=total_time,
                            mask=mask, gt=gt, vis_process_step=vis_process_step)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise = None):
        '''
        x_start has already normalized outside,
        x_start(target) and pred_x0(model_out) are both at normalized scale.
        '''
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def inference(self, img, model=None, total_time=None, img_normalized=False, noise=None,
                mask=None, vis_process_step=None):
        model = self.model if model is None else model
        if total_time is None:
            total_time = self.num_timesteps
        q_time = torch.zeros((img.shape[0],), dtype=torch.long, device=img.device) + \
            (total_time if total_time < self.num_timesteps else self.num_timesteps - 1)
        if img_normalized:
            x_start = img
        else:
            x_start = self.normalize_to_neg_one_to_one(img)
        noise = default(noise, lambda:torch.randn_like(x_start))
        x_t = self.q_sample(x_start=x_start, t=q_time, noise=noise)
        pred = self.sample(model, img=x_t, total_time=total_time,
                        mask=mask, gt=x_start,
                        vis_process_step=vis_process_step)

        if self.objective == 'pred_noise':
            pred_x_0 = self.predict_start_from_noise(x_t, total_time, pred)
        elif self.objective == 'pred_x0':
            pred_x_0 = pred
        return pred_x_0

    def forward(self, img, model = None, infer_time=None,
                mask=None, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        if self.training:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            img = self.normalize_to_neg_one_to_one(img)
            return self.p_losses(img, t, *args, **kwargs)
        else:
            return self.inference(img, model, total_time=infer_time,
                                mask=mask, *args, **kwargs)

# functions and submodels for model
def update_ema_params(target, source, decay_rate=0.9999):
    targParams = dict(target.named_parameters())
    srcParams = dict(source.named_parameters())
    for k in targParams:
        targParams[k].data.mul_(decay_rate).add_(srcParams[k].data, alpha=1 - decay_rate)

class norm_helper(object):
    def __init__(self, train_norm, train_std):
        self.train_norm = train_norm
        self.train_std = train_std
        self.test_norm = 0
        self.test_std = 1

    def normalize(self, inp):
        '''inp is [N, C, H, W]'''
        if self.train_norm is None:
            return inp

        if isinstance(self.train_norm, float):
            self.test_norm = inp.contiguous().detach().view(inp.shape[0], -1).mean(dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            self.test_std = inp.contiguous().detach().view(inp.shape[0], -1).std(dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        else:
            self.test_norm = inp.contiguous().detach().view(*inp.shape[: 2], -1).mean(dim=-1, keepdim=True).unsqueeze(-1)
            self.test_std = inp.contiguous().detach().view(*inp.shape[: 2], -1).std(dim=-1, keepdim=True).unsqueeze(-1) + 1e-5
            if isinstance(self.train_norm, list):
                self.train_norm = torch.tensor(self.train_norm, device=inp.device, dtype=inp.dtype).view(1, len(self.train_norm), 1, 1)
                self.train_std = torch.tensor(self.train_std, device=inp.device, dtype=inp.dtype).view(1, len(self.train_std), 1, 1)
        out = (inp - self.test_norm) / self.test_std
        out = out * self.train_std + self.train_norm
        return out

    def unnormalize(self, inp):
        '''inp is [N, C, H, W]'''
        if self.train_norm is None:
            return inp

        out = (inp - self.train_norm) / self.train_std
        out = out * self.test_std + self.test_norm
        return out

# model
class UNetGD(nn.Module):
    def __init__(
        self,
        inplanes,
        instrides,
        dim,

        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        learned_sinusoidal_dim = 16,

        image_size = 32,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l2',
        objective = 'pred_x0',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0.,
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.,
        use_input_norm=True,

        infer_time=500,
        infer_use_ema=False,

        ema_update_every = 10,
        ema_decay = 0.995,

        save_recon=None,
        initializer=None,

        train_area_norm=None,
        train_area_std=None,

        vis_process_step=None,
    ):
        super().__init__()
        assert isinstance(inplanes, list) and len(inplanes) == 1
        assert isinstance(instrides, list) and len(instrides) == 1

        channels = inplanes[0]
        self.unet_model = Unet(
            dim,
            init_dim,
            out_dim,
            dim_mults,
            channels,
            self_condition,
            resnet_block_groups,
            learned_variance,
            learned_sinusoidal_cond,
            learned_sinusoidal_dim )
        self.diffusion_model = GaussianDiffusion(
            self.unet_model,
            timesteps = timesteps,
            image_size=image_size,
            sampling_timesteps = sampling_timesteps,
            loss_type = loss_type,
            objective = objective,
            beta_schedule = beta_schedule,
            p2_loss_weight_gamma = p2_loss_weight_gamma,
            p2_loss_weight_k = p2_loss_weight_k,
            ddim_sampling_eta = ddim_sampling_eta,
            use_input_norm=use_input_norm)
        self.ema = copy.deepcopy(self.unet_model)
        self.ema_update_every = ema_update_every
        self.ema_decay = ema_decay
        self.ema_iter = 0
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=instrides[0])

        self.infer_time = infer_time
        self.infer_use_ema = infer_use_ema

        self.train_area_norm = train_area_norm
        self.train_area_std = train_area_std

        self.vis_process_step = vis_process_step if vis_process_step is None \
                else sorted(vis_process_step, reverse=True)

        self.save_recon = save_recon
        initialize_from_cfg(self, initializer)

    def forward(self, inp, mask=None):
        feature_align = inp['feature_align']
        if self.training:
            output = self.diffusion_model(feature_align)
            ret = {'loss': output}
        else:
            if self.infer_use_ema:
                infer_model = self.ema
                infer_model.eval()
            else:
                infer_model = None
            with torch.no_grad():
                norm_tool = norm_helper(self.train_area_norm, self.train_area_std)
                feat_inp = norm_tool.normalize(feature_align)
                feat_out = self.diffusion_model(feat_inp, model=infer_model,
                                    infer_time=self.infer_time, mask=mask,
                                    vis_process_step=self.vis_process_step)
                # visualize process
                if self.vis_process_step is not None:
                    feat_vis_list = [norm_tool.unnormalize(f_tmp) for f_tmp in feat_out[1: ]]
                    feat_out = feat_out[0]
                else:
                    feat_vis_list = None
                output = norm_tool.unnormalize(feat_out)
            # B, 1, H, W
            pred = torch.sqrt(
                torch.sum((feature_align - output) ** 2, dim=1, keepdim=True))
            # B x 1 x H_img x W_img
            pred = self.upsample(pred)
            ret = {
                'feature_rec': output,
                'feature_align': feature_align,
                'pred': pred,
            }
        # save npy
        if not self.training and self.save_recon:
            if feat_vis_list is not None:
                save_list = [output] + feat_vis_list
                save_name_list = ['output'] + ['step_%d_with_noise' % s_tmp for s_tmp in self.vis_process_step]
            else:
                save_list = [output]
                save_name_list = ['']
            for save_rec, save_rec_name in zip(save_list, save_name_list):
                clsnames = inp["clsname"]
                filenames = inp["filename"]
                for clsname, filename, feat_rec in zip(clsnames, filenames, save_rec):
                    filedir, filename = os.path.split(filename)
                    _, defename = os.path.split(filedir)
                    filename_, _ = os.path.splitext(filename)
                    save_dir = os.path.join(self.save_recon.save_dir,
                                            self.__class__.__name__, save_rec_name,
                                            clsname, defename)
                    os.makedirs(save_dir, exist_ok=True)
                    feature_rec_np = feat_rec.detach().cpu().numpy()
                    np.save(os.path.join(save_dir, filename_ + ".npy"), feature_rec_np)

        return ret

    def ema_update(self):
        self.ema_iter += 1
        if self.ema_iter % self.ema_update_every == 0:
            update_ema_params(self.ema, self.unet_model, decay_rate=self.ema_decay)

    def parameters(self, recurse: bool = True):
        return self.unet_model.parameters(recurse)
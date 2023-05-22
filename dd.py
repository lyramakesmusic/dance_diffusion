import torch
import torch.nn as nn
import torchaudio
from torch.utils import data
from einops import rearrange
from scipy.io.wavfile import write
from audio_diffusion.models import DiffusionAttnUnet1D
from audio_diffusion.utils import Stereo, PadCrop
from copy import deepcopy
from glob import glob
import os
import math
import k_diffusion as K

class dd:
    def __init__(self, ckpt_path, sample_size=131072, sample_rate=44100):
        super().__init__()
        dd_args = self._get_dd_args(sample_size, sample_rate)
        self.model = self._load_model(dd_args, ckpt_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        del self.model.diffusion
        self.model_fn = self.model.diffusion_ema

    def _get_dd_args(self, sample_size, sample_rate):
        dd_args = lambda _: None
        setattr(dd_args, "sample_size", sample_size)
        setattr(dd_args, "sample_rate", sample_rate)
        setattr(dd_args, "latent_dim", 0)
        return dd_args

    def _load_model(self, dd_args, ckpt_path):
        model = DiffusionAttnUnet1D(dd_args, n_attn_layers=4)
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])
        return model.requires_grad_(False).to(self.device)

    def load_to_device(self, path, sr):
        audio, file_sr = torchaudio.load(path)
        if sr != file_sr:
            audio = torchaudio.transforms.Resample(file_sr, sr)(audio)
        audio = audio.to(self.device)
        return audio

    def save_np_to_wav(self, folder, data, sample_rate, name):
        gen_count = len(os.listdir(f'{folder}'))
        fpath = f'{folder}/{name}{gen_count}.wav'
        write(fpath, sample_rate, data)
        return fpath

    def dd_sample(self, noise, steps=100, sampler_type="v-iplms", sigma_min=0.0001, sigma_max=1, rho=7.0, rtol=0.01, atol=0.01):
        denoiser = K.external.VDenoiser(self.model_fn)
        sigmas = K.sampling.get_sigmas_karras(
            steps, sigma_min, sigma_max, rho, device=self.device
        )
        t = torch.linspace(1, 0, steps + 1, device=self.device)[:-1]
        step_list = self.get_crash_schedule(t)
        return K.sampling.iplms_sample(
            self.model_fn, noise, step_list, {}
        )

    def get_crash_schedule(self, t):
        sigma = torch.sin(t * math.pi / 2) ** 2
        alpha = (1 - sigma ** 2) ** 0.5
        return self.alpha_sigma_to_t(alpha, sigma)

    def alpha_sigma_to_t(self, alpha, sigma):
        return torch.atan2(sigma, alpha) / math.pi * 2

    def gen(self, output_folder, batch_size, sample_size, steps):
        noise = torch.randn([batch_size, 2, sample_size]).to(self.device)
        generated = self.dd_sample(self.model_fn, noise, steps)
        generated = generated.clamp(-1, 1)
        save_paths = [
            self.save_np_to_wav(output_folder, gen_sample.cpu().numpy().T, sample_size, "gen")
            for gen_sample in generated
        ]
        return save_paths

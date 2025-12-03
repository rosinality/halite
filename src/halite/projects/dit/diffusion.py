import torch
from torch import nn
from torch.nn import functional as F


class Diffusion(nn.Module):
    def __init__(
        self,
        input_shape,
        n_labels,
        steps=50,
        method="heun",
        guidance_interval=(0.0, 1.0),
        guidance_scale=1.0,
        label_drop_p=0.1,
        p_mean=-0.8,
        p_std=0.8,
        noise_scale=1.0,
        eps_t=5e-2,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.n_labels = n_labels
        self.steps = steps
        self.method = method
        self.guidance_interval = guidance_interval
        self.guidance_scale = guidance_scale
        self.label_drop_p = label_drop_p
        self.p_mean = p_mean
        self.p_std = p_std
        self.noise_scale = noise_scale
        self.eps_t = eps_t

    def sample_t(self, batch, device=None):
        z = torch.randn(batch, device=device) * self.p_std + self.p_mean

        return torch.sigmoid(z)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_p
        out = torch.where(drop, torch.full_like(labels, self.n_labels), labels)
        return out

    def loss(self, fn, input, labels=None, **forward_kwargs):
        if labels is not None and self.training:
            labels = self.drop_labels(labels)

        t = self.sample_t(input.shape[0], device=input.device).view(
            -1, *([1] * (input.ndim - 1))
        )
        e = torch.randn_like(input) * self.noise_scale

        z = t * input + (1 - t) * e
        v = (input - z) / (1 - t).clamp_min(self.eps_t)

        x_pred = fn(z, t.flatten(), labels, **forward_kwargs).to(torch.float32)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.eps_t)

        loss = F.mse_loss(v_pred, v)

        return loss

    @torch.no_grad()
    def generate(self, fn, labels, seed=None, **forward_kwargs):
        device = labels.device
        batch = labels.shape[0]

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        z = self.noise_scale * torch.randn(
            batch, *self.input_shape, generator=generator, device=device
        )
        timesteps = (
            torch.linspace(0, 1, self.steps + 1, device=device)
            .view(-1, *([1] * z.ndim))
            .expand(-1, batch, -1, -1, -1)
        )

        if self.method == "euler":
            step_fn = self._euler_step

        elif self.method == "heun":
            step_fn = self._heun_step

        else:
            raise NotImplementedError(f"Method {self.method} not implemented")

        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = step_fn(fn, z, t, t_next, labels, **forward_kwargs)

        z = self._euler_step(fn, z, timesteps[-2], timesteps[-1], labels)

        return z

    def _forward_fn(self, fn, z, t, labels, **forward_kwargs):
        x_cond = fn(z, t.flatten(), labels, **forward_kwargs).to(torch.float32)
        v_cond = (x_cond - z) / (1 - t).clamp_min(self.eps_t)

        x_uncond = fn(z, t.flatten(), torch.full_like(labels, self.n_labels))
        v_uncond = (x_uncond - z) / (1 - t).clamp_min(self.eps_t)

        low, high = self.guidance_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        guidance_interval = torch.where(interval_mask, self.guidance_scale, 1.0)

        return v_uncond + guidance_interval * (v_cond - v_uncond)

    def _euler_step(self, fn, z, t, t_next, labels, **forward_kwargs):
        v_pred = self._forward_fn(fn, z, t, labels, **forward_kwargs)
        z_next = z + (t_next - t) * v_pred

        return z_next

    def _heun_step(self, fn, z, t, t_next, labels, **forward_kwargs):
        v_pred_t = self._forward_fn(fn, z, t, labels, **forward_kwargs)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_fn(
            fn, z_next_euler, t_next, labels, **forward_kwargs
        )

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred

        return z_next


class EquilibriumMatchingJiT(Diffusion):
    def get_c_t(self, t):
        interp = 0.8
        start = 1.0
        c_t = (
            torch.minimum(
                start - (start - 1) / (interp) * t,
                1 / (1 - interp) - 1 / (1 - interp) * t,
            )
            * 4
        )

        return c_t

    def loss(self, fn, input, labels=None, **forward_kwargs):
        if labels is not None and self.training:
            labels = self.drop_labels(labels)

        t = self.sample_t(input.shape[0], device=input.device).view(
            -1, *([1] * (input.ndim - 1))
        )
        e = torch.randn_like(input) * self.noise_scale

        c_t = self.get_c_t(t)

        z = t * input + (1 - t) * e
        v = c_t * (input - z) / (1 - t).clamp_min(self.eps_t)

        x_pred = fn(z, t.flatten() * 0, labels, **forward_kwargs).to(torch.float32)
        v_pred = (x_pred - c_t * z) / (1 - t).clamp_min(self.eps_t)

        loss = F.mse_loss(v_pred, v)

        return loss

    def _forward_fn(self, fn, z, t, labels, **forward_kwargs):
        c_t = self.get_c_t(t)

        x_cond = fn(z, t.flatten() * 0, labels, **forward_kwargs).to(torch.float32)
        v_cond = (x_cond - c_t * z) / (1 - t).clamp_min(self.eps_t)

        x_uncond = fn(z, t.flatten() * 0, torch.full_like(labels, self.n_labels))
        v_uncond = (x_uncond - c_t * z) / (1 - t).clamp_min(self.eps_t)

        low, high = self.guidance_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        guidance_interval = torch.where(interval_mask, self.guidance_scale, 1.0)

        return v_uncond + guidance_interval * (v_cond - v_uncond)

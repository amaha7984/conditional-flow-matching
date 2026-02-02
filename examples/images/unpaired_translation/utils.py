import torch
from torchvision.utils import save_image
from torchdiffeq import odeint


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1.0 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x in iter(dataloader):
            yield x


@torch.no_grad()
def translate_tensor_ode(
    model,
    x0,
    steps=50,
    method="dopri5",
    tol=1e-5,
):
    """
    Integrate dx/dt = v_theta(t, x) from t=0->1 starting at x0.
    Works for both pixel tensors and latent tensors.
    model signature: model(t, x) -> dx/dt
    """
    model.eval()

    if method == "euler":
        t_grid = torch.linspace(0, 1, steps + 1, device=x0.device, dtype=x0.dtype)
        x = x0
        dt = 1.0 / float(steps)
        for k in range(steps):
            t = t_grid[k]
            dx = model(t, x)
            x = x + dt * dx
        return x

    t_span = torch.linspace(0, 1, 2, device=x0.device, dtype=x0.dtype)
    xT = odeint(model, x0, t_span, rtol=tol, atol=tol, method=method)[-1]
    return xT


@torch.no_grad()
def decode_latents_to_images(vae, z, scale_factor):
    """
    z is assumed scaled by scale_factor during training.
    To decode, invert scaling.
    Returns images in [-1,1].
    """
    z_in = z / scale_factor
    x = vae.decode(z_in).sample
    return x


@torch.no_grad()
def translate_and_save_grid(
    model,
    vae,
    xA,
    out_path,
    scale_factor=0.18215,
    ode_steps=50,
    ode_method="dopri5",
    ode_tol=1e-5,
):
    """
    Saves a grid: first row = inputs (A), second row = translated outputs (fake B).

    - If vae is NOT None: run latent pipeline
        xA -> encode -> zA -> ODE -> zB_hat -> decode -> xB_hat
    - If vae is None: run pixel pipeline
        xA -> ODE -> xB_hat

    xA is in [-1,1].
    """
    model.eval()

    if vae is not None:
        vae.eval()
        zA = vae.encode(xA).latent_dist.mean * scale_factor
        zB_hat = translate_tensor_ode(model, zA, steps=ode_steps, method=ode_method, tol=ode_tol)
        xB_hat = decode_latents_to_images(vae, zB_hat, scale_factor)
    else:
        # pixel space translation directly
        xB_hat = translate_tensor_ode(model, xA, steps=ode_steps, method=ode_method, tol=ode_tol)

    # Convert to [0,1] for saving
    xA_vis = (xA.clamp(-1, 1) + 1) / 2.0
    xB_vis = (xB_hat.clamp(-1, 1) + 1) / 2.0

    grid = torch.cat([xA_vis, xB_vis], dim=0)
    save_image(grid, out_path, nrow=xA.shape[0])

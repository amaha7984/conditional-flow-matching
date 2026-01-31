import copy
import os
import glob

import torch
from absl import app, flags
from torchvision import transforms
from PIL import Image
from tqdm import trange

from diffusers.models import AutoencoderKL

from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModelWrapper

from utils import ema, infiniteloop, translate_and_save_grid


FLAGS = flags.FLAGS

# -------------------------
# Data
# -------------------------
flags.DEFINE_string(
    "data_root",
    "/aul/homes/amaha038/InnovativeMapSynthesis/pytorch-CycleGAN-and-pix2pix/datasets/horse2zebra",
    help="dataset root containing trainA/trainB/testA/testB",
)
flags.DEFINE_integer("image_size", 256, help="input image size")
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_integer("batch_size", 16, help="batch size (OT cost is O(B^2))")

# -------------------------
# Mode: latent vs pixel
# -------------------------
flags.DEFINE_bool(
    "latent",
    False,
    help="run in VAE latent space (same as current code). If neither --latent nor --pixel is set, defaults to latent.",
)
flags.DEFINE_bool(
    "pixel",
    False,
    help="run directly in pixel space (x in [-1,1], 3xHxW).",
)

# -------------------------
# Model/Training
# -------------------------
flags.DEFINE_float("lr", 2e-4, help="target learning rate")
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer("total_steps", 200001, help="total training steps")
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")

# -------------------------
# OT-CFM
# -------------------------
flags.DEFINE_float("sigma", 0.0, help="sigma for probability path (0.0 = deterministic path)")
flags.DEFINE_bool("ot_replace", True, help="sample OT pairs with replacement (official default True)")

# -------------------------
# VAE (only used for latent)
# -------------------------
flags.DEFINE_string("vae_name", "stabilityai/sd-vae-ft-mse", help="diffusers VAE id")
flags.DEFINE_float("scale_factor", 0.18215, help="latent scale factor")

# -------------------------
# UNet (latent defaults)
# -------------------------
flags.DEFINE_integer("num_channel", 192, help="base channel of UNet")
flags.DEFINE_integer("num_res_blocks", 3, help="res blocks per level")
flags.DEFINE_string("attention_resolutions", "16,8,4", help="attention resolutions (latent defaults)")

# -------------------------
# UNet (pixel overrides; only used when --pixel)
# Keep these defaults conservative to avoid exploding compute.
# You can edit these later if needed.
# -------------------------
flags.DEFINE_integer("pixel_num_channel", 64, help="base channel of UNet for pixel space")
flags.DEFINE_integer("pixel_num_res_blocks", 2, help="res blocks per level for pixel space")
flags.DEFINE_string("pixel_attention_resolutions", "32,16,8", help="attention resolutions for pixel space")

# -------------------------
# Saving/Eval
# -------------------------
flags.DEFINE_string("output_dir", "./results_horse2zebra_vae/", help="output_directory")
flags.DEFINE_integer("save_step", 10000, help="frequency of saving checkpoints")
flags.DEFINE_integer("sample_n", 8, help="how many testA images to translate for preview grids")


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / float(FLAGS.warmup)


class ImageFolderNoLabel(torch.utils.data.Dataset):
    """
    Minimal folder dataset that returns only an image tensor.
    Expects a folder containing images (jpg/png/etc).
    """

    def __init__(self, folder, transform):
        self.folder = folder
        self.transform = transform
        exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(folder, e)))
        files = sorted(files)
        if len(files) == 0:
            raise RuntimeError(f"No images found in: {folder}")
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)


def build_dataloaders():
    tfm = transforms.Compose(
        [
            transforms.Resize(FLAGS.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(FLAGS.image_size),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1,1]
        ]
    )

    trainA = ImageFolderNoLabel(os.path.join(FLAGS.data_root, "trainA"), tfm)
    trainB = ImageFolderNoLabel(os.path.join(FLAGS.data_root, "trainB"), tfm)
    testA = ImageFolderNoLabel(os.path.join(FLAGS.data_root, "testA"), tfm)
    testB = ImageFolderNoLabel(os.path.join(FLAGS.data_root, "testB"), tfm)

    loaderA = torch.utils.data.DataLoader(
        trainA,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    loaderB = torch.utils.data.DataLoader(
        trainB,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    testA_loader = torch.utils.data.DataLoader(
        testA,
        batch_size=FLAGS.sample_n,
        shuffle=False,
        num_workers=FLAGS.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    testB_loader = torch.utils.data.DataLoader(
        testB,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=FLAGS.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    return loaderA, loaderB, testA_loader, testB_loader


def _resolve_space():
    """
    Priority:
      - if --pixel: pixel
      - elif --latent: latent
      - elif neither given: latent (default, matches  latent behavior)
    """
    if FLAGS.pixel and FLAGS.latent:
        raise ValueError("Choose only one: --pixel OR --latent")
    if FLAGS.pixel:
        return "pixel"
    # default
    return "latent"


def parse_attention_resolutions(s):
    return s


def _build_model_and_optional_vae(space: str):
    """
    Returns:
      net_model, ema_model, vae_or_none
    """
    if space == "latent":
        # VAE (frozen)
        vae = AutoencoderKL.from_pretrained(FLAGS.vae_name).to(device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

        # SD-VAE latent for 256x256 -> (B,4,32,32)
        net_model = UNetModelWrapper(
            dim=(4, 32, 32),
            num_res_blocks=FLAGS.num_res_blocks,
            num_channels=FLAGS.num_channel,
            channel_mult=[1, 2, 3, 4],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions=parse_attention_resolutions(FLAGS.attention_resolutions),
            dropout=0.1,
        ).to(device)

        ema_model = copy.deepcopy(net_model)
        return net_model, ema_model, vae

    # -------------------------
    # Pixel space
    # -------------------------
    # Pixel for 256x256 -> (B,3,256,256)
    # Use smaller base channels by default to keep it feasible.
    net_model = UNetModelWrapper(
        dim=(3, FLAGS.image_size, FLAGS.image_size),
        num_res_blocks=FLAGS.pixel_num_res_blocks,
        num_channels=FLAGS.pixel_num_channel,
        channel_mult=[1, 2, 3, 4],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions=parse_attention_resolutions(FLAGS.pixel_attention_resolutions),
        dropout=0.1,
    ).to(device)

    ema_model = copy.deepcopy(net_model)
    return net_model, ema_model, None


def train(argv):
    space = _resolve_space()

    if space == "latent":
        print("Training OT-CFM (unpaired A->B) in SD-VAE LATENT space")
    else:
        print("Training OT-CFM (unpaired A->B) in PIXEL space")

    print("data_root:", FLAGS.data_root)
    print("lr, total_steps, ema_decay, save_step:", FLAGS.lr, FLAGS.total_steps, FLAGS.ema_decay, FLAGS.save_step)
    print("sigma:", FLAGS.sigma, "ot_replace:", FLAGS.ot_replace)
    if space == "latent":
        print("vae_name:", FLAGS.vae_name, "scale_factor:", FLAGS.scale_factor)

    # Data
    loaderA, loaderB, testA_loader, _ = build_dataloaders()
    loopA = infiniteloop(loaderA)
    loopB = infiniteloop(loaderB)

    # Model (+VAE if latent)
    net_model, ema_model, vae = _build_model_and_optional_vae(space)

    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    # OT-CFM matcher (official)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=FLAGS.sigma)

    if space == "latent":
        savedir = os.path.join(FLAGS.output_dir, "otcfm_latent_horse2zebra_singlegpu")
    else:
        savedir = os.path.join(FLAGS.output_dir, "otcfm_pixel_horse2zebra_singlegpu")
    os.makedirs(savedir, exist_ok=True)

    # Fixed test batch for previews
    fixed_testA = next(iter(testA_loader)).to(device)

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            net_model.train()
            optim.zero_grad(set_to_none=True)

            xA = next(loopA).to(device)  # [-1,1]
            xB = next(loopB).to(device)  # [-1,1]

            # Choose representation: latent z or pixel x
            if space == "latent":
                # Encode to latents (deterministic)
                with torch.no_grad():
                    zA = vae.encode(xA).latent_dist.mean * FLAGS.scale_factor
                    zB = vae.encode(xB).latent_dist.mean * FLAGS.scale_factor

                if FLAGS.ot_replace:
                    t, zt, ut = FM.sample_location_and_conditional_flow(zA, zB)
                else:
                    zA2, zB2 = FM.ot_sampler.sample_plan(zA, zB, replace=False)
                    t, zt, ut = FM.sample_location_and_conditional_flow(zA2, zB2)

                vt = net_model(t, zt)
                loss = torch.mean((vt - ut) ** 2)

            else:
                # Pixel space flow matching on x directly
                if FLAGS.ot_replace:
                    t, xt, ut = FM.sample_location_and_conditional_flow(xA, xB)
                else:
                    xA2, xB2 = FM.ot_sampler.sample_plan(xA, xB, replace=False)
                    t, xt, ut = FM.sample_location_and_conditional_flow(xA2, xB2)

                vt = net_model(t, xt)
                loss = torch.mean((vt - ut) ** 2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            pbar.set_postfix(loss=float(loss.item()))

            # Save previews + checkpoints
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                # normal
                translate_and_save_grid(
                    model=net_model,
                    vae=vae,  # None if pixel
                    xA=fixed_testA,
                    out_path=os.path.join(savedir, f"normal_translate_step_{step}.png"),
                    scale_factor=FLAGS.scale_factor,  # ignored if pixel
                )
                # ema
                translate_and_save_grid(
                    model=ema_model,
                    vae=vae,  # None if pixel
                    xA=fixed_testA,
                    out_path=os.path.join(savedir, f"ema_translate_step_{step}.png"),
                    scale_factor=FLAGS.scale_factor,  # ignored if pixel
                )

                ckpt = {
                    "net_model": net_model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "sched": sched.state_dict(),
                    "optim": optim.state_dict(),
                    "step": step,
                    "cfg": {
                        "space": space,
                        "data_root": FLAGS.data_root,
                        "sigma": FLAGS.sigma,
                        "ot_replace": FLAGS.ot_replace,
                    },
                }
                if space == "latent":
                    ckpt["cfg"].update(
                        {
                            "vae_name": FLAGS.vae_name,
                            "scale_factor": FLAGS.scale_factor,
                        }
                    )

                torch.save(ckpt, os.path.join(savedir, f"otcfm_{space}_horse2zebra_step_{step}.pt"))


if __name__ == "__main__":
    app.run(train)

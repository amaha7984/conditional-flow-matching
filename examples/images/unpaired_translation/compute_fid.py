import os
import sys
import glob

import torch
from absl import flags
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from diffusers.models import AutoencoderKL
from cleanfid import fid

from torchcfm.models.unet.unet import UNetModelWrapper
from utils import translate_tensor_ode, decode_latents_to_images


FLAGS = flags.FLAGS

# -------------------------
# Data / IO
# -------------------------
flags.DEFINE_string(
    "data_root",
    "/aul/homes/amaha038/InnovativeMapSynthesis/pytorch-CycleGAN-and-pix2pix/datasets/horse2zebra",
    help="dataset root containing testA/testB",
)
flags.DEFINE_string("ckpt_path", "", help="path to saved .pt checkpoint")
flags.DEFINE_string("out_dir", "./fid_runs_horse2zebra_vae/", help="where to write generated images")

# -------------------------
# Mode: latent vs pixel
# -------------------------
flags.DEFINE_bool(
    "latent",
    False,
    help="run in VAE latent space. If neither --latent nor --pixel is set, defaults to latent.",
)
flags.DEFINE_bool(
    "pixel",
    False,
    help="run directly in pixel space (x in [-1,1], 3xHxW).",
)

# -------------------------
# VAE (only used for latent)
# -------------------------
flags.DEFINE_string("vae_name", "stabilityai/sd-vae-ft-mse", help="diffusers VAE id")
flags.DEFINE_float("scale_factor", 0.18215, help="latent scale factor")

# -------------------------
# UNet (latent)
# -------------------------
flags.DEFINE_integer("num_channel", 192, help="base channel of UNet (latent)")
flags.DEFINE_integer("num_res_blocks", 3, help="res blocks per level (latent)")
flags.DEFINE_string("attention_resolutions", "16,8,4", help="attention resolutions (latent)")

# -------------------------
# UNet (pixel) - safer defaults
# -------------------------
flags.DEFINE_integer("pixel_num_channel", 64, help="base channel of UNet (pixel)")
flags.DEFINE_integer("pixel_num_res_blocks", 2, help="res blocks per level (pixel)")
flags.DEFINE_string("pixel_attention_resolutions", "32,16,8", help="attention resolutions (pixel)")

# -------------------------
# Generation / Integration
# -------------------------
flags.DEFINE_integer("image_size", 256, help="input image size")
flags.DEFINE_integer("batch_size", 32, help="batch size for generation")
flags.DEFINE_integer("num_gen", 5000, help="how many testA images to translate for FID")
flags.DEFINE_integer("integration_steps", 50, help="euler steps if method=euler")
flags.DEFINE_string("integration_method", "dopri5", help="dopri5 or euler")
flags.DEFINE_float("tol", 1e-5, help="ode solver tol")


def parse_attention_resolutions(s):
    return s


def _resolve_space():
    if FLAGS.pixel and FLAGS.latent:
        raise ValueError("Choose only one: --pixel OR --latent")
    if FLAGS.pixel:
        return "pixel"
    # default to latent
    return "latent"


class ImageFolderNoLabel(torch.utils.data.Dataset):
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


def _load_state_dict_flexible(net, state_dict):
    # Handle DataParallel "module." prefix if present
    try:
        net.load_state_dict(state_dict)
        return
    except RuntimeError:
        from collections import OrderedDict

        new_state = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state[k[len("module."):]] = v
            else:
                new_state[k] = v
        net.load_state_dict(new_state)


def main(argv):
    if FLAGS.ckpt_path == "":
        raise ValueError("Please provide --ckpt_path")

    space = _resolve_space()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # transforms for testA
    tfm = transforms.Compose(
        [
            transforms.Resize(FLAGS.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(FLAGS.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1,1]
        ]
    )

    testA_dir = os.path.join(FLAGS.data_root, "testA")
    testB_dir = os.path.join(FLAGS.data_root, "testB")
    testA = ImageFolderNoLabel(testA_dir, tfm)
    loaderA = torch.utils.data.DataLoader(
        testA,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True,
    )

    # Optional VAE for latent mode
    vae = None
    if space == "latent":
        vae = AutoencoderKL.from_pretrained(FLAGS.vae_name).to(device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

    # Model
    if space == "latent":
        net = UNetModelWrapper(
            dim=(4, 32, 32),
            num_res_blocks=FLAGS.num_res_blocks,
            num_channels=FLAGS.num_channel,
            channel_mult=[1, 2, 3, 4],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions=parse_attention_resolutions(FLAGS.attention_resolutions),
            dropout=0.1,
        ).to(device)
    else:
        net = UNetModelWrapper(
            dim=(3, FLAGS.image_size, FLAGS.image_size),
            num_res_blocks=FLAGS.pixel_num_res_blocks,
            num_channels=FLAGS.pixel_num_channel,
            channel_mult=[1, 2, 3, 4],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions=parse_attention_resolutions(FLAGS.pixel_attention_resolutions),
            dropout=0.1,
        ).to(device)

    ckpt = torch.load(FLAGS.ckpt_path, map_location=device)
    state_dict = ckpt.get("ema_model", ckpt.get("net_model", ckpt))
    _load_state_dict_flexible(net, state_dict)
    net.eval()

    # Output folder for generated images
    run_name = os.path.splitext(os.path.basename(FLAGS.ckpt_path))[0]
    gen_dir = os.path.join(FLAGS.out_dir, f"gen_{space}_{run_name}")
    os.makedirs(gen_dir, exist_ok=True)

    # Generate translated images from testA
    saved = 0
    pbar = tqdm(total=min(FLAGS.num_gen, len(testA)), desc=f"Generating fakeB for FID ({space})")
    for xA in loaderA:
        xA = xA.to(device)
        bsz = xA.shape[0]

        with torch.no_grad():
            if space == "latent":
                zA = vae.encode(xA).latent_dist.mean * FLAGS.scale_factor
                zB_hat = translate_tensor_ode(
                    model=net,
                    x0=zA,
                    steps=FLAGS.integration_steps,
                    method=FLAGS.integration_method,
                    tol=FLAGS.tol,
                )
                xB_hat = decode_latents_to_images(vae, zB_hat, FLAGS.scale_factor)
            else:
                xB_hat = translate_tensor_ode(
                    model=net,
                    x0=xA,
                    steps=FLAGS.integration_steps,
                    method=FLAGS.integration_method,
                    tol=FLAGS.tol,
                )

            xB_vis = (xB_hat.clamp(-1, 1) + 1) / 2.0  # [0,1]

        # Save each image individually as required by folder-based FID
        for i in range(bsz):
            if saved >= FLAGS.num_gen:
                break
            out_path = os.path.join(gen_dir, f"{saved:06d}.png")
            save_image(xB_vis[i], out_path)
            saved += 1
            pbar.update(1)

        if saved >= FLAGS.num_gen:
            break

    pbar.close()

    print("Computing FID between:")
    print("  gen_dir:", gen_dir)
    print("  ref_dir:", testB_dir)

    score = fid.compute_fid(gen_dir, testB_dir, batch_size=FLAGS.batch_size)
    print("FID:", score)


if __name__ == "__main__":
    FLAGS(sys.argv)
    main(sys.argv)

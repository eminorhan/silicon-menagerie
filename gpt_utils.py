import random
import numpy as np
import torch
import yaml
import gptmodel
from omegaconf import OmegaConf
from vqmodel import VQModel, GumbelVQ
from huggingface_hub import hf_hub_download


def get_available_models():
    available_models = [
        'say_gimel'
        ]

    return available_models

def load_model(gpt_name):
    # check
    assert gpt_name in ["say_gimel", "s_gimel", "a_gimel", "y_gimel", "imagenet100_gimel", "imagenet10_gimel", "imagenet1_gimel"], "Unrecognized GPT model!"

    # switcher maps gpt model name to corresponding vq encoder model name
    switcher = {
        "say_gimel": "say_32x32_8192",
        "s_gimel": "s_32x32_8192",
        "a_gimel": "a_32x32_8192",
        "y_gimel": "y_32x32_8192",
        "imagenet100_gimel": "imagenet_16x16_16384",
        "imagenet10_gimel": "imagenet_16x16_16384",
        "imagenet1_gimel": "imagenet_16x16_16384"
    }

    # assign corresponding vq model name
    vq_name = switcher[gpt_name]

    # download checkpoint from hf (TODO: check if these load correctly)
    gpt_model_ckpt = hf_hub_download(repo_id="eminorhan/gpt_saycam", subfolder="gpt_pretrained_models", filename=gpt_name+".pt")
    vq_config_ckpt = hf_hub_download(repo_id="eminorhan/gpt_saycam", subfolder="vqgan_pretrained_models", filename=vq_name+".yaml")
    vq_model_ckpt = hf_hub_download(repo_id="eminorhan/gpt_saycam", subfolder="vqgan_pretrained_models", filename=vq_name+".ckpt")

    # load gpt model
    gpt_config = gptmodel.__dict__['GPT_gimel'](8192, 1023)  # args: vocab_size, block_size (TODO: handle this better)
    gpt_model = gptmodel.GPT(gpt_config)

    gpt_model_ckpt = torch.load(gpt_model_ckpt, map_location='cpu')
    gpt_model.load_state_dict(gpt_model_ckpt['model_state_dict'])

    # load vq model
    vq_config = load_config(vq_config_ckpt, display=True)
    vq_model = load_vqgan(vq_config, ckpt_path=vq_model_ckpt)

    return gpt_model, vq_model

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
        return config

def load_vqgan(config, ckpt_path=None, gumbel=False):
    if gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
        
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def generate_images_freely(gpt_model, vq_model, n_samples=1):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    gpt_model.to(device)
    gpt_model.eval()

    vq_model.to(device)
    vq_model.eval()

    # sample latents
    with torch.no_grad():
        s = gpt_model.sample_freely(n_samples=n_samples)

    # decode latents into images
    z = vq_model.quantize.get_codebook_entry(s, (n_samples, 32, 32, 256))  # TODO: handle this better
    x = vq_model.decode(z)
    return x

def generate_images_from_half(gpt_model, vq_model, img_dir, n_imgs=1, n_samples_per_img=2):
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    gpt_model.to(device)
    gpt_model.eval()

    vq_model.to(device)
    vq_model.eval()

    # data preprocessing
    transform = Compose([Resize(288), CenterCrop(256), ToTensor()])
    dataset = ImageFolder(img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=n_imgs, shuffle=True, num_workers=4, pin_memory=True)

    for it, (imgs, _) in enumerate(loader):
        imgs = preprocess_vqgan(imgs)
        imgs = imgs.to(device)
        z, _, [_, _, indices] = vq_model.encode(imgs)
        indices = indices.reshape(n_imgs, -1)

        # only 1 iteration
        if it == 0:
            break

    n_samples = n_samples_per_img * n_imgs

    # sample latents
    with torch.no_grad():
        s = gpt_model.sample_from_half(indices, n_samples=n_samples_per_img)

    # decode latents into images
    z = vq_model.quantize.get_codebook_entry(s, (n_samples, 32, 32, 256))  # TODO: handle these better
    x = vq_model.decode(z)
    x[:, :, 126, :] = 1  # draw a line in the middle of image

    return x

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x
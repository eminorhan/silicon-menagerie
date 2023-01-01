import random
import numpy as np
import torch
import yaml
import gptmodel
from omegaconf import OmegaConf
from vqmodel import VQModel, GumbelVQ
from huggingface_hub import hf_hub_download


def load_model(gpt_name, vq_name):

    # checks
    assert gpt_name in ["gimel_say", "gimel_s", "gimel_a", "gimel_y", "gimel_imagenet100", "gimel_imagenet10", "gimel_imagenet1"], "Unrecognized data!"
    assert vq_name in ["say_32x32_8192", "s_32x32_8192", "a_32x32_8192", "y_32x32_8192"], "Unrecognized VQ model!"

    # download checkpoint from hf (TODO: check if these load correctly)
    gpt_model_ckpt = hf_hub_download(repo_id="eminorhan/gpt_saycam/gpt_pretrained_models/", filename=gpt_name+".pt")
    vq_config_ckpt = hf_hub_download(repo_id="eminorhan/gpt_saycam/vqgan_pretrained_models/", filename=vq_name+".yaml")
    vq_model_ckpt = hf_hub_download(repo_id="eminorhan/gpt_saycam/vqgan_pretrained_models/", filename=vq_name+".ckpt")

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x
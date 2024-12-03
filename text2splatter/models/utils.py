import torch
from typing import Dict
from huggingface_hub import hf_hub_download

from text2splatter.splatter_image.scene.gaussian_predictor import GaussianSplatPredictor


def load_encoder_weights(
        encoder: torch.nn.Module, 
        splatter_cfg: Dict, 
        dataset_name: str, 
    ) -> torch.nn.Module:
    """
    Loads pre-trained weights into the encoder from a specified dataset.

    Args:
        encoder (torch.nn.Module): The encoder model to load weights into.
        splatter_cfg (dict): Configuration dictionary for the GaussianSplatPredictor.
        dataset_name (str): The name of the dataset to load weights from. If "gso", loads the latest model.
        device (torch.device): The device to map the loaded weights to.

    Returns:
        torch.nn.Module: The encoder model with loaded weights.
    """

    if dataset_name == "gso":
        model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                            filename="model_{}.pth".format("latest"))
    else:
        model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                            filename="model_{}.pth".format(dataset_name))
    gaussian_predictor = GaussianSplatPredictor(splatter_cfg)
    ckpt_loaded = torch.load(model_path)
    gaussian_predictor.load_state_dict(ckpt_loaded["model_state_dict"])
    source_enc_state_dict = gaussian_predictor.network_with_offset.encoder.enc.state_dict()
    encoder.enc.load_state_dict(source_enc_state_dict, strict=True)
    return encoder


def load_decoder_weights(
        decoder: torch.nn.Module, 
        splatter_cfg: Dict, 
        dataset_name: str, 
    ) -> torch.nn.Module:
    """
    Loads pre-trained weights into the decoder from a specified dataset.

    Args:
        decoder (torch.nn.Module): The decoder model to load weights into.
        splatter_cfg (dict): Configuration dictionary for the GaussianSplatPredictor.
        dataset_name (str): The name of the dataset to load weights from. If "gso", loads the latest model.
        device (torch.device): The device to map the loaded weights to.

    Returns:
        torch.nn.Module: The decoder model with loaded weights.
    """

    if dataset_name == "gso":
        model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                            filename="model_{}.pth".format("latest"))
    else:
        model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                            filename="model_{}.pth".format(dataset_name))
    gaussian_predictor = GaussianSplatPredictor(splatter_cfg)
    ckpt_loaded = torch.load(model_path)
    gaussian_predictor.load_state_dict(ckpt_loaded["model_state_dict"])
    source_dec_state_dict = gaussian_predictor.network_with_offset.encoder.dec.state_dict()
    decoder.dec.load_state_dict(source_dec_state_dict, strict=True)
    decoder.out.load_state_dict(gaussian_predictor.network_with_offset.out.state_dict(), strict=True)
    return decoder

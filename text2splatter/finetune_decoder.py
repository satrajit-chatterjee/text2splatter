import logging
import math
import os
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from torchvision import transforms
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL

from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card

from text2splatter.utils import parse_args
from text2splatter.models import (
    SongUNetEncoder, 
    SongUNetDecoder, 
    Text2SplatDecoder,
    load_encoder_weights, 
    load_decoder_weights
)
from text2splatter.data.dataset_factory import get_dataset


logger = get_logger(__name__)


def freeze_params(params: list) -> None:
    """Freezes the parameters of the given list of parameters.
    
    Args:
        params (list): List of parameters to freeze.
    
    Returns:
        None
    """
    for param in params:
        param.requires_grad = False

def save_model_card(
        repo_id: str,
        images=None,
        base_model=str,
        prompt=str,
        repo_folder=None
    ) -> None:
    """Saves the model card to the specified repository.
    
    Args:
        repo_id (str): The repository ID.
        images (list, optional): List of images to include in the model card. Defaults to None.
        base_model (str, optional): The base model used. Defaults to str.
        prompt (str, optional): The prompt used for training. Defaults to str.
        repo_folder (str, optional): The folder where the model card will be saved. Defaults to None.

    Returns:
        None
    """

    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
    # Custom Diffusion - {repo_id}

    These are Custom Diffusion adaption weights for {base_model}. The weights were trained on {prompt} using [Custom Diffusion](https://www.cs.cmu.edu/~custom-diffusion). You can find some example images in the following. \n
    {img_str}

    \nFor more details on the training, please follow [this link](https://github.com/huggingface/diffusers/blob/main/examples/custom_diffusion).
    """
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        prompt=prompt,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "image-to-image",
        "diffusers",
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-3d-diffusion",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text-3d-diffusion", config=vars(args))

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    '''
    Load the VAE, extract decoder, adjust layers, and set to train.
    '''
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    # vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    # image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    # Freeze VAE parameters
    vae.requires_grad_(False)
    gaussian_splat_decoder = deepcopy(vae.decoder)

    # We need input/output of this shape:
    # input.shape: torch.Size([4, 256, 16, 16])
    # output.shape: torch.Size([4, 24, 128, 128])
    gaussian_splat_decoder.conv_out =  torch.nn.Conv2d(128, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # with torch.no_grad():
    #     gaussian_splat_decoder.conv_out.weight.copy_(vae.decoder.conv_out.weight.repeat(24 // 3, 1, 1, 1))


    # Set gaussian_splat_decoder to train
    # gaussian_splat_decoder.requires_grad_(True)
    # gaussian_splat_decoder.mid_block.requires_grad_(False)
    gaussian_splat_decoder.requires_grad_(False)
    gaussian_splat_decoder.conv_out.requires_grad_(True)

    '''
    Load splatter-image encoder/decoder model. This provides ground-truth latents. 
    '''
    dataset_name = args.splatter_dataset_name
    if dataset_name == "gso" or dataset_name == "objaverse":
        cfg_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                                    filename="config_{}.yaml".format("objaverse"))
    else:
        cfg_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                                    filename="config_{}.yaml".format(dataset_name))

    splatter_cfg = OmegaConf.load(cfg_path)
        
    gt_encoder = SongUNetEncoder(splatter_cfg)
    gt_encoder = load_encoder_weights(gt_encoder, splatter_cfg, dataset_name)

    text2splat_dec = Text2SplatDecoder(splatter_cfg)
    split_dimensions, scale_inits, bias_inits = text2splat_dec.get_splits_and_inits(True, splatter_cfg)
    gt_decoder = SongUNetDecoder(
        splatter_cfg,
        split_dimensions,
        bias_inits,
        scale_inits
    )
    gt_decoder = load_decoder_weights(gt_decoder, splatter_cfg, dataset_name)

    # Set Splatter Image encoder/decoder to eval
    gt_encoder.eval()
    gt_decoder.eval()
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    gaussian_splat_decoder.to(accelerator.device, dtype=weight_dtype)
    gt_encoder.to(accelerator.device, dtype=weight_dtype)
    gt_decoder.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        gaussian_splat_decoder.enable_gradient_checkpointing()
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        if args.with_prior_preservation:
            args.learning_rate = args.learning_rate * 2.0

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    optimizer = optimizer_class(
        gaussian_splat_decoder.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # lpips_loss_fn = lpips.LPIPS(net='alex').to(accelerator.device, dtype=weight_dtype)
    # lpips_loss_fn.eval()

    if dataset_name == "gso":
        splatter_cfg.data.category = "gso"
        split = "test"
    else:
        splatter_cfg.data.category = dataset_name
        split = "train"

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            # transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            # transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    # TODO: Implement validation split
    GSO_ROOT = "/data/satrajic/google_scanned_objects/scratch/shared/beegfs/cxzheng/dataset_new/google_scanned_blender_25_w2c/" # Change this to your data directory
    PROMPTS_FOLDER = "../data/gso/prompts.json"
    PATH_FOLDER = "../data/gso/paths.json"
    splatter_cfg.data.category = "gso"
    dataset = get_dataset(splatter_cfg, split, transform=None, gso_root=GSO_ROOT, prompts_folder=PROMPTS_FOLDER, path_folder=PATH_FOLDER)
    train_dataloader = DataLoader(
                                dataset, 
                                batch_size=args.train_batch_size, 
                                shuffle=True, 
                                num_workers=10
                            )
    
    validation_dataloader = DataLoader(
                                dataset,
                                batch_size=1,
                                shuffle=True,
                                num_workers=10
                            )

    '''
    Train on only one datum for debugging purposes.
    '''
    # NotImplementedError("Debugging mode not implemented yet.")

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    (
        optimizer, 
        lr_scheduler, 
        train_dataloader, 
        gaussian_splat_decoder, 
     ) = accelerator.prepare(
                        optimizer, 
                        lr_scheduler, 
                        train_dataloader, 
                        gaussian_splat_decoder, 
                    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"Number of accelerators: {accelerator.num_processes}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # if args.resume_from_checkpoint != "latest":
        #     path = os.path.basename(args.resume_from_checkpoint)
        # else:
        #     # Get the most recent checkpoint
        #     dirs = os.listdir(args.output_dir)
        #     dirs = [d for d in dirs if d.startswith("checkpoint")]
        #     dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        #     path = dirs[-1] if len(dirs) > 0 else None

        # if path is None:
        #     accelerator.print(
        #         f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        #     )
        #     args.resume_from_checkpoint = None
        #     initial_global_step = 0
        # else:
        #     accelerator.print(f"Resuming from checkpoint {path}")
        gaussian_splat_decoder.load_state_dict(torch.load("/data/satrajic/saved_models/text-3d-diffusion/decoder/checkpoint-31250/gaussian_splat_decoder.pth"))
        # global_step = int(path.split("-")[1])
        global_step = 0

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    validation_mse_loss = []
    for _ in range(first_epoch, args.num_train_epochs):
        gaussian_splat_decoder.train()
        for _, data in enumerate(train_dataloader):
            with accelerator.accumulate(gaussian_splat_decoder):
                # input_images.shape: torch.Size([4, 1, 3, 128, 128])
                input_images = data[1]

                # input_images.shape: torch.Size([4, 3, 128, 128])
                # input_images = input_images.reshape(-1, *input_images.shape[2:])

                with torch.no_grad():
                    latents, skips = gt_encoder(input_images.to(accelerator.device, dtype=weight_dtype))
                    gt_gaussian_splats = gt_decoder(latents, skips)

                transformed_input_images = input_images.to(accelerator.device, dtype=weight_dtype)
                transformed_input_images = train_transforms(transformed_input_images)
                vae_latents = vae.encode(transformed_input_images).latent_dist.sample()
                vae_latents = vae_latents * vae.config.scaling_factor
                pred_gaussian_splats = gaussian_splat_decoder(vae_latents)
                loss = F.mse_loss(pred_gaussian_splats, gt_gaussian_splats, reduction="mean")
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(gaussian_splat_decoder.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}/")
                        os.makedirs(save_path, exist_ok=True)
                        torch.save(gaussian_splat_decoder.state_dict(), save_path + f"gaussian_splat_decoder.pth")
                        logger.info(f"Saved state to {save_path}")

            logs = {"MSE loss": loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                if global_step % args.validation_steps == 0:
                    logger.info(
                        f"Running validation... \n"
                    )
                    gaussian_splat_decoder.eval()
                    for _ in range(args.num_validation_images):
                        validation_input_images = next(iter(validation_dataloader))[1]

                        with torch.no_grad():
                            val_latents, val_skips = gt_encoder(validation_input_images.to(accelerator.device, dtype=weight_dtype))
                            val_gt_gaussian_splats = gt_decoder(val_latents, val_skips)
                        
                        validation_input_images = validation_input_images.to(accelerator.device, dtype=weight_dtype)
                        validation_input_images = train_transforms(validation_input_images)
                        val_vae_latents = vae.encode(validation_input_images).latent_dist.mean
                        val_vae_latents = val_vae_latents * vae.config.scaling_factor
                        val_latents_pred = gaussian_splat_decoder(val_vae_latents)

                        val_loss = F.mse_loss(val_latents_pred, val_gt_gaussian_splats, reduction="mean")

                        validation_mse_loss.append(val_loss.detach().item())

                    for tracker in accelerator.trackers:
                        if tracker.name == "wandb":
                            tracker.log(
                                {
                                    "validation/mse_loss": np.mean(validation_mse_loss)
                                }
                            )
                    torch.cuda.empty_cache()
                    gaussian_splat_decoder.train()

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)

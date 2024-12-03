# text2splatter
Text to 3D gaussian splatting using Stable Diffusion

### Installation
```bash
pip install -e .
```

---

## Decoder Finetuning Command
```bash
CUDA_VISIBLE_DEVICES=0 python finetune_decoder.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2" --train_batch_size 4 --report_to wandb --num_train_epochs 10 --checkpoints_total_limit 5 --enable_xformers_memory_efficient_attention --resolution 512 --seed 42 --output_dir "/data/satrajic/saved_models/text-3d-diffusion/decoder" --learning_rate 1e-4
```

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_decoder.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2" --train_batch_size 4 --report_to wandb --num_train_epochs 10 --checkpoints_total_limit 5 --enable_xformers_memory_efficient_attention --resolution 512 --seed 42 --output_dir "/data/satrajic/saved_models/text-3d-diffusion/decoder" --learning_rate 1e-4 --resume_from_checkpoint latest
```

## UNet Finetuning Command
```bash
CUDA_VISIBLE_DEVICES=1 python finetune_unet.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2" --train_batch_size 16 --report_to wandb --num_train_epochs 10 --checkpoints_total_limit 5 --enable_xformers_memory_efficient_attention --resolution 512 --dream_training --center_crop --random_flip --lr_scheduler linear
```

## LoRA UNet Finetuning Command
```bash
CUDA_VISIBLE_DEVICES=1 python finetune_unet_lora.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2" --train_batch_size 1 --report_to wandb --num_train_epochs 10 --checkpoints_total_limit 5 --enable_xformers_memory_efficient_attention --resolution 512 --random_flip --learning_rate 1e-4 --seed 42 --validation_prompt "a duck in the style of ttsplat"
```

### Notes
- [Non-LoRA] Do dream training, on only attention layers, look at other stuff like snr_gamma, EMA, but importantly, augment all captions with "isolated on *black* background". Might be worthwhile doing the black pixel thing from here: https://www.crosslabs.org//blog/diffusion-with-offset-noise

- [LoRA] Do LoRA training. Also augment captions with "isolated on *black* background" and do the black pixel thing: https://www.crosslabs.org//blog/diffusion-with-offset-noise
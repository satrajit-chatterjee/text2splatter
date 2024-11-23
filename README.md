# text2splatter
Text to 3D gaussian splatting using Stable Diffusion

### Installation
```bash
pip install -e .
```

---

## Decoder Finetuning Command
```bash
CUDA_VISIBLE_DEVICES=0 python finetune_decoder.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2" --train_batch_size 4 --rep
ort_to wandb --num_train_epochs 10 --checkpoints_total_limit 5 --enable_xformers_memory_efficient_attention  
```

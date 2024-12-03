import torch
import torch.nn as nn
from diffusers.pipelines import StableDiffusionPipeline
import matplotlib.pyplot as plt

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
prompt = "a jenga tower in 3d rendering style"
image = pipe(prompt).images[0]

# save the PIL image
plt.imshow(image)
plt.axis("off")
plt.savefig("results/sd2_test.png")

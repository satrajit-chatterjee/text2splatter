import os
import json
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
import torch
from transformers import CLIPTextModel, CLIPTokenizer

pretrained_model_name_or_path="stabilityai/stable-diffusion-2"

GSO_ROOT = "/Users/paulkathmann/code/UPenn/ESE5460/final_project/data/scratch/shared/beegfs/cxzheng/dataset_new/google_scanned_blender_25_w2c/" # Change this to your data directory
PROMPTS_FOLDER = "/Users/paulkathmann/code/UPenn/ESE5460/final_project/text2splatter/data/gso/prompts.json"
PATH_FOLDER = "/Users/paulkathmann/code/UPenn/ESE5460/final_project/text2splatter/data/gso/paths.json"
assert GSO_ROOT is not None, "Update path of the dataset"

class GSODataset(torch.utils.data.Dataset):
    def __init__(self,
                 cfg,
                 dataset_name = "test",
                 transform = None
                 ) -> None:
        
        super(GSODataset).__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer", revision=None
        )
        self.transform = transform
        self.cfg = cfg
        self.convert_to_tensor = transforms.ToTensor()
        self.root_dir = GSO_ROOT
        assert dataset_name != "train", "No training on GSO dataset!"

        self.dataset_name = dataset_name
        
        self.prompts = json.load(open(PROMPTS_FOLDER))
        self.paths = json.load(open(PATH_FOLDER))

        print('============= length of dataset %d =============' % len(self.paths))
        

    def tokenize_prompts(self, captions):
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    def __len__(self):
        return len(self.paths)
                                                      
    # batch size, num_images_per_class, 3, 128, 128
    # (25.000, 3, 128, 128), (25.000)
    # (str, (3, 128, 128))

    # batch_size = selected_images_per_class, 3, 128, 128
    def __getitem__(self, index): # returns torch.Size([4, 25, 3, 128, 128])
        image_class_folder = self.paths[index]
        filepath = os.path.join(GSO_ROOT, image_class_folder)

        image_class_name = image_class_folder.split("/")[0]
        prompt = self.tokenize_prompts(str(np.random.choice(self.prompts[image_class_name])))
        if self.transform:
            image = self.transform(Image.open(filepath))
        else:
            image = self.convert_to_tensor(Image.open(filepath))
        
        return prompt, image
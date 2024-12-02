import json
import os
import random

# replace with the path to the dataset!
ROOT_FOLDER = "/Users/paulkathmann/code/UPenn/ESE5460/final_project/data/scratch/shared/beegfs/cxzheng/dataset_new/google_scanned_blender_25_w2c/"


def get_image_paths(path_folder: str, image_folders: list[str]):
    image_paths = []
    for image_folder in image_folders:
        image_folder += "/render_mvs_25/model"
        image_folder = os.path.join(ROOT_FOLDER, image_folder)
        for image in os.listdir(image_folder):
            if image.endswith(".png") or image.endswith(".jpg"):
                image_path_end = "/".join(image_folder.split("/")[-3:])
                image_paths.append(os.path.join(image_path_end, image))
    
    # store image in a json file
    with open(f"{path_folder}/paths.json", "w") as f:
        json.dump(image_paths, f)

def store_prompt_paths(prompts_folder: str, image_folders: list[str]):
    prompts = {}
    for image_folder in image_folders:
        image_folder += "/render_mvs_25/model"
        image_folder = os.path.join(ROOT_FOLDER, image_folder)
        for file in os.listdir(image_folder):
            if file.endswith("prompts.json"):
                with open(os.path.join(image_folder, file)) as f:
                    file_contents = json.load(f)
                    if isinstance(file_contents, str):
                        file_contents = json.loads(file_contents)
                    image_folder = image_folder.split("/")[-3]
                    prompts[image_folder] = file_contents.get("prompts")
    with open(f"{prompts_folder}/prompts.json", "w") as f:
        json.dump(prompts, f)
        
def get_image_folders(gso_folder: str):
    for image_folder in os.listdir(ROOT_FOLDER):
        image_folder += "/render_mvs_25/model"
        image_folder = os.path.join(ROOT_FOLDER, image_folder)


def main():
    image_folders = os.listdir(ROOT_FOLDER)
    random.shuffle(image_folders)
    validation_folders = image_folders[:5]
    training_folders = image_folders[5:]

    gso_folder_training = "../../data/gso/training"
    gso_folder_validation = "../../data/gso/validation"

    get_image_paths(gso_folder_training, training_folders)
    store_prompt_paths(gso_folder_training, training_folders)
    
    get_image_paths(gso_folder_validation, validation_folders)
    store_prompt_paths(gso_folder_validation, validation_folders)

if __name__ == "__main__":
    main()
    
    
    
    
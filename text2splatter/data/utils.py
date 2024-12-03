import json
import os

# replace with the path to the dataset!
ROOT_FOLDER = "/data/satrajic/google_scanned_objects/scratch/shared/beegfs/cxzheng/dataset_new/google_scanned_blender_25_w2c/"


def get_image_paths(path_folder: str):
    image_paths = []
    for image_folder in os.listdir(ROOT_FOLDER):
        image_folder += "/render_mvs_25/model"
        image_folder = os.path.join(ROOT_FOLDER, image_folder)
        for image in os.listdir(image_folder):
            if image.endswith(".png") or image.endswith(".jpg"):
                image_path_end = "/".join(image_folder.split("/")[-3:])
                image_paths.append(os.path.join(image_path_end, image))
    
    # store image in a json file
    with open(f"{path_folder}/paths.json", "w") as f:
        json.dump(image_paths, f)

def store_prompts_in_one_file(prompts_folder: str):
    prompts = {}
    for image_folder in os.listdir(ROOT_FOLDER):
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


def main():
    gso_folder = "../../data/gso/"
    get_image_paths(gso_folder)
    store_prompts_in_one_file(gso_folder)

if __name__ == "__main__":
    main()
    
    
    
    
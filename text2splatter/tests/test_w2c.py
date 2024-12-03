# load all npy files from:
# 
import os
import numpy as np
ROOT_FOLDER = "/Users/paulkathmann/code/UPenn/ESE5460/final_project/data/scratch/shared/beegfs/cxzheng/dataset_new/google_scanned_blender_25_w2c"
values = [[] for _ in range(12)]
print(f"{values=}")
for file in os.listdir(ROOT_FOLDER):
    print(f"processing file {file}")
    file_path = os.path.join(ROOT_FOLDER, file, "render_mvs_25/model/")
    images = os.listdir(file_path)
    for image in images:
        if image.endswith(".npy"):
            print(image)
            x = np.load(os.path.join(file_path, image))
            print(f"shape: {x.shape}")
            print(f"{x=}")
            x = x.flatten()
            print(f"{x.shape=}")
            for i in range(12):
                values[i].append(x[i])
    
import matplotlib.pyplot as plt

# compute mean and std of each list
for i in range(12):
    print(f"{i=}")
    print(f"mean: {np.mean(values[i])}")
    print(f"std: {np.std(values[i])}")


    
fig, axs = plt.subplots(3, 4, figsize=(20, 20))
for i in range(3):
    for j in range(4):
        axs[i, j].hist(values[i*4+j], bins=100)
        axs[i, j]. set_title(f"Value {i*4+j}, mean: {np.mean(values[i*4+j]):.2f}, std: {np.std(values[i*4+j]):.2f}")
        
        
plt.show()



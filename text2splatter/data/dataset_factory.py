from .gso_dataset import GSODataset

def get_dataset(cfg, name, transform=None, gso_root=None, prompts_folder=None, path_folder=None):
    if cfg.data.category == "gso":
        return GSODataset(cfg, name, transform, gso_root, prompts_folder, path_folder)
    else:
        raise NotImplementedError(f"Dataset {cfg.data.category} not implemented")
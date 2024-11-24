from .gso_dataset import GSODataset

def get_dataset(cfg, name, transform=None):
    if cfg.data.category == "gso":
        return GSODataset(cfg, name, transform)
    else:
        raise NotImplementedError(f"Dataset {cfg.data.category} not implemented")
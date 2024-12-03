from .gso_dataset import GSODataset

def get_dataset(cfg, train, transform=None, root=None, metadata=None, style_prompt="in the style of ttsplat"):
    if cfg.data.category == "gso":
        return GSODataset(cfg, train, transform, root, metadata, style_prompt)
    else:
        raise NotImplementedError(f"Dataset {cfg.data.category} not implemented")
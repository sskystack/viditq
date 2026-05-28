import torch


def load_precomputed_text_embeds(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if "text_embeds" not in data:
        raise KeyError(f"{path} does not contain a 'text_embeds' mapping")
    return data


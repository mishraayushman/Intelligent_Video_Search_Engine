import torch
import open_clip
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model, _, _preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", device=DEVICE
)
_tokenizer = open_clip.get_tokenizer("ViT-B-32")
_model.eval()


def encode_text(query: str) -> torch.Tensor:
    """
    Encode a natural language query into a normalised CLIP embedding.
    Returns a (1, 512) CPU tensor.
    """
    tokens = _tokenizer([query]).to(DEVICE)
    with torch.no_grad():
        emb = _model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu()


def encode_image(image_path: str) -> torch.Tensor:
    """
    Encode an image file into a normalised CLIP embedding.
    Returns a (1, 512) CPU tensor.
    """
    image = _preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = _model.encode_image(image)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu()

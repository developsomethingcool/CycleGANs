# training/__init__.py

from .evaluator import evaluate_pix2pix
from .trainer import train_pix2pix


# Optional: Define what is available when importing the package
__all__ = ['train_pix2pix', 'evaluate_pix2pix']

# training/__init__.py

from .evaluator import evaluate_pix2pix
from .trainer import train_cycle_gan

# Optional: Define what is available when importing the package
__all__ = ['train_cycle_gan', 'evaluate_pix2pix']

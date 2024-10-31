# models/__init__.py
from .generator import ResNetGenerator
from .discriminator import PatchGANDiscriminator

# Optionally, you could add other imports or package-level variables here

# You could also define an __all__ list to control what is imported
__all__ = ['ResNetGenerator', 'PatchGANDiscriminator']

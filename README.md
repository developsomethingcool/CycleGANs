# CycleGANs Bidirectional Architecture Implementation

This project is an implementation of the CycleGAN architecture for bidirectional image-to-image translation between two domains. The model learns to translate images from Domain A to Domain B and the other way around without paired data, using cycle consistency loss.

# Introduction
CycleGAN makes it possible to mapp the images between two different domains without relying on a paired dataset. This implementation can be adapted to different domains.

# Project structure
'''
project_root/       
│
├── data/
│   ├── init.py
│   ├── dataset.py            # Custom dataset classes
│   └── dataloader.py         # Data loading utilities
│
├── models/
│   ├── init.py
│   ├── generator.py          # Generator model
│   ├── discriminator.py      # Discriminator model
│
├── utils/
│   ├── init.py
│   ├── utils.py              # Utility functions
│
├── training/
│   ├── init.py
│   ├── trainer.py            # Training loop
│   └── evaluator.py          # Evaluation loop
│
├── main.py                   # Main script to run the training/evaluation
└── requirements.txt          # List of dependencies
'''

dataloader.py: Creates data loaders for training, validation, and testing datasets.
dataset.py: Defines a structure of a custom dataset class FirstDomainToSecondDomain for loading and preprocessing images from both domains.
generator.py: Implementation of a generator network based on ResNet-based architecture.
discriminator.py: Implementation of a PatchGAN discriminator architecture.
trainer.py: Training loop for the CycleGAN model training.
evaluator.py: Evaluator for the trained model and generator for translated images.
utils.py: Utility functions for saving/loading checkpoints, initializing weights, and visualizing results.

# Features
Bidirectional image translation between two domains
Training and fine tuning from checkpoints is supported
Visualization of images during the training

# Installation
1. Clone the Repository
git clone https://github.com/developsomethingcool/cyclegan-bidirectional.git
cd cyclegan-bidirectional
2. Install Dependencies
You can install the required packages using pip:
pip install -r requirements.txt
If a requirements.txt file is not provided, install the packages individually:
pip install torch torchvision numpy Pillow matplotlib tqdm

# Datasets
Dataset should be prepared for both domains
Domain A: Images from the first domain
Domain B: Images from the second domain

Datadirectories should look the following way
├── data/
│ ├──trainA/
│    ├── image1.jpg
│    ├── image2.jpg 
│ ├──trainB/
│     ├── image1.jpg
│    ├── image2.jpg
Make sure that images are formatted and preprocessed as needed

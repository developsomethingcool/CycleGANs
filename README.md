# CycleGANs Bidirectional Architecture Implementation

This project is an implementation of the CycleGAN architecture for bidirectional image-to-image translation between two domains. The model learns to translate images from Domain A to Domain B and the other way around without paired data, using cycle consistency loss.

# Introduction
CycleGAN makes it possible to mapp the images between two different domains without relying on a paired dataset. This implementation can be adapted to different domains.

# Project structure
```
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
```

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
git clone https://github.com/developsomethingcool/CycleGANs.git
cd CycleGANs
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Prepare your dataset (see Configuration):

## Usage

### Training
To train the CycleGAN model:
```bash
python main.py --task train --unity_image_dir <path_to_domain_A_images> --real_image_dir <path_to_domain_B_images>


### Evaluation
To evaluate the model on a validation set:
```bash
python main.py --task eval --unity_image_dir <path_to_domain_A_images> --real_image_dir <path_to_domain_B_images>


### Image Generation

To generate images using the trained CycleGAN:
```bash
python main.py --task gen --unity_image_dir <path_to_domain_A_images>


## Project Structure

```bash
CycleGANs/
├── data/                       # Dataloader utilities
│   ├── dataloader.py           # Loads datasets and applies preprocessing
│   ├── dataset.py              # Custom dataset implementation
├── models/                     # Generator and discriminator models
│   ├── generator.py            # ResNet-based generator
│   ├── discriminator.py        # PatchGAN-based discriminator
├── training/                   # Training and evaluation logic
│   ├── trainer.py              # Training loop
│   ├── evaluator.py            # Evaluation loop
├── utils/                      # Helper functions
│   ├── utils.py                # Checkpointing, visualization, etc.
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── main.py                     # Entry point for training, evaluation, and generation
├── requirements.txt            # Python dependencies

## Configuration

### Dataset

Prepare two directories for your datasets:

- `trainA/`: Contains images from the first domain (e.g., sketches or synthetic images).
- `trainB/`: Contains images from the second domain (e.g., real-world images).

Ensure that images are in a format compatible with PyTorch's `torchvision`.


### Hyperparameters

Modify hyperparameters in `main.py` or pass them as command-line arguments:

- `--num_epochs`: Number of training epochs (default: `200`).
- `--batch_size`: Batch size for training (default: `4`).
- `--lr`: Learning rate for optimizers (default: `2e-4`).
- `--lambda_cycle`: Weight for cycle consistency loss (default: `10`).
- `--lambda_identity`: Weight for identity loss (default: `5`).

## Examples

### Visualization

During training, visualizations of the input, generated, and reconstructed images are saved in the `visualization_results` directory.

### Generated Images

After training, generate images using the command in the [Image Generation](#image-generation) section. Results will be saved in the `generated_images` directory.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [CycleGAN Paper](https://junyanz.github.io/CycleGAN/)
- [PyTorch Framework](https://pytorch.org/)
- OpenAI's tools and resources




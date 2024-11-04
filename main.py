import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from data import get_dataloaders
from models.generator import ResNetGenerator
from models.discriminator import PatchGANDiscriminator
from training import train_cycle_gan
from training.evaluator import evaluate_cyclegan
from utils.utils import load_checkpoint, generate_images
import tarfile
import os

def main():
    task = 'train'  # Options: 'train', 'eval', 'gen'
    unity_image_dir = 'trainA'
    real_image_dir = 'trainB'
    
    #checkpoint_path = None
    checkpoint_path = "cyclegan_checkpoint_epoch_70.pth.tar"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 200
    batch_size = 4
    lr = 2e-4
    lambda_l1 = 50
    lambda_cycle=10 
    lambda_identity=5

    print(f"Task: {task}")
    print(f"Edge images directory: {unity_image_dir}")
    print(f"Real images directory: {real_image_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"L1 weight: {lambda_l1}")

    # Create DataLoader
    train_loader, val_loader, test_loader = get_dataloaders(
        unity_image_dir, real_image_dir, batch_size=batch_size, num_workers=4
    )

    # Initialize models
    generator_AB = ResNetGenerator().to(device)
    generator_BA = ResNetGenerator().to(device)
    
    discriminator_A = PatchGANDiscriminator().to(device) if task == 'train' else None
    discriminator_B = PatchGANDiscriminator().to(device) if task == 'train' else None

    generator_AB.to(device)
    generator_BA.to(device)

    #discriminator_A.to(device)
    #discriminator_B.to(device)

    # Initialize optimizers
    opt_gen = optim.Adam(list(generator_AB.parameters()) + list(generator_BA.parameters()), lr=lr, betas=(0.5, 0.999))
    
    opt_disc_A = optim.Adam(discriminator_A.parameters(), lr=lr, betas=(0.5, 0.999)) if task == 'train' else None
    opt_disc_B = optim.Adam(discriminator_B.parameters(), lr=lr, betas=(0.5, 0.999)) if task == 'train' else None

    #Initializing learning rate scheduler
    if task == "train":
        scheduler_gen = optim.lr_scheduler.LambdaLR(
            opt_gen,
            #lr_lambda=lambda epoch: 1.0 - max(0, epoch - num_epochs) / float(num_epochs//2) 
            lr_lambda=lambda epoch: 2.0 - max(0, epoch - num_epochs/2) / float(num_epochs/2)
        )
        scheduler_disc_A = optim.lr_scheduler.LambdaLR(
            opt_disc_A,
            #lr_lambda=lambda epoch: 0.25 - max(0, epoch - num_epochs) / float(num_epochs//2) 
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - num_epochs/2) / float(num_epochs/2)
        )
        scheduler_disc_B = optim.lr_scheduler.LambdaLR(
            opt_disc_B,
            #lr_lambda=lambda epoch: 0.25 - max(0, epoch - num_epochs) / float(num_epochs//2) 
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - num_epochs/2) / float(num_epochs/2)
        )
    else:
        scheduler_gen = None
        scheduler_disc_A = None
        scheduler_disc_B = None

    # Load checkpoint
    start_epoch = 71
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load generators
        load_checkpoint(
            checkpoint_path, 
            generator_AB, 
            'generator_AB_state_dict', 
            optimizer=opt_gen, 
            optimizer_key='opt_gen_state_dict', 
            scheduler=scheduler_gen, 
            scheduler_key='scheduler_gen_state_dict', 
            device=device
        )
        load_checkpoint(
            checkpoint_path, 
            generator_BA, 
            'generator_BA_state_dict', 
            optimizer=None, 
            scheduler=None,  
            device=device
        )
        if task == 'train' and 'discriminator_A_state_dict' in checkpoint:
            load_checkpoint(
                checkpoint_path, 
                discriminator_A, 
                'discriminator_A_state_dict', 
                optimizer=opt_disc_A, 
                optimizer_key='opt_disc_A_state_dict', 
                scheduler=scheduler_disc_A, 
                scheduler_key='scheduler_disc_A_state_dict', 
                device=device
            )
            load_checkpoint(
                checkpoint_path,
                discriminator_B,
                'discriminator_B_state_dict',
                optimizer=opt_disc_B,
                optimizer_key='opt_disc_B_state_dict',
                scheduler=scheduler_disc_B,
                scheduler_key='scheduler_disc_B_state_dict',
                device=device
            )
        if task == 'train' and 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1  
            print(f"Resuming training from epoch {start_epoch}")



    # Perform task
    if task == 'train':
        print("Starting training...")
        train_cycle_gan(
            generator_AB, generator_BA, discriminator_A, discriminator_B, 
            train_loader, test_loader, opt_gen, opt_disc_A, opt_disc_B, 
            scheduler_gen, scheduler_disc_A, scheduler_disc_B, 
            num_epochs=num_epochs, start_epoch=start_epoch, 
            lr=lr, lambda_cycle=lambda_cycle, lambda_identity=lambda_identity, device=device
            )
        
    elif task == 'eval':
        print("Starting evaluation...")
        evaluate_cyclegan(generator_AB, generator_BA, val_loader, device, save_path='evaluation_results', num_images_to_save=16)
    elif task == 'gen':
        print("Generating images...")
        generate_images(generator_AB, generator_BA, test_loader, device, save_path='generated_images', num_images_to_save=64)

if __name__ == "__main__":
    main()

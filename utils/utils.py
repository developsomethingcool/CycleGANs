import torch
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print(f"Saving checkpoint to {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, model_key, optimizer=None, optimizer_key=None,  scheduler=None, scheduler_key=None, device='cpu'):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint[model_key])
    print(f"Checkpoint for {model_key} loaded successfully")

    # Load optimizer if provided
    if optimizer and optimizer_key:
        optimizer.load_state_dict(checkpoint[optimizer_key])
        print(f"Optimizer state for {optimizer_key} loaded successfully")

    # Load scheduler state if provided
    if scheduler and scheduler_key and scheduler_key in checkpoint:
        scheduler.load_state_dict(checkpoint[scheduler_key])
        print(f"Scheduler state for {scheduler_key} loaded successfully")

def generate_images(generator_AB, generator_BA, dataloader, device, save_path='generated_images', num_images_to_save=10):
    """
    Function to generate and save images using a trained generators in both ways.
    """
    generator_AB.eval()
    generator_BA.eval()

    os.makedirs(save_path, exist_ok=True)  

    with torch.no_grad():
        for i, (images_A, images_B) in enumerate(dataloader):
            images_A = images_A.to(device)
            images_B = images_B.to(device)

            # Generate fake images
            fake_B = generator_AB(images_A)
            fake_A = generator_BA(images_B)
            
            # Cycle consitency
            reconstructed_A = generator_BA(fake_B)
            reconstructed_B = generator_AB(fake_A)

            if i * dataloader.batch_size >= num_images_to_save:
                break

            # Concatenate images for easier visualization
            # Each row: Original A, Fake B, Reconstructed A
            # Each row: Original B, Fake A, Reconstructed B
            for j in range(images_A.size(0)):
                img_A = images_A[j]
                img_fake_B = fake_B[j]
                img_recon_A = reconstructed_A[j]

                img_B = images_B[j]
                img_fake_A = fake_A[j]
                img_recon_B = reconstructed_B[j]

                # Create a grid of images
                grid_ABA = torch.stack([img_A, img_fake_B, img_recon_A], dim=0)
                grid_BAB = torch.stack([img_B, img_fake_A, img_recon_B], dim=0)

                # Save the concatenated images
                save_image(grid_ABA, os.path.join(save_path, f'A_to_B_{i * dataloader.batch_size + j}.png'), normalize=True)
                save_image(grid_BAB, os.path.join(save_path, f'B_to_A_{i * dataloader.batch_size + j}.png'), normalize=True)

            if i < num_images_to_save // dataloader.batch_size:
                save_image(fakes, os.path.join(save_path, f'generated_{i}.png'), normalize=True)

    print(f"Images saved to {save_path}")

def visualize_results(generator_AB, generator_BA, images_A, images_B, epoch, save_path='visualization_results', device="cuda"):
    """
    Visualize and save the results of both translation directions along with cycle consistency.
    """
    generator_AB.eval()
    generator_BA.eval()

    with torch.no_grad():
        images_A = images_A.to(device)
        images_B = images_B.to(device)

        # Generate fake images
        fake_B = generator_AB(images_A)
        fake_A = generator_BA(images_B)

        # Reconstruct images for cycle consistency
        reconstructed_A = generator_BA(fake_B)
        reconstructed_B = generator_AB(fake_A)

    # Move images to CPU and convert to numpy
    images_A = images_A.cpu().detach().numpy()
    images_B = images_B.cpu().detach().numpy()
    fake_B = fake_B.cpu().detach().numpy()
    fake_A = fake_A.cpu().detach().numpy()
    reconstructed_A = reconstructed_A.cpu().detach().numpy()
    reconstructed_B = reconstructed_B.cpu().detach().numpy()

    # Number of images to visualize
    num_images = min(images_A.shape[0], 4)  # Adjust as needed

    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Set up the plot
    fig, axes = plt.subplots(4, num_images, figsize=(15, 12))
    for i in range(num_images):
        # Row 1: Original A
        img_A = np.transpose(images_A[i], (1, 2, 0))
        img_A = (img_A * 0.5) + 0.5  # Rescale from [-1, 1] to [0, 1]
        axes[0, i].imshow(np.clip(img_A, 0, 1))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original A')

        # Row 2: Fake B
        img_fake_B = np.transpose(fake_B[i], (1, 2, 0))
        img_fake_B = (img_fake_B * 0.5) + 0.5
        axes[1, i].imshow(np.clip(img_fake_B, 0, 1))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Fake B (A→B)')

        # Row 3: Reconstructed A
        img_recon_A = np.transpose(reconstructed_A[i], (1, 2, 0))
        img_recon_A = (img_recon_A * 0.5) + 0.5
        axes[2, i].imshow(np.clip(img_recon_A, 0, 1))
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Reconstructed A (A→B→A)')

        # Row 4: Original B
        img_B = np.transpose(images_B[i], (1, 2, 0))
        img_B = (img_B * 0.5) + 0.5
        axes[3, i].imshow(np.clip(img_B, 0, 1))
        axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_title('Original B')

        # Row 5: Fake A
        img_fake_A = np.transpose(fake_A[i], (1, 2, 0))
        img_fake_A = (img_fake_A * 0.5) + 0.5
        axes[4, i].imshow(np.clip(img_fake_A, 0, 1))
        axes[4, i].axis('off')
        if i == 0:
            axes[4, i].set_title('Fake A (B→A)')

        # Row 6: Reconstructed B
        img_recon_B = np.transpose(reconstructed_B[i], (1, 2, 0))
        img_recon_B = (img_recon_B * 0.5) + 0.5
        axes[5, i].imshow(np.clip(img_recon_B, 0, 1))
        axes[5, i].axis('off')
        if i == 0:
            axes[5, i].set_title('Reconstructed B (B→A→B)')

    plt.suptitle(f'Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Save the figure
    save_file_path = os.path.join(save_path, f'epoch_{epoch}_visualization.png')
    plt.savefig(save_file_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved visualization to {save_file_path}")


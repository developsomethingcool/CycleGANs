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

def visualize_results(edges, fakes, edges2, fakes2, epoch, save_path='visualization_results'):
    # Ensure tensors are in the correct format: [batch_size, channels, height, width]
    if len(edges.shape) == 2:
        edges = edges.unsqueeze(0)  # Add batch dimension if necessary
    elif len(edges.shape) == 3:
        edges = edges.unsqueeze(1)  # Add channel dimension if necessary

    # Convert tensors to numpy arrays and rescale values
    edges = edges.cpu().detach().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5
    #real_images = real_images.cpu().detach().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5
    fakes = fakes.cpu().detach().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5
    # Convert tensors to numpy arrays and rescale values
    edges2 = edges2.cpu().detach().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5
    #real_images = real_images.cpu().detach().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5
    fakes2 = fakes2.cpu().detach().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5

    # Get the minimum number of images to display
    num_images = min(edges.shape[0], 4)

    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Create a subplot
    fig, axes = plt.subplots(4, num_images, figsize=(15, 8))
    for i in range(num_images):
        if num_images == 1:
            axes[0].imshow(edges[i])
            axes[0].axis('off')
            axes[1].imshow(fakes[i])
            axes[1].axis('off')
            axes[2].imshow(edges2[i])
            axes[2].axis('off')
            axes[3].imshow(fakes2[i])
            axes[3].axis('off')
        else:
            axes[0, i].imshow(edges[i])
            axes[0, i].axis('off')
            axes[1, i].imshow(fakes[i])
            axes[1, i].axis('off')
            axes[2, i].imshow(edges2[i])
            axes[2, i].axis('off')
            axes[3, i].imshow(fakes2[i])
            axes[3, i].axis('off')


    plt.suptitle(f'Epoch {epoch}')
    
    # Save the figure instead of displaying it
    save_file_path = os.path.join(save_path, f'epoch_{epoch}_visualization.png')
    plt.savefig(save_file_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved visualization to {save_file_path}")


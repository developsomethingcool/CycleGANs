import torch
import torch.nn as nn
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import tqdm

def denormalize(tensor):
    return tensor * 0.5 + 0.5

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
    # Set generators to evaluation mode
    generator_AB.eval()
    generator_BA.eval()

    # Create directory for saving images
    os.makedirs(save_path, exist_ok=True)  

    images_saved = 0  # Counter for saved images

    with torch.no_grad():
        for i, (images_A, images_B) in enumerate(tqdm.tqdm(dataloader, desc="Generating Images")):
            images_A = images_A.to(device)
            images_B = images_B.to(device)

            # Generate fake images
            fake_B = generator_AB(images_A)
            fake_A = generator_BA(images_B)
            
            # Cycle consistency
            reconstructed_A = generator_BA(fake_B)  # A -> B -> A
            reconstructed_B = generator_AB(fake_A)  # B -> A -> B

            # Iterate over each image in the batch
            batch_size = images_A.size(0)
            for j in range(batch_size):
                if images_saved >= num_images_to_save:
                    break

                # Extract individual images
                img_A = images_A[j]
                img_fake_B = fake_B[j]
                img_recon_A = reconstructed_A[j]

                img_B = images_B[j]
                img_fake_A = fake_A[j]
                img_recon_B = reconstructed_B[j]

                # Denormalize images
                img_A_denorm = denormalize(img_A)
                fake_B_denorm = denormalize(img_fake_B)
                recon_A_denorm = denormalize(img_recon_A)
                img_B_denorm = denormalize(img_B)
                fake_A_denorm = denormalize(img_fake_A)
                recon_B_denorm = denormalize(img_recon_B)

                # Clamp images to [0,1] to ensure proper visualization
                img_A_denorm = torch.clamp(img_A_denorm, 0, 1)
                fake_B_denorm = torch.clamp(fake_B_denorm, 0, 1)
                recon_A_denorm = torch.clamp(recon_A_denorm, 0, 1)
                img_B_denorm = torch.clamp(img_B_denorm, 0, 1)
                fake_A_denorm = torch.clamp(fake_A_denorm, 0, 1)
                recon_B_denorm = torch.clamp(recon_B_denorm, 0, 1)

                # Create a grid for A -> B -> A
                grid_ABA = torch.cat((img_A_denorm, fake_B_denorm, recon_A_denorm), dim=2)  # Concatenate horizontally

                # Create a grid for B -> A -> B
                grid_BAB = torch.cat((img_B_denorm, fake_A_denorm, recon_B_denorm), dim=2)  # Concatenate horizontally

                # Save the concatenated grids
                save_image(grid_ABA, os.path.join(save_path, f'A_to_B_{images_saved}.png'), normalize=False)
                save_image(grid_BAB, os.path.join(save_path, f'B_to_A_{images_saved}.png'), normalize=False)

                images_saved += 1  # Increment the counter

            if images_saved >= num_images_to_save:
                break  # Exit the loop once the desired number of images is saved

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

def initialize_weights(net):
    """
    Networks weights initialization
    """
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

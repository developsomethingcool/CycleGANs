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

def generate_images(generator, dataloader, device, save_path='generated_images', num_images_to_save=10):
    """
    Function to generate and save images using a trained generator.
    """
    generator.eval()
    os.makedirs(save_path, exist_ok=True)  

    with torch.no_grad():
        for i, (edges, _) in enumerate(dataloader):
            edges = edges.to(device)
            fakes = generator(edges)

            if i < num_images_to_save // dataloader.batch_size:
                save_image(fakes, os.path.join(save_path, f'generated_{i}.png'), normalize=True)

    print(f"Images saved to {save_path}")

def visualize_results(edges, real_images, fakes, epoch, save_path='visualization_results'):
    
    
    # Check the dimensions of the tensors
    #print(f"Edges shape: {edges.shape}")
    #print(f"Real images shape: {real_images.shape}")
    #print(f"Fakes shape: {fakes.shape}")

    # Ensure tensors are in the correct format: [batch_size, channels, height, width]
    if len(edges.shape) == 2:
        edges = edges.unsqueeze(0)  # Add batch dimension if necessary
    elif len(edges.shape) == 3:
        edges = edges.unsqueeze(1)  # Add channel dimension if necessary

    # Convert tensors to numpy arrays and rescale values
    edges = edges.cpu().detach().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5
    real_images = real_images.cpu().detach().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5
    fakes = fakes.cpu().detach().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5

    # Get the minimum number of images to display
    num_images = min(edges.shape[0], 4)

    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Create a subplot
    fig, axes = plt.subplots(3, num_images, figsize=(15, 8))
    for i in range(num_images):
        if num_images == 1:
            axes[0].imshow(edges[i])
            axes[0].axis('off')
            axes[1].imshow(fakes[i])
            axes[1].axis('off')
            axes[2].imshow(real_images[i])
            axes[2].axis('off')
        else:
            axes[0, i].imshow(edges[i])
            axes[0, i].axis('off')
            axes[1, i].imshow(fakes[i])
            axes[1, i].axis('off')
            axes[2, i].imshow(real_images[i])
            axes[2, i].axis('off')

    plt.suptitle(f'Epoch {epoch}')
    
    # Save the figure instead of displaying it
    save_file_path = os.path.join(save_path, f'epoch_{epoch}_visualization.png')
    plt.savefig(save_file_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved visualization to {save_file_path}")

# def visualize_images(generator, dataloader, epoch, save_path='visualized_images', num_images_to_save=10):
#     """
#     Function to visualize and save images during training.
#     """
    
#     # Correctly set the device as a torch.device object
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#      # Debug print
#     print(f"Visualizing on device: {device}")

#     generator.eval()
#     os.makedirs(save_path, exist_ok=True) 
#     print(f"Saving images to: {save_path}")

#     with torch.no_grad():
#         for i, (edges, _) in enumerate(dataloader):
#             edges = edges.to(device)  # Make sure device is valid
#             fakes = generator(edges)

#             if i < num_images_to_save // dataloader.batch_size:
#                 save_image(fakes, os.path.join(save_path, f'epoch {epoch} generated_{i}.png'), normalize=True)


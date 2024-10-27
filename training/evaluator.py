import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torchvision.utils import save_image

def denormalize(tensor):
    return tensor * 0.5 + 0.5

def evaluate_pix2pix(generator, dataloader, device, save_path='evaluation_results', num_images_to_save=16):
    generator.eval()
    os.makedirs(save_path, exist_ok=True)

    criterion_l1 = nn.L1Loss()

    total_l1_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, (edges, reals) in enumerate(tqdm(dataloader, desc='Evaluating')):
            edges = edges.to(device)
            reals = reals.to(device)
            fakes = generator(edges)

            # Debugging: Print output statistics
            print(f"Batch {i}, Fake Images Mean: {fakes.mean().item()}, Std: {fakes.std().item()}")

            # Compute L1 loss
            l1_loss = criterion_l1(fakes, reals)
            total_l1_loss += l1_loss.item()
            num_batches += 1

            # Save images for visualization
            if i < num_images_to_save:
                # Denormalize images
                edges_denorm = denormalize(edges)
                fakes_denorm = denormalize(fakes)
                reals_denorm = denormalize(reals)

                # Clamp images to [0,1]
                edges_denorm = torch.clamp(edges_denorm, 0, 1)
                fakes_denorm = torch.clamp(fakes_denorm, 0, 1)
                reals_denorm = torch.clamp(reals_denorm, 0, 1)

                # Save samples directly for debugging
                if i == 0:
                    save_image(edges_denorm, os.path.join(save_path, 'edge_sample.png'))
                    save_image(reals_denorm, os.path.join(save_path, 'real_sample.png'))

                # Concatenate and save images
                images_to_save = torch.cat((edges_denorm, fakes_denorm, reals_denorm), dim=3)
                save_image(images_to_save, os.path.join(save_path, f'eval_{i}.png'))

    avg_l1_loss = total_l1_loss / num_batches
    print(f"Average L1 Loss over validation set: {avg_l1_loss}")
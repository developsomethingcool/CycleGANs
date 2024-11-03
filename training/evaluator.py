import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torchvision.utils import save_image

def denormalize(tensor):
    return tensor * 0.5 + 0.5

def evaluate_cyclegan(generator_AB, generator_BA, dataloader, device, save_path='evaluation_results', num_images_to_save=16):
    generator_AB.eval()
    generator_BA.eval()

    os.makedirs(save_path, exist_ok=True)

    criterion_l1 = nn.L1Loss()

    total_l1_loss_AB = 0.0
    total_l1_loss_BA = 0.0
    num_batches = 0

    #Number saved images
    images_saved = 0

    with torch.no_grad():
        for i, (images_A, images_B) in enumerate(tqdm(dataloader, desc='Evaluating CycleGAN')):
            images_A = images_A.to(device)
            images_B = images_B.to(device)

            # Forward pass: A -> B and B -> A
            fakes_B = generator_AB(images_A)
            fakes_A = generator_BA(images_B)

            reconstructed_A = generator_BA(fakes_B)  # A -> B -> A
            reconstructed_B = generator_AB(fakes_A)  # B -> A -> B

            # Debugging: Print output statistics for both generators
            print(f"Batch {i}:")
            print(f"  Fakes B - Mean: {fakes_B.mean().item():.4f}, Std: {fakes_B.std().item():.4f}")
            print(f"  Fakes A - Mean: {fakes_A.mean().item():.4f}, Std: {fakes_A.std().item():.4f}")
            print(f"  Reconstructed A - Mean: {reconstructed_A.mean().item():.4f}, Std: {reconstructed_A.std().item():.4f}")
            print(f"  Reconstructed B - Mean: {reconstructed_B.mean().item():.4f}, Std: {reconstructed_B.std().item():.4f}")

            # Compute L1 loss
            l1_loss_AB = criterion_l1(reconstructed_A, images_A)
            total_l1_loss_AB += l1_loss_AB.item()
            l1_loss_BA = criterion_l1(reconstructed_B, images_B)
            total_l1_loss_BA += l1_loss_BA.item()
            num_batches += 1

            # Save images for visualization
            if i < num_images_to_save:
                # Denormalize images
                images_A_denorm = denormalize(images_A)
                fakes_A_denorm = denormalize(fakes_A)
                reconstructed_A_denorm = denormalize(reconstructed_A)
                fakes_B_denorm = denormalize(fakes_B)
                reconstructed_B_denorm = denormalize(reconstructed_B)
                images_B_denorm = denormalize(images_B)

                 # Clamp images to [0, 1] to ensure proper visualization
                images_A_denorm = torch.clamp(images_A_denorm, 0, 1)
                fakes_B_denorm = torch.clamp(fakes_B_denorm, 0, 1)
                reconstructed_A_denorm = torch.clamp(reconstructed_A_denorm, 0, 1)
                images_B_denorm = torch.clamp(images_B_denorm, 0, 1)
                fakes_A_denorm = torch.clamp(fakes_A_denorm, 0, 1)
                reconstructed_B_denorm = torch.clamp(reconstructed_B_denorm, 0, 1)

                 # Iterate over batch elements
                batch_size = images_A.size(0)
                for j in range(batch_size):
                    if images_saved >= num_images_to_save:
                        break

                    # Select the j-th image in the batch
                    img_A = images_A_denorm[j]
                    fakes_B_img = fakes_B_denorm[j]
                    recon_A_img = reconstructed_A_denorm[j]

                    img_B = images_B_denorm[j]
                    fakes_A_img = fakes_A_denorm[j]
                    recon_B_img = reconstructed_B_denorm[j]

                    # Concatenate images horizontally: A -> fakes B -> recon A
                    row1 = torch.cat((img_A, fakes_B_img, recon_A_img), dim=2)  # Concatenate along width

                    # Concatenate images horizontally: B -> fakes A -> recon B
                    row2 = torch.cat((img_B, fakes_A_img, recon_B_img), dim=2)  # Concatenate along width

                    # Concatenate the two rows vertically
                    grid = torch.cat((row1, row2), dim=1)  # Concatenate along height

                    # Save the concatenated grid
                    save_image(grid, os.path.join(save_path, f'eval_{images_saved}.png'), normalize=False)
                    images_saved += 1

     # Calculate average cycle consistency losses
    avg_cycle_loss_AB = total_l1_loss_AB / num_batches
    avg_cycle_loss_BA = total_l1_loss_BA / num_batches

    print(f"\nAverage Cycle Consistency L1 Loss (A→B→A): {avg_cycle_loss_AB:.4f}")
    print(f"Average Cycle Consistency L1 Loss (B→A→B): {avg_cycle_loss_BA:.4f}")
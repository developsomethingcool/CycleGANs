import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.utils import save_checkpoint, load_checkpoint, generate_images, visualize_results

# Parameters
n_discriminator_updates = 1 
n_generator_updates = 1      


def train_pix2pix(generator, discriminator, train_dataloader, visualization_loader, opt_gen, opt_disc, scheduler_gen, scheduler_disc, num_epochs=100, start_epoch=1, lr=2e-4, lambda_l1=100, device="cuda"):
 
    #Binary cross entropy with sigmoid layer
    criterion_gan = nn.BCELoss()  
    # L1 loss for pixel-wise similarity
    criterion_l1 = nn.L1Loss()    

    for epoch in range(start_epoch, num_epochs+1):
        loop = tqdm(train_dataloader, leave=True, desc=f"Epoch [{epoch}/{num_epochs}]")

        for idx, (edges, reals) in enumerate(loop):
            edges, reals = edges.to(device), reals.to(device)

            # Update the Discriminator (n times)
            for _ in range(n_discriminator_updates):
                # Training of discriminator
                discriminator.train()
                generator.eval()

                # Test discriminator on real images
                preds_real = discriminator(edges, reals)

                # Dynamically create the real labels 
                #real_label = torch.ones_like(preds_real, device=device)
                real_label = torch.full_like(preds_real, 0.9, device=device)

                # Compute the loss of discriminator on real images
                loss_disc_real = criterion_gan(preds_real, real_label)

                # Generate fake images based on edges
                fakes = generator(edges)

                #Test discriminator on fake images 
                preds_fake = discriminator(edges, fakes.detach())

                #Dynamically create fake labels
                fake_label = torch.zeros_like(preds_fake, device=device)
                #fake_label = torch.full_like(preds_fake, 0.0, device=device)

                #Compute the loss of discriminator on fake images
                loss_disc_fake = criterion_gan(preds_fake, fake_label)

                # Total discriminator loss
                loss_disc = (loss_disc_real + loss_disc_fake) / 2

                #Backprop and optimize discriminator
                #Cleare the gradient
                opt_disc.zero_grad()
                #Compute the backprop 
                loss_disc.backward()
                #Update parameters
                opt_disc.step()
            
            # Update the Generator (m times)
            for _ in range(n_generator_updates):
            # Training of a generator
                generator.train()
                discriminator.eval()

                #Generate fake images
                fakes = generator(edges)
                #Testing discriminator on fake images
                preds_fake = discriminator(edges, fakes)
                #Loss of generator on fake images
                real_label = torch.ones_like(preds_fake, device=device)
                real_label = torch.full_like(preds_real, 0.9, device=device)

                #fake_label = torch.zeros_like(preds_fake, device=device)
                loss_gan = criterion_gan(preds_fake, real_label) 

                #Normalize L1 loss, per pixel difference between fake and real images
                loss_l1 = criterion_l1(fakes, reals) * lambda_l1 

                # Total generator loss
                loss_gen = loss_gan + loss_l1

                # Backprop and optimize generator
                opt_gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()

            # Update progress bar
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss_gen=loss_gen.item(), loss_disc=loss_disc.item())

        # Step the schedulers at the end of each epoch
        if scheduler_gen and scheduler_disc:
            scheduler_gen.step()
            scheduler_disc.step()

            # Log the current learning rates
            current_lr_gen = scheduler_gen.get_last_lr()[0]
            current_lr_disc = scheduler_disc.get_last_lr()[0]
            print(f"Epoch [{epoch}/{num_epochs}] - Generator LR: {current_lr_gen:.6f}, Discriminator LR: {current_lr_disc:.6f}")

        # Save checkpoint
        if (epoch) % 5 == 0:
            save_checkpoint({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'opt_gen_state_dict': opt_gen.state_dict(),
                'opt_disc_state_dict': opt_disc.state_dict(),
                'scheduler_gen_state_dict': scheduler_gen.state_dict(),
                'scheduler_disc_state_dict': scheduler_disc.state_dict(),
            }, filename=f"pix2pix_checkpoint_epoch_{epoch}.pth.tar")

        if epoch % 2 == 0:
            visualize_results(edges, reals, fakes, epoch)
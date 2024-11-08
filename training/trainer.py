import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.utils import save_checkpoint, load_checkpoint, generate_images, visualize_results

# Number of discriminator updates per iteration
n_discriminator_updates = 1 
# Number of generator updates per iteration
n_generator_updates = 2      


def train_cycle_gan(generator_AB, generator_BA, discriminator_A, discriminator_B, train_dataloader, visualization_loader, opt_gen, opt_disc_A, opt_disc_B, scheduler_gen, scheduler_disc_A, scheduler_disc_B, num_epochs=100, start_epoch=1, lr=2e-4, lambda_cycle=10, lambda_identity=0, device="cuda"):
    # Training function for CycleGAN architecture

    #Mean squared error, adversarial loss
    criterion_gan = nn.MSELoss()
    
    # L1 loss, cycle consistency and identity loss
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()   

    for epoch in range(start_epoch, num_epochs+1):
        loop = tqdm(train_dataloader, leave=True, desc=f"Epoch [{epoch}/{num_epochs}]")

        for idx, (images_A, images_B) in enumerate(loop):
            # Move images to use on GPU
            images_A, images_B = images_A.to(device), images_B.to(device)

            # Update the Discriminator (n times)
            for _ in range(n_discriminator_updates):
                # Training of discriminator
                discriminator_A.train()
                discriminator_B.train()

                # Set generator for evaluation
                generator_AB.eval()
                generator_BA.eval()

                # Zero gradients for discriminators to zero
                opt_disc_A.zero_grad()
                opt_disc_B.zero_grad()

                # Generate fake images
                with torch.no_grad():
                    fake_B = generator_AB(images_A)
                    fake_A = generator_BA(images_B)

                # Test discriminator on real_life images
                preds_real_A = discriminator_A(images_A)
                
                # Real labels = 1, fake labels = 0
                real_label_A = torch.ones_like(preds_real_A, device=device)
                fake_label_A = torch.zeros_like(preds_real_A, device=device)
                
                loss_D_A_real = criterion_gan(preds_real_A, real_label_A)

                #Test discriminator on unity images 
                preds_fake_A = discriminator_A(fake_A.detach())

                #Compute the loss of discriminator on fake images
                loss_D_A_fake = criterion_gan(preds_fake_A, fake_label_A)

                # Total loss D_A
                loss_D_A = (loss_D_A_real + loss_D_A_fake) / 2

                #Backprop and optimize discriminator D_A
                #Compute the backprop 
                loss_D_A.backward()
                #Update parameters
                opt_disc_A.step()

                # Test discriminator on real_life images
                preds_real_B = discriminator_B(images_B)
                
                 # Real labels = 1, fake labels = 0  
                real_label_B = torch.ones_like(preds_real_B, device=device)
                fake_label_B = torch.zeros_like(preds_real_B, device=device)

                loss_D_B_real = criterion_gan(preds_real_B, real_label_B)

                #Test discriminator on unity images 
                preds_fake_B = discriminator_B(fake_B.detach())

                #Compute the loss of discriminator on fake images
                loss_D_B_fake = criterion_gan(preds_fake_B, fake_label_B)

                # Total loss D_A
                loss_D_B = (loss_D_B_real + loss_D_B_fake) / 2

                #Backprop and optimize discriminator D_A
                #Compute the backprop 
                loss_D_B.backward()
                #Update parameters
                opt_disc_B.step()


            # Set generators to train mode
            generator_AB.train()
            generator_BA.train()

            # Set discriminators to eval mode (to provide consistent feedback)
            discriminator_A.eval()
            discriminator_B.eval()

            # Update the Generator (m times)
            for _ in range(n_generator_updates):
            
                # Training of a generator
                opt_gen.zero_grad()

                fake_B = generator_AB(images_A)
                fake_A = generator_BA(images_B)

                # Adversarial loss for generator
                preds_fake_B = discriminator_B(fake_B)

                # Real labels = 1, fake labels = 0  
                real_label_B = torch.ones_like(preds_fake_B, device=device)
                fake_label_B = torch.zeros_like(preds_fake_B, device=device)

                loss_GAN_AB = criterion_gan(preds_fake_B, real_label_B)

                preds_fake_A = discriminator_A(fake_A)

                # Real labels = 1, fake labels = 0  
                real_label_A = torch.ones_like(preds_fake_A, device=device)
                fake_label_A = torch.zeros_like(preds_fake_A, device=device)
                
                loss_GAN_BA = criterion_gan(preds_fake_A, real_label_A)

                # Cycle consistency loss
                reconstructed_A = generator_BA(fake_B)
                loss_cycle_A = criterion_cycle(reconstructed_A, images_A) 

                reconstructed_B = generator_AB(fake_A)
                loss_cycle_B = criterion_cycle(reconstructed_B, images_B)

                loss_cycle = (loss_cycle_A + loss_cycle_B) * lambda_cycle

                # Identity loss
                if lambda_identity > 0:
                    identity_B = generator_AB(images_B)
                    loss_identity_B = criterion_identity(identity_B, images_B) * lambda_identity

                    identity_A = generator_BA(images_A)
                    loss_identity_A = criterion_identity(identity_A, images_A) * lambda_identity

                    loss_identity = loss_identity_A + loss_identity_B
                else:
                    loss_identity = 0

                 # Total generator loss
                loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle + loss_identity

                # Backprop and optimize generator
                loss_G.backward()
                opt_gen.step()


                # Update progress bar
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(loss_gen=loss_G.item(), loss_disc_A=loss_D_A.item(), loss_disc_B=loss_D_B.item())

        # Step the schedulers at the end of each epoch
        if scheduler_gen and scheduler_disc_A and scheduler_disc_B:
            scheduler_gen.step()
            scheduler_disc_A.step()
            scheduler_disc_B.step()

            # Log the current learning rates
            current_lr_gen = scheduler_gen.get_last_lr()[0]
            current_lr_disc_A = scheduler_disc_A.get_last_lr()[0]
            current_lr_disc_B = scheduler_disc_B.get_last_lr()[0]
            print(f"Epoch [{epoch}/{num_epochs}] - Generator LR: {current_lr_gen:.6f}, Discriminator A LR: {current_lr_disc_A:.6f}, Discriminator B LR: {current_lr_disc_B:.6f}")

        # Save checkpoint
        if (epoch) % 5 == 0:
            save_checkpoint({
                'epoch': epoch,
                'generator_AB_state_dict': generator_AB.state_dict(),
                'generator_BA_state_dict': generator_BA.state_dict(),
                'discriminator_A_state_dict': discriminator_A.state_dict(),
                'discriminator_B_state_dict': discriminator_B.state_dict(),
                'opt_gen_state_dict': opt_gen.state_dict(),
                'opt_disc_A_state_dict': opt_disc_A.state_dict(),
                'opt_disc_B_state_dict': opt_disc_B.state_dict(),
                'scheduler_gen_state_dict': scheduler_gen.state_dict(),
                'scheduler_disc_A_state_dict': scheduler_disc_A.state_dict(),
                'scheduler_disc_B_state_dict': scheduler_disc_B.state_dict(),
            }, filename=f"cyclegan_checkpoint_epoch_{epoch}.pth.tar")

        # if epoch % 2 == 0:
        visualize_results(images_A, fake_B, images_B, fake_A, epoch)
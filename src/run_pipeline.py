from datetime import datetime
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torchvision import utils, transforms
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from datasets import load_dataset
from diffusers import DDPMScheduler, UNet2DModel

image_size: int = 0
batch_size: int = 0
device: torch.device

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def make_grid(images, size=64):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im

def save_images(x, loop_count):
    for i in range(x.shape[0]):
        img = x[i] * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
        img = img.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
        img = Image.fromarray(np.array(img).astype(np.uint8))
        
        try:
            os.mkdir(f'output_images_{loop_count}')
        except:
            pass
        img.save(f'output_images_{loop_count}/image_{i:03d}.png')

def transform(images):
    preprocess = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),  # Resize
        transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
        transforms.ToTensor(),  # Convert to tensor (0, 1)
        transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
    ]
    )

    images = [preprocess(image.convert("RGB")) for image in images["image"]]
    return {"images": images}

def load_data(folder_path: str) -> torch.utils.data.DataLoader:
    dataset = load_dataset("imagefolder", data_dir=folder_path, split="train")

    dataset.set_transform(transform)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True
    )

    return train_dataloader

def define_model() -> UNet2DModel:
    model = UNet2DModel(
        sample_size=image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(64, 128, 128, 256),  # More channels -> more parameters
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",  # a regular ResNet upsampling block
        ),
    )
    model.to(device)
    return model

def train_model(model, train_dataloader, noise_scheduler):
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

    losses = []

    for epoch in range(30):
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"].to(device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Get the model prediction
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # Calculate the loss
            loss = F.mse_loss(noise_pred, noise)
            loss.backward(loss)
            losses.append(loss.item())

            # Update the model parameters with the optimizertime.time.now
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 5 == 0:
            loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
            print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}, time: {datetime.now()}")

def sample_model(model, noise_scheduler, loop_count):
    sample = torch.randn(1000, 3, image_size, image_size).to(device)

    for i, t in enumerate(noise_scheduler.timesteps):

        # Get model pred
        with torch.no_grad():
            residual = model(sample, t).sample

        # Update sample with step
        sample = noise_scheduler.step(residual, t, sample).prev_sample

        save_images(sample, loop_count)

def check_image_sample(data_loader):
    xb = next(iter(data_loader))["images"].to(device)[:8]
    print("X shape:", xb.shape)
    print(show_images(xb).resize((8 * 64, 64), resample=Image.NEAREST))


def main():
    """
    While dataset.entropy() > baseline and mutual_info(dataset_{t-1}, dataset_{t}) > baseline,
    """
    loop_count: int = 2

    noise_scheduler = DDPMScheduler(num_train_timesteps=4000, beta_schedule="squaredcos_cap_v2")
    model = define_model()

    while loop_count < 11: # 10 loops
        folder_path: str = f"src/output_images_{loop_count - 1}"
        while not os.path.exists(folder_path):
            time.sleep(144000)
            data_loader = load_data(folder_path)
            check_image_sample(data_loader)
            trained_model = train_model(model, data_loader, noise_scheduler)
            sample_model(trained_model, noise_scheduler, loop_count)
            loop_count += 1


if len(sys.argv) !=4:
    print(("\nrun_pipeline.py <image_size> <batch_size> <number_epochs> \n"))
    sys.exit()
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    number_epochs = sys.argv[3]
    main()
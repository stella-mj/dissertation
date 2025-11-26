import kagglehub
from kagglehub import KaggleDatasetAdapter

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt

def import_data() -> any:
    
  # Set the path to the file you'd like to load
  file_path = "\Users\stell\Documents\uni\CS\VIP\dissertation\datasets"

  # Load the latest version
  hf_dataset = kagglehub.load_dataset(
    KaggleDatasetAdapter.HUGGING_FACE,
    "basmarg/apple-dataset-images",
    file_path,
    # Provide any additional arguments like 
    # sql_query, hf_kwargs, or pandas_kwargs. See 
    # the documenation for more information:
    # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterhugging_face
  )

  print("Hugging Face Dataset:", hf_dataset)

  return hf_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
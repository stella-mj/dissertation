import torch
print("Cuda is available: {}".format(torch.cuda.is_available()))
print(torch.cuda.get_device_name(0))
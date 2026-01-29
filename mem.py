import gc
import torch

for obj in dir():
    if not obj.startswith('_'):
        del globals()[obj]

gc.collect()
torch.cuda.empty_cache()
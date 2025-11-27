def load_data() -> any:
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    x, y = next(iter(train_dataloader))
    print('Input shape:', x.shape)
    print('Labels:', y)
    plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')

def corrupt(x, amount):
  """Corrupt the input `x` by mixing it with noise according to `amount`"""
  noise = torch.rand_like(x)
  amount = amount.view(-1, 1, 1, 1) # Sort shape so broadcasting works
  return x*(1-amount) + noise*amount 
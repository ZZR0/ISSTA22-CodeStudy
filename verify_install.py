import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    # Test tensor allocation on each GPU
    x = torch.rand(1000, 1000).to(f'cuda:{i}')
    print(f"Successfully allocated tensor on GPU {i}")

import torch 

test_tensor = torch.rand((4, 7))

first_dim = test_tensor.shape[0]
second_dim = test_tensor.shape[1]

print("Ayo, torch dims:", first_dim, second_dim)
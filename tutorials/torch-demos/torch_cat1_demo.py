import torch 

added_vec_dim = 6
bert_dim = 8


tensor1 = torch.rand(bert_dim)

# tensor2 = torch.rand((added_vec_dim))
tensor2 = torch.tensor([0.6 + (i / 10) for i in range(3)])

aug_tensor = torch.cat([tensor1, tensor2], 0)


print("Tensor1:\n", tensor1)
print("Tensor2:\n", tensor2)

print("Concatenated tensor:", aug_tensor.size())
print(aug_tensor)
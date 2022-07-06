import torch 

batch_size = 4

added_vec_dim = 6
bert_dim = 8


tensor1 = torch.rand((batch_size, bert_dim))

# tensor2 = torch.rand((added_vec_dim))
tensor2 = torch.tensor([0.6 + (i / 10) for i in range(3)])
tensor2 = torch.unsqueeze(tensor2, 0)
tensor2 = torch.cat([tensor2] * batch_size, 0)

aug_tensor = torch.cat([tensor1, tensor2], 1)


print("Tensor1:\n", tensor1)
print("Tensor2:\n", tensor2)

print("Concatenated tensor:", aug_tensor.size())
print(aug_tensor)
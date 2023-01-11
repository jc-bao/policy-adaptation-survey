import torch
from icecream import ic

matrix_0 = torch.tensor([[1,0],[0,1]]) # Tensor(input_dim, input_dim)
matrix_1 = torch.tensor([[0,1],[1,0]]) # Tensor(input_dim, input_dim)
matrix = torch.stack([matrix_0]) # Tensor(output_dim, input_dim, input_dim)
vector = torch.tensor([0]) # Tensor(output_dim)
x = torch.tensor([[1,2], [1,2]]) # Tensor(batch_size, input_dim)
# quadratic function
q_0 = torch.einsum('bi,ij,bj->b', x, matrix_0, x) # Tensor(batch_size)
q_1 = torch.einsum('bi,ij,bj->b', x, matrix_1, x) # Tensor(batch_size)
quadratic = q_0 + q_1 # Tensor(batch_size)
quadratic = torch.einsum('bi,oij,bj->bo', x, matrix, x)  
# quadratic + linear
y = quadratic + vector # Tensor(batch_size)
ic(y)
import torch
# import numpy as np

M = torch.zeros(2, 3) + 1
m1 = torch.zeros(2, 3) + 2
m2 = torch.zeros(3, 3) + 10

A = torch.addmm(M, m1, m2)
print("A:\n", A)

B = torch.addmm(M, m1, m2, beta=1, alpha=-2)
print("\nB 1-2:\n", B)
print("\nM:\n", M)
C = M.addmm_(m1, m2, beta=1, alpha=-2)
# C = M.addmm_(m1, m2, 1, -2) will cause a error, so beta and alpha must be allocated precisely 
# TypeError: addmm_() takes 2 positional arguments but 4 were given

print("\nC 1-2:\n", C)
print("\nM 1-2:\n", M)

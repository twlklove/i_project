from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = x.new_ones(5, 3, dtype=torch.double)
# new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)
# override dtype!
print(x)
# result has the same size

print(x.size())


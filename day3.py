import open3d as o3d
import numpy as np

a = o3d.core.Tensor([[1, 2, 3], [4, 5, 6]])
print("Created from list:\n{}".format(a))

b = o3d.core.Tensor(np.array([0, 1, 2]))
print("\nCreated from numpy array:\n{}".format(b))

a_float = o3d.core.Tensor([0.0, 1.0, 2.0])
print("\nDefault dtype and device:\n{}".format(a_float))

a_cpu = o3d.cpu.Tensor([0, 1, 2])
a_gpu = a_cpu.cuda(0)
print(a_gpu)
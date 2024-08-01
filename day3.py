import open3d as o3d
import numpy as np

# Crează un set de puncte simplu folosind NumPy
points = np.random.rand(9000, 3)

# Transformă setul de puncte într-un PointCloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Afișează PointCloud-ul
o3d.visualization.draw_geometries([pcd])

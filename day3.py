import open3d as o3d
import numpy as np


#citirea si vizualizarea datelor PCD initiale

def read_visualise(pcd_name):
    pcd = o3d.io.read_point_cloud(pcd_name)
    pcd.paint_uniform_color([0, 0, 1]) #punem toate punctele initiale pe culoarea albastra
    o3d.visualization.draw_geometries([pcd])


#pcd_name = "files/transit_4.pcd"
#read_visualise(pcd_name)


#preprocesare
# prin filtrare eliminam punctele izolate. punctele care sunt prea departe de majoritatea punctelor vecine for fi sterse
def highlight (pcd, nb_neighbors = 20, std_ratio = 2.0):
    """Filtreaza outliers si evidentiaza """
    # cl = obiectul ce contine rezultatele filtrarii
    # ind = vector de indecsi pentru punctele care au fost considerate valide
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors = nb_neighbors, std_ratio = std_ratio)
    pcd_filtered = pcd.select_by_index(ind)
    pcd_outliers = pcd.select_by_index(ind, invert = True)

    #coloram punctele
    pcd_filtered.paint_uniform_color([0.0, 1.0, 0.0]) #verde pentru punctele valabile
    pcd_outliers.paint_uniform_color([1.0, 0.0, 0.0]) #rosu

 #   o3d.visualization.draw_geometries([pcd_filtered, pcd_outliers])

    return pcd_filtered, pcd_outliers

#pcd_name = "files/transit_4.pcd"
#pcd = o3d.io.read_point_cloud(pcd_name)
#pcd_filtered, pcd_outliers = highlight(pcd)


def filter (pcd, nb_neighbors = 20, std_ratio = 2.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors = nb_neighbors, std_ratio = std_ratio)
    pcd_filtered = pcd.select_by_index(ind)

    pcd_filtered.paint_uniform_color([0.0, 1.0, 0.0]) #verde pentru punctele valabile

    o3d.visualization.draw_geometries([pcd_filtered])

    return pcd_filtered

#pcd_name = "files/transit_4.pcd"
#pcd = o3d.io.read_point_cloud(pcd_name)
#pcd_filtered = filter(pcd)



#downsampling ==> reducem numarul de puncte din nor prin preluarea unui subset
#voxel downsampling

def downsample_voxel(pcd, voxel_size=0.1):

    print(f"Downsampling cu dimensiunea voxelului de {voxel_size:.3f} metri...")
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)
    print(f"Numarul de puncte inainte de downsampling: {len(pcd.points)}")
    print(f"Numarul de puncte dupa downsampling: {len(pcd_downsampled.points)}")

    return pcd_downsampled

#pipeline principal
#pcd = o3d.io.read_point_cloud(pcd_name)
#pcd_filtered = filter(pcd)
#pcd_downsampled = downsample_voxel(pcd_filtered, voxel_size= 0.1)

def slice_point_cloud(pcd, y_value, thickness = 0.01):
    """Feliem norul de puncte utilizand segment_plane"""
    plane_model = [0, 1, 0, -y_value]
    #segmentarea norului de puncte
    _, inliers = pcd.segment_plane(distance_threshold=thickness, ransac_n=3, num_iterations=1000)
    sliced_pcd = pcd.select_by_index(inliers)
    return sliced_pcd





pcd_name = "files/transit_4.pcd"
pcd = o3d.io.read_point_cloud(pcd_name)
pcd_filtered_1, pcd_outliers = highlight(pcd)
pcd_filtered_2 = filter(pcd)
pcd_downsampled = downsample_voxel(pcd_filtered_2, voxel_size= 0.1)

nr_slices = 10
y_min, y_max = pcd_downsampled.get_min_bound()[1], pcd_downsampled.get_max_bound()[1]
y_interval = (y_max - y_min)/nr_slices

all_slices = []
for i in range(nr_slices):
    y_value = y_min + i * y_interval
    slice = slice_point_cloud(pcd_downsampled, y_value)
    slice.paint_uniform_color([1.0, 0.0, 0.0])
    all_slices.append(slice)

o3d.visualization.draw_geometries(all_slices + [pcd_outliers])



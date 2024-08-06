import open3d as o3d
import numpy as np


class PointCloudProcessor:
    def __init__(self, pcd_name):
        self.pcd = o3d.io.read_point_cloud(pcd_name)


    def visualize(self):
        o3d.visualization.draw_geometries([self.pcd])

    def filter_outliers(self, nb_neighbors = 20, std_ratio = 2.0, visualize = False):
        #eliminam punctele care se afla la o distanta prea mare fata de majoritatea punctelor vecine, adica outliers
        #punctele sunt erori de masuratori
        #nb_neighbors = numarul de vecini considerati pentru fiecare punct
        #std_ratio = factor care determina cat de departe poate fi un punct de vecinii sai
        #punctele care sunt la o distanta mai mare de std_ratio vor fi eliminate
        #remove_statistical_outiler calculeaza distanta medie si abaterea standard

        _, ind = self.pcd.remove_statistical_outlier(nb_neighbors, std_ratio) #ind este un vector NumPy ce contine indexul punctelor care se incadreaza in distanta
        self.pcd = self.pcd.select_by_index(ind) #actualizam norul de puncte cu punctele ramase
        if visualize:
            self.visualize()

    def downsample_voxel(self, voxel_size=0.01): #reducem numarul de puncte din norul de puncte prin aplicarea voxel gridului
        self.pcd = self.pcd.voxel_down_sample(voxel_size)

    def visualize_voxel_grid(self, voxel_size=0.05):
        print('Display voxel grid')
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd, voxel_size=voxel_size)
        o3d.visualization.draw_geometries([voxel_grid])

    def normalize_and_color(self): #normalizam si coloram norul de puncte pentru a-l vizualiza mai usor
        self.pcd.scale(1 / np.max(self.pcd.get_max_bound() - self.pcd.get_min_bound()), center=self.pcd.get_center())
        N = len(self.pcd.points)
        self.pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))

    def visualize(self):
        o3d.visualization.draw_geometries([self.pcd])


def main():
    pcd_name = "files/transit_4.pcd"
    processor = PointCloudProcessor(pcd_name)
    processor.filter_outliers(visualize=True)
    processor.downsample_voxel(voxel_size=0.1)
    processor.normalize_and_color()

    print('Displaying input point cloud: ')
    processor.visualize()
    processor.visualize_voxel_grid(voxel_size=0.05)



if __name__ == "__main__":
    main()
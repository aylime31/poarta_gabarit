import os
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt


class PointCloudProcessor:
    def __init__(self, pcd_name):
        self.pcd = o3d.io.read_point_cloud(pcd_name)

    def visualize(self, points=None):
        if points is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.visualization.draw_geometries([pcd])
        else:
            o3d.visualization.draw_geometries([self.pcd])

    def filter_outliers(self, nb_neighbors=15, std_ratio=2.0, visualize=False):
        #eliminam punctele care se afla la o distanta prea mare fata de majoritatea punctelor vecine, adica outliers
        #punctele sunt erori de masuratori, nu fac parte din obiect
        #nb_neighbors = numarul de vecini considerati pentru fiecare punct
        #std_ratio = factor care determina cat de departe poate fi un punct de vecinii sai
        #punctele care sunt la o distanta mai mare de std_ratio vor fi eliminate
        #remove_statistical_outiler calculeaza distanta medie si abaterea standard

        # ind este un vector NumPy ce contine indexul punctelor care se incadreaza in distanta
        _, ind = self.pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
        # actualizam norul de puncte cu punctele ramase
        self.pcd = self.pcd.select_by_index(ind)
        if visualize:
            self.visualize()

    # reducem numarul de puncte din norul de puncte prin aplicarea voxel gridului
    def downsample_voxel(self, voxel_size=0.01):
        self.pcd = self.pcd.voxel_down_sample(voxel_size)

    @staticmethod
    def rain_drop_remove(pcd, search_radius=0.3, std_dev_multiplier=1.0):
        #functie pentru eliminarea punctelor care sunt considerate picaturi de ploaie
        #search_radius = raza de cautare in jurul fiecarui punct pentru a gasi vecinii
        #std_dev_multiplier = controleaza variatia distantei fata de vecini a.i. sa fie considerat o picatura de ploaie

        # cream un arbore folosit la cautarea vecinilor
        points = np.asarray(pcd.points)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        removing = []
        mean_dists = []
        std_devs = []

        for i, point in enumerate(points):
            # cauta cei mai apropiati max_nn vecini in raza data
            [k, idx, _] = pcd_tree.search_radius_vector_3d(point, search_radius)
            # ocolim punctele cu mai putin de 3 vecini
            if k < 3:
                continue

            neighbors = points[idx, :]

            # lungimea ca diferenta dintre punctul curent si vecinul sau
            distances = np.linalg.norm(neighbors - point, axis=1)

            mean_dist = np.mean(distances)
            std_dev = np.std(distances)

            mean_dists.append(mean_dist)
            std_devs.append(std_dev)

            #verificam daca punctul este picatura de ploaie:
            threshold = std_dev_multiplier * mean_dist
            if std_dev > threshold:
                removing.append(i)

        """  #distributia distantelor pentru a ajusta pragurile
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(mean_dists, bins=50, color='blue', alpha=0.7)
        plt.title('Distributia distantelor medii')
        plt.xlabel('Distanta medie')
        plt.ylabel('Frecventa')

        plt.subplot(1, 2, 2)
        plt.hist(std_devs, bins=50, color='red', alpha=0.7)
        plt.title('Distributia deviatiilor standard')
        plt.xlabel('Deviatia standard')
        plt.ylabel('Frecventa')

        plt.tight_layout()
        plt.show() """

        without_rain_pcd = pcd.select_by_index(removing, invert=True)
        return without_rain_pcd

    def slice_point_cloud_by_points(self, num_slices, axis, output_dir):
        points_np = np.asarray(self.pcd.points, dtype=np.float64)

        #sortam punctele in functie de axa aleasa
        sorted_indices = np.argsort(points_np[:, axis])
        sorted_points = points_np[sorted_indices]

        num_points = len(points_np)
        points_per_slice = num_points // num_slices
        slice_files = []

        for i in range(num_slices):

            #felia 1 de la 0 la points_per_slice
            #felia 2 de la points_per_slice la 2*points_per_slice
            #felia 3 de la 2*points_per_slice la 3*points_per_slice etc
            start_idx = i*points_per_slice
            end_idx = (i+1)*points_per_slice

            #ultima felie va lua toate punctele ramase
            if i == num_slices - 1:
                end_idx = num_points

            slice_points = sorted_points[start_idx:end_idx]

            slice_pcd = o3d.geometry.PointCloud()
            slice_pcd.points = o3d.utility.Vector3dVector(slice_points)

            slice_filename = os.path.join(output_dir, f"slice_{i + 1}.pcd")
            o3d.io.write_point_cloud(slice_filename, slice_pcd)
            slice_files.append(slice_filename)

            print(f"Numarul de puncte in felia {i + 1}: {len(slice_points)}")

        return slice_files

    @staticmethod
    def calculate_slice_dimensions(file_paths):
        slice_dimensions = []
        for slice_idx, file_path in enumerate(file_paths):
            pcd_slice = o3d.io.read_point_cloud(file_path)
            points_np = np.asarray(pcd_slice.points)

            x_min, y_min, z_min = points_np.min(axis=0)
            x_max, y_max, z_max = points_np.max(axis=0)

            width = x_max - x_min
            height = z_max - z_min

            slice_dimensions.append({"width": width, "height": height})

        return slice_dimensions

    def detect_oversize(self, slices, width_limit, height_limit):
        total_height = 0

        all_points = np.asarray(self.pcd.points)
        all_colors = np.zeros((len(all_points), 3))

        for i, slice in enumerate(slices):
            slice_points = np.asarray(slice.points)
            x_min = slice_points[:, 0].min()
            x_max = slice_points[:, 0].max()
            z_min = slice_points[:, 2].min()
            z_max = slice_points[:, 2].max()

            width = x_max - x_min
            height = z_max - z_min

            total_height += height

            height_exceed = total_height > height_limit
            width_exceed = width > width_limit

            if height_exceed or width_exceed:
                if height_exceed and width_exceed:
                    #galben pentru depasirea ambelor limite
                    print(f"\nDepasire detectata in felia {i + 1}:")
                    print(f"  Width: {width:.2f}, Maximum width: {width_limit}")
                    print(f"  Accumulated height: {total_height:.2f}, Maximum height: {height_limit}")
                    color = [1, 1, 0]

                elif width_exceed:
                    #rosu pentru depasirea limitei de lațime
                    print(f"Depasire detectata in felia {i + 1}:")
                    print(f"  Width: {width:.2f}, Maximum width: {width_limit}")
                    color = [1, 0, 0]
                elif height_exceed:
                    #verde pentru depașirea limitei de inaltime
                    print(f"Depasire detectata in felia {i + 1}:")
                    print(f"  Accumulated height: {total_height:.2f}, Maximum height: {height_limit}")
                    color = [0, 1, 0]

                for point in slice_points:
                    idx = np.where(np.all(np.isclose(all_points, point), axis=1))[0]
                    if idx.size > 0:
                        all_colors[idx[0]] = color


        self.pcd.colors = o3d.utility.Vector3dVector(all_colors)
        self.visualize()


def analyze(director, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(director):
        if filename.endswith(".pcd"):
            file_path = os.path.join(director, filename)
            print(f"Processing file: {file_path}")

            file_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
            if not os.path.exists(file_output_dir):
                os.makedirs(file_output_dir)

            processor = PointCloudProcessor(file_path)

            print(f"Numarul de puncte initial: {len(processor.pcd.points)}")
            #eliminam picaturile de ploaie:
            processor.pcd = processor.rain_drop_remove(processor.pcd)
            print(f"Numarul de puncte dupa rain_drop_remove: {len(processor.pcd.points)}")

            #filtram zgomotul:
            processor.filter_outliers(visualize=True)
            print(f"Numarul de puncte dupa filter_outliers: {len(processor.pcd.points)}")

            processor.downsample_voxel(voxel_size=0.025)
            print(f"Numarul de puncte dupa voxel: {len(processor.pcd.points)}\n")

            #calculam dimensiunile inițiale:
            points_np = np.asarray(processor.pcd.points)
            x_min = points_np[:, 0].min()
            x_max = points_np[:, 0].max()

            z_min = points_np[:, 2].min()
            z_max = points_np[:, 2].max()

            width = x_max - x_min
            height = z_max - z_min

            num_slices = 5
            axis = 2  #alegem axa Z pentru a felia norul de puncte
            file_paths = processor.slice_point_cloud_by_points(num_slices, axis, file_output_dir)

            width_limit = 2.5
            height_limit = 4

            slice_dim = processor.calculate_slice_dimensions(file_paths)
            slices = [o3d.io.read_point_cloud(file_path) for file_path in file_paths]

            consistent_width = all(np.isclose(slice_dim[i]["width"], width) for i in range(num_slices))
            print(f"Latimea este constanta pe toate feliile: {consistent_width}, latimea initiala este: {width}")

            for i in range(num_slices):
                print(
                    f"Felia {i + 1} are latimea: {slice_dim[i]['width']:.2f} "
                    f"si inaltimea: {slice_dim[i]['height']:.2f}")

            total_height = sum(slice_dim[i]["height"] for i in range(num_slices))
            print(f"Suma inaltimilor feliilor: {total_height:.2f} (inaltimea originala: {height:.2f})")

            processor.detect_oversize(slices, width_limit, height_limit)



def main():
    input_director = "files"
    output_director = "slices"
    analyze(input_director, output_director)
#de facut: de stabilit daca trebuie sa elimin prima si ultima felie, detectare manuala


if __name__ == "__main__":
    main()

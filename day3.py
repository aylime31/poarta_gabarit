import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import RANSACRegressor


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

    def filter_outliers(self, nb_neighbors=20, std_ratio=2.0, visualize=False):
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
    def rain_drop_remove(pcd, search_radius=0.3, max_nn=90, std_dev_multiplier=1.0):
        #functie pentru eliminarea punctelor care sunt considerate picaturi de ploaie
        #search_radius = raza de cautare in jurul fiecarui punct pentru a gasi vecinii
        #max_nn = numarul maxim de vecini
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

        #distributia distantelor pentru a ajusta pragurile
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
        plt.show()

        without_rain_pcd = pcd.select_by_index(removing, invert=True)
        return without_rain_pcd

    def slice_point_cloud(self, num_slices, axis, output_dir):
        points_np = np.asarray(self.pcd.points)

        min_val = points_np[:, axis].min()
        max_val = points_np[:, axis].max()
        total_depth = max_val - min_val

        slice_depth = total_depth / num_slices
        slice_files = []

        for i in range(num_slices):
            lower_bound = min_val + i * slice_depth
            upper_bound = lower_bound + slice_depth

            #filtram punctele care se află în intervalul [lower_bound, upper_bound] pe axa specificată
            mask = (points_np[:, axis] >= lower_bound) & (points_np[:, axis] < upper_bound)
            slice_points = points_np[mask]

            #cream un nou nor de puncte pentru felie
            slice_pcd = o3d.geometry.PointCloud()
            slice_pcd.points = o3d.utility.Vector3dVector(slice_points)

            #salveaza felia
            slice_filename = f"{output_dir}/slice_{i + 1}.pcd"
            o3d.io.write_point_cloud(slice_filename, slice_pcd)
            slice_files.append(slice_filename)

        return slice_files



    def calculate_slice_dimensions(self, file_paths):
        slice_dimensions = []  #lista pentru a stoca dimensiunile
        for slice_idx, file_path in enumerate(file_paths):
            pcd_slice = o3d.io.read_point_cloud(file_path)
            points_np = np.asarray(pcd_slice.points)


            x_min, y_min, z_min = points_np.min(axis=0)
            x_max, y_max, z_max = points_np.max(axis=0)

            width = x_max - x_min
            height = z_max - z_min

            slice_dimensions.append({"width": width, "height": height})

        return slice_dimensions

    def detect_oversize(self, pcd_slice, threshold, axis):
        #pcd_slice = felia de nor de puncte
        #threshold = pragul

        points = np.asarray(pcd_slice.points)

        x = np.delete(points, axis, 1)
        y = points[:, axis]

        #cream modelul RANSAC
        ransac = RANSACRegressor(min_samples=3, residual_threshold=threshold)
        ransac.fit(x, y)

        #calcul erori
        errors = np.abs(ransac.predict(x) - y)

        #identificare depasiri
        inlier_mask = errors < threshold
        outlier_mask = np.logical_not(inlier_mask)
        oversize_indices = np.where(outlier_mask)[0]

        return oversize_indices

    def visualize_oversizes(self, pcd_slice, oversize_ind_width, oversize_ind_height):
        color_pcd = o3d.geometry.PointCloud()
        color_pcd.points = pcd_slice.points
        colors = np.asarray(color_pcd.colors)

        colors[oversize_ind_width] = [1, 0, 0]  #rosu pentru depasiri de latime
        colors[oversize_ind_height] = [0, 0, 1]  # albastru pentru depasiri de latime
        color_pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([color_pcd])


def main():
    pcd_name = "files/604-semitruck_tanker-00148.pcd"
    output = "slices"
    processor = PointCloudProcessor(pcd_name)
    processor.visualize()

    print(f"Numarul de puncte initial: {len(processor.pcd.points)}")
    #eliminam picaturile de ploaie:
    processor.pcd = processor.rain_drop_remove(processor.pcd)
    print(f"Numarul de puncte dupa rain_drop_remove: {len(processor.pcd.points)}")

    #filtram zgomotul:
    processor.filter_outliers(visualize=True)
    print(f"Numarul de puncte dupa filter_outliers: {len(processor.pcd.points)}")

    processor.downsample_voxel(voxel_size=0.05)
    print(f"Numarul de puncte dupa voxel: {len(processor.pcd.points)}\n")

    processor.visualize()

    points_np = np.asarray(processor.pcd.points)
    x_min = points_np[:, 0].min()
    x_max = points_np[:, 0].max()

    z_min = points_np[:, 2].min()
    z_max = points_np[:, 2].max()

    width = x_max - x_min
    height = z_max - z_min


    num_slices = 5
    axis = 2  #alegem axa Z pentru a felia norul de puncte
    file_paths = processor.slice_point_cloud( num_slices, axis, output)

    width_limit = 2.55
    height_limit = 4.0

    slice_dim = processor.calculate_slice_dimensions(file_paths)
    for slice_idx, file_path in enumerate(file_paths):
        pcd_slice = o3d.io.read_point_cloud(file_path)
        processor.visualize(np.asarray(pcd_slice.points))
        dimensions = slice_dim[slice_idx]




    consistent_width = all(np.isclose(slice_dim[i]["width"], width) for i in range(num_slices))
    print(f"Latimea este constanta pe toate feliile: {consistent_width}, latimea initiala este: {width}")

    for i in range(num_slices):
        print(f"Felia {i + 1} are latimea: {slice_dim[i]['width']:.2f} si inaltimea: {slice_dim[i]['height']:.2f}")


    total_height = sum(slice_dim[i]["height"] for i in range(num_slices))
    print(f"Suma inaltimilor feliilor: {total_height:.2f} (inaltimea originala: {height:.2f})")


if __name__ == "__main__":
    main()

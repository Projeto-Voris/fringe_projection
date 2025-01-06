import numpy as np
import cupy as cp
import yaml
import matplotlib.pyplot as plt
import time
import gc
from cupyx.fallback_mode.fallback import ndarray
import open3d as o3d

class InverseTriangulation:
    def __init__(self, yaml_file):

        self.yaml_file = yaml_file
        self.left_images = ndarray([])
        self.right_images = ndarray([])
        self.left_mask = ndarray([])
        self.right_mask = ndarray([])

        # Initialize all camera parameters in a single nested dictionary
        self.camera_params = {
            'left': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'right': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'stereo': {'R': np.array([]), 'T': np.array([])}
        }

        self.read_yaml_file()

        self.z_scan_step = None
        self.num_points = None
        self.max_gpu_usage = self.set_datalimit() // 3

        # self.uv_left = []
        # self.uv_right = []

    def read_images(self, left_imgs, right_imgs, left_mask, right_mask):
        if len(left_imgs) != len(right_imgs):
            raise Exception("Number of images do not match")
        self.left_images = cp.asarray(left_imgs)
        self.right_images = cp.asarray(right_imgs)
        self.left_mask = cp.asarray(left_mask)
        self.right_mask = cp.asarray(right_mask)

    # def points3d(self, x_lim=(-5, 5), y_lim=(-5, 5), z_lim=(0, 5), xy_step=1.0, z_step=1.0, visualize=False):
    #     """
    #         Create a 3D space of combination from linear arrays of X Y Z
    #         Parameters:
    #             x_lim: Begin and end of linear space of X
    #             y_lim: Begin and end of linear space of Y
    #             z_lim: Begin and end of linear space of Z
    #             xy_step: Step size between X and Y
    #             z_step: Step size between Z and X
    #             visualize: Visualize the 3D space
    #         Returns:
    #             cube_points: combination of X Y and Z
    #         """
    #     x_lin = np.arange(x_lim[0], x_lim[1], xy_step)
    #     y_lin = np.arange(y_lim[0], y_lim[1], xy_step)
    #     z_lin = np.arange(z_lim[0], z_lim[1], z_step)
    #
    #     mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
    #
    #     c_points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)
    #
    #     if visualize:
    #         self.plot_3d_points(x=c_points[:, 0], y=c_points[:, 1], z=c_points[:, 2])
    #
    #     self.num_points = c_points.shape[0]
    #     self.z_scan_step = np.unique(c_points[:, 2]).shape[0]
    #
    #     return c_points.astype(np.float16)

    def points3d_zstep(self, x_lim=(-5, 5), y_lim=(-5, 5), xy_step=1.0, z_lin=np.arange(0, 100, 0.1), visualize=False):
        """
            Create a 3D space of combination from linear arrays of X Y Z
            Parameters:
                x_lim: Begin and end of linear space of X
                y_lim: Begin and end of linear space of Y
                z_lin: numpy array of z to be tested
                xy_step: Step size between X and Y
                visualize: Visualize the 3D space
            Returns:
                cube_points: combination of X Y and Z
        """
        x_lin = np.arange(x_lim[0], x_lim[1], xy_step)
        y_lin = np.arange(y_lim[0], y_lim[1], xy_step)

        mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')

        c_points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)

        if visualize:
            self.plot_3d_points(x=c_points[:, 0], y=c_points[:, 1], z=c_points[:, 2])

        self.num_points = c_points.shape[0]
        self.z_scan_step = np.unique(c_points[:, 2]).shape[0]

        return c_points

    def points3D_arrays(self, x_lin: ndarray, y_lin: ndarray, z_lin: ndarray, visualize: bool = True) -> ndarray:
        """
        Crete 3D meshgrid of points based on input vectors of x, y and z
        :param x_lin: linear space of x points
        :param y_lin: linear space of y points
        :param z_lin: linear space of z points
        :param visualize: If true plot a 3d graph of points
        :return: 3D meshgrid points size (N,3) where N = len(x)*len(y)*len(z)
        """
        mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
        points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)

        if visualize:
            self.plot_3d_points(x=points[:, 0], y=points[:, 1], z=points[:, 2])

        self.num_points = points.shape[0]
        self.z_scan_step = np.unique(points[:, 2]).shape[0]

        return cp.asarray(points)

    def plot_3d_points(self, x, y, z, color=None, title='Plot 3D of max correlation points'):
        """
        Plot 3D points as scatter points where color is based on Z value
        Parameters:
            x: array of x positions
            y: array of y positions
            z: array of z positions
            color: Vector of point intensity grayscale
        """
        # x = x.get()
        # y = y.get()
        # z = z.get()

        if color is None:
            color = z
        cmap = 'viridis'
        # Plot the 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.title.set_text(title)

        scatter = ax.scatter(x, y, z, c=color, cmap=cmap, marker='o')
        # ax.set_zlim(0, np.max(z))
        colorbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        colorbar.set_label('Z Value Gradient')

        # Add labels
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    def read_yaml_file(self):
        """
        Read YAML file to extract cameras parameters
        """
        # Load the YAML file
        with open(self.yaml_file) as file:  # Replace with your file path
            params = yaml.safe_load(file)

            # Parse the matrices
        self.camera_params['left']['kk'] = np.array(params['camera_matrix_left'], dtype=np.float64)
        self.camera_params['left']['kc'] = np.array(params['dist_coeffs_left'], dtype=np.float64)
        self.camera_params['left']['r'] = np.array(params['rot_matrix_left'], dtype=np.float64)
        self.camera_params['left']['t'] = np.array(params['t_left'], dtype=np.float64)

        self.camera_params['right']['kk'] = np.array(params['camera_matrix_right'], dtype=np.float64)
        self.camera_params['right']['kc'] = np.array(params['dist_coeffs_right'], dtype=np.float64)
        self.camera_params['right']['r'] = np.array(params['rot_matrix_right'], dtype=np.float64)
        self.camera_params['right']['t'] = np.array(params['t_right'], dtype=np.float64)

        # self.camera_params['stereo']['R'] = np.array(params['R'], dtype=np.float64)
        # self.camera_params['stereo']['T'] = np.array(params['T'], dtype=np.float64)

    def save_points(self, data, filename, delimiter=','):
        """
        Save a 2D NumPy array to a CSV file.

        :param array: 2D numpy array
        :param filename: Output CSV filename
        """
        # Save the 2D array as a CSV file
        np.savetxt(filename, data, delimiter=delimiter)
        print(f"Array saved to {filename}")

    def set_datalimit(self):
        """
        Identify gpu limit
        """
        # Create a device object for the first GPU (device ID 0)
        device_id = 0
        cp.cuda.Device(device_id).use()  # Set the current device
        # Get the total memory in bytes using runtime API
        total_memory = cp.cuda.runtime.getDeviceProperties(device_id)['totalGlobalMem']
        # Convert bytes to GB
        return total_memory / (1024 ** 3)

    def transform_gcs2ccs(self, points_3d, cam_name):
        """
        Transform Global Coordinate System (xg, yg, zg)
         to Camera's Coordinate System (xc, yc, zc) and transform to Image's plane (uv)
         Returns:
             uv_image_points: (2,N) reprojected points to image's plane
        """
        # Convert all inputs to CuPy arrays for GPU computation
        xyz_gcs = cp.asarray(points_3d)
        k = cp.asarray(self.camera_params[cam_name]['kk'])
        dist = cp.asarray(self.camera_params[cam_name]['kc'])
        rot = cp.asarray(self.camera_params[cam_name]['r'])
        tran = cp.asarray(self.camera_params[cam_name]['t'])

        # Estimate the size of the input and output arrays
        # num_points = xyz_gcs.shape[0]
        bytes_per_float32 = 8  # Simulate double-precision float usage

        # Estimate the memory required per point for transformation and intermediate steps
        memory_per_point = (4 * 3 * bytes_per_float32) + (3 * bytes_per_float32)  # For xyz_gcs_1 and xyz_ccs
        total_memory_required = self.num_points * memory_per_point

        # Adjust the batch size based on memory limitations
        if total_memory_required > self.max_gpu_usage * 1024 ** 3:
            points_per_batch = int(
                (self.max_gpu_usage * 1024 ** 3 // memory_per_point) // 10)  # Reduce batch size more aggressively
            # print(f"Processing {points_per_batch} points per batch due to memory limitations.")
        else:
            points_per_batch = self.num_points  # Process all points at once

        # Initialize an empty list to store results (on the CPU)
        uv_points_list = []

        # Process points in batches
        for i in range(0, self.num_points, points_per_batch):
            end = min(i + points_per_batch, self.num_points)
            xyz_gcs_batch = xyz_gcs[i:end]

            # Debug: Check the shape of the batch
            # print(f"Processing batch {i // points_per_batch + 1}, size: {xyz_gcs_batch.shape}")

            # Add one extra line of ones to the global coordinates
            ones = cp.ones((xyz_gcs_batch.shape[0], 1), dtype=cp.float16)  # Double-precision floats
            xyz_gcs_1 = cp.hstack((xyz_gcs_batch, ones))

            # Create the rotation and translation matrix
            rt_matrix = cp.vstack(
                (cp.hstack((rot, tran[:, None])), cp.array([0, 0, 0, 1], dtype=cp.float16))
            )

            # Multiply the RT matrix with global points [X; Y; Z; 1]
            xyz_ccs = cp.dot(rt_matrix, xyz_gcs_1.T)
            del xyz_gcs_1  # Immediately delete

            # Normalize by dividing by Z to get normalized image coordinates
            epsilon = 1e-10  # Small value to prevent division by zero
            xyz_ccs_norm = cp.hstack(
                (xyz_ccs[:2, :].T / cp.maximum(xyz_ccs[2, :, cp.newaxis], epsilon),
                 cp.ones((xyz_ccs.shape[1], 1), dtype=cp.float16))
            ).T
            del xyz_ccs  # Immediately delete

            # Apply distortion using the GPU
            xyz_ccs_norm_dist = self.undistorted_points(xyz_ccs_norm.T, dist)
            del xyz_ccs_norm  # Free memory

            # Compute image points using the intrinsic matrix K
            uv_points_batch = cp.dot(k, xyz_ccs_norm_dist.T)
            del xyz_ccs_norm_dist  # Free memory

            # Debug: Check the shape of the result
            # print(f"uv_points_batch shape: {uv_points_batch.shape}")

            # Transfer results back to CPU after processing each batch
            uv_points_list.append(cp.asnumpy(uv_points_batch))

            # Free GPU memory after processing each batch
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        # Ensure consistent dimensions when concatenating batches
        try:
            # Concatenate all batches along axis 0 (rows)
            uv_points = cp.hstack(uv_points_list)  # Use np.hstack for matching shapes

        except ValueError as e:
            print(f"Error during concatenation: {e}")
            raise

        return uv_points[:2, :].astype(cp.float16)

    def undistorted_points(self, points, dist):
        """
        GPU version of the undistorted points function using CuPy.
        Applies radial and tangential distortion.

        Parameters:
            points: 2D array of normalized image coordinates [x, y].
            dist: Distortion coefficients [k1, k2, p1, p2, k3].

        Returns:
            Distorted points on the GPU.
        """
        # Extract distortion coefficients
        k1, k2, p1, p2, k3 = dist

        # Split points into x and y coordinates
        x, y = points[:, 0], points[:, 1]

        # Calculate r^2 (squared distance from the origin)
        r2 = x ** 2 + y ** 2

        # Radial distortion
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3

        # Tangential distortion
        x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
        y_tangential = p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y

        # Compute distorted coordinates
        x_distorted = x * radial + x_tangential
        y_distorted = y * radial + y_tangential

        # Stack the distorted points
        distorted_points = cp.vstack([x_distorted, y_distorted, cp.ones_like(x)]).T

        # Clean up intermediate variables to free memory
        del x, y, r2, radial, x_tangential, y_tangential
        cp.get_default_memory_pool().free_all_blocks()

        return distorted_points

    def bi_interpolation(self, images, uv_points, window_size=3):
        """
        Perform bilinear interpolation on a stack of images at specified uv_points on the GPU.

        Parameters:
        ----------
        images : (height, width, num_images) array or (height, width) for a single image.
        uv_points : (2, N) array of UV points where N is the number of points.
        window_size : int
            Unused here but kept for consistency with spatial functions.

        Returns:
        -------
        interpolated_cpu : np.ndarray
            Interpolated pixel values for each point.
        std_cpu : np.ndarray
            Standard deviation of the corner pixels used for interpolation.
        """
        images = cp.asarray(images)
        uv_points = cp.asarray(uv_points)

        if len(images.shape) == 2:  # Convert single image to a stack with one image
            images = images[:, :, cp.newaxis]

        height, width, num_images = images.shape

        # Estimate memory usage per point
        memory_per_point = 4 * num_images * 4
        points_per_batch = max(1, int(self.max_gpu_usage * 1024 ** 3 // memory_per_point))

        # Output arrays on GPU
        interpolated = cp.zeros((self.num_points, num_images), dtype=cp.float16)
        std = cp.zeros((self.num_points, num_images), dtype=cp.float16)

        for i in range(0, self.num_points, points_per_batch):
            end = min(i + points_per_batch, self.num_points)
            uv_batch = uv_points[:, i:end]

            # Compute integer and fractional parts of UV coordinates
            x = uv_batch[0].astype(cp.int32)
            y = uv_batch[1].astype(cp.int32)

            x1 = cp.clip(cp.floor(x).astype(cp.int32), 0, width - 1)
            y1 = cp.clip(cp.floor(y).astype(cp.int32), 0, height - 1)
            x2 = cp.clip(x1 + 1, 0, width - 1)
            y2 = cp.clip(y1 + 1, 0, height - 1)

            x_diff = x - x1
            y_diff = y - y1
            for k in range(num_images):
                # Vectorized extraction of corner pixels
                p11 = images[y1, x1, k]  # Top-left
                p12 = images[y2, x1, k]  # Bottom-left
                p21 = images[y1, x2, k]  # Top-right
                p22 = images[y2, x2, k]  # Bottom-right

                # Bilinear interpolation
                interpolated_batch = (
                        p11 * (1 - x_diff) * (1 - y_diff) +
                        p21 * x_diff * (1 - y_diff) +
                        p12 * (1 - x_diff) * y_diff +
                        p22 * x_diff * y_diff
                )

                std_batch = cp.std(cp.vstack([p11, p12, p21, p22]), axis=0)

                # Store results in GPU arrays
                interpolated[i:end, k] = interpolated_batch
                std[i:end, k] = std_batch
            del p11, p12, p21, p22, std_batch, interpolated_batch
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        # Convert results to CPU
        # interpolated_cpu = cp.asnumpy(interpolated_gpu)
        # std_cpu = cp.asnumpy(std_gpu)

        if images.shape[2] == 1:  # Flatten output for single image
            interpolated_cpu = interpolated[:, 0]
            std_cpu = std[:, 0]

        return interpolated, std

    def phase_map(self,interp_left, interp_right, debug=False):
        """
        Identify minimum phase map value
        Parameters:
            interp_left: left interpolated points
            interp_right: right interpolated points
            debug: if true, visualize phi_map array
        Returns:
            phi_min_id: indices of minimum phase map values.
        """
        phi_map = []
        phi_min_id = []

        for k in range(self.num_points // self.z_scan_step):
            diff_phi = np.abs(interp_left[self.z_scan_step * k:(k + 1) * self.z_scan_step]
                              - interp_right[self.z_scan_step * k:(k + 1) * self.z_scan_step])
            phi_min_id.append(np.argmin(diff_phi) + k * self.z_scan_step)
            if debug:
                phi_map.append(diff_phi)
        if debug:
            plt.figure()
            plt.plot(phi_map)
            plt.show()

        return phi_min_id

    def fringe_masks(self, uv_l, uv_r, std_l, std_r, phi_id, min_thresh=-0.01, max_thresh=1.01):
        """
        Mask from fringe process to remove outbounds points.
        Paramenters:
            std_l: STD interpolation image's points
            std_r: STD interpolation image's points
            phi_id: Indices for min phase difference
            min_thresh: max threshold for STD
            max_thresh: min threshold for STD
        Returns:
             valid_mask: Valid 3D points on image's plane
        """
        # Verifica se as coordenadas estão dentro dos limites das máscaras
        valid_u_l = (uv_l[0, :] >= 0) & (uv_l[0, :] < self.left_mask.shape[1])
        valid_v_l = (uv_l[1, :] >= 0) & (uv_l[1, :] < self.left_mask.shape[0])
        valid_u_r = (uv_r[0, :] >= 0) & (uv_r[0, :] < self.right_mask.shape[1])
        valid_v_r = (uv_r[1, :] >= 0) & (uv_r[1, :] < self.right_mask.shape[0])

        # Aplica as verificações de validade nas coordenadas UV para evitar indexação fora do limite
        valid_uv_l = valid_u_l & valid_v_l
        valid_uv_r = valid_u_r & valid_v_r

        # Verifica os pontos válidos nas máscaras (aplica as coordenadas para obter as máscaras)
        valid_uv_l &= (self.left_mask[uv_l[1, :].clip(0, self.left_mask.shape[0] - 1).astype(int),
                uv_l[0, :].clip(0, self.left_mask.shape[1] - 1).astype(int)] > 0)

        valid_uv_r &= (self.right_mask[uv_r[1, :].clip(0, self.right_mask.shape[0] - 1).astype(int),
                uv_r[0, :].clip(0, self.right_mask.shape[1] - 1).astype(int)] > 0)

        # Combine as verificações dos limites
        valid_uv = valid_uv_r & valid_uv_l

        # Máscara para `phi_id`
        phi_mask = cp.zeros(uv_l.shape[1], dtype=bool)
        phi_mask[phi_id] = True

        # Verificação dos thresholds de `std` para pontos válidos
        valid_l = (min_thresh < std_l) & (std_l < max_thresh)
        valid_r = (min_thresh < std_r) & (std_r < max_thresh)
        valid_std = valid_r[:, 0] & valid_l[:, 0]

        # Retorne a máscara final considerando os pontos válidos em `uv`, `phi` e `std`
        mask = valid_uv & phi_mask & valid_std
        return mask

    def fringe_process(self, points_3d: ndarray, save_points: bool = True, visualize: bool = False) -> ndarray:
        """
        Zscan for stereo fringe process
        Parameters:
            save_points: boolean to save or not image
            visualize: boolean to visualize result
        :return:
            measured_pts: Valid 3D global coordinate points
        """
        t0 = time.time()
        uv_left = self.transform_gcs2ccs(points_3d, cam_name='left')
        uv_right = self.transform_gcs2ccs(points_3d, cam_name='right')
        inter_left, std_left = self.bi_interpolation(self.left_images, uv_left)
        inter_right, std_right = self.bi_interpolation(self.right_images, uv_right)

        phi_min_id = self.phase_map(inter_left, inter_right)
        fringe_mask = self.fringe_masks(uv_l = uv_left, uv_r = uv_right, std_l = std_left, std_r = std_right, phi_id = phi_min_id)
        measured_pts = points_3d[fringe_mask]

        print('Zscan result dt: {} s'.format(round(time.time() - t0), 2))

        if save_points:
            self.save_points(measured_pts, filename='./sm3_duto.csv')

        if visualize:
            self.plot_3d_points(measured_pts[:, 0], measured_pts[:, 1], measured_pts[:, 2], color=None,
                                title="Fringe process output points")

        return measured_pts

    def filter_points_by_depth(self, points, depth_threshold=0.05):
        # Se 'points' for um array Cupy
        if isinstance(points, cp.ndarray):
            points = points.get()  # Converte para NumPy

        # Converte o numpy array para um objeto PointCloud do Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Cria a octree com base na profundidade, qunado maior mais divisão
        octree = o3d.geometry.Octree(max_depth=20)
        octree.convert_from_point_cloud(pcd, size_expand=0.01)

        filtered_points = []

        # Função para processar os blocos da octree
        def process_leaf(node, node_info):
            # Verifica se o nó atual é um bloco
            if isinstance(node, o3d.geometry.OctreeLeafNode):
                # Obtém os pontos do nó
                points_in_leaf = np.asarray([pcd.points[idx] for idx in node.indices])

                # Calcula a profundidade média e desvio padrão da coordenada Z
                mean_depth = np.mean(points_in_leaf[:, 2])
                std_depth = np.std(points_in_leaf[:, 2])

                # Filtra os pontos com base na profundidade media de cada bloco
                for point in points_in_leaf:
                    if np.abs(point[2] - mean_depth) <= depth_threshold:
                        filtered_points.append(point)

        # Processa todos os nós da octree
        octree.traverse(process_leaf)

        # Cria uma nova nuvem de pontos com os pontos filtrados
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        filtered_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=1.0)

        return filtered_pcd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import cv2

class inverse_triangulation():
    def __init__(self):
        pass

    def points3d(self, x_lim, y_lim, z_lim, xy_step, z_step, visualize=True):
        x_lin = np.arange(x_lim[0], x_lim[1], xy_step)
        y_lin = np.arange(y_lim[0], y_lim[1], xy_step)
        z_lin = np.arange(z_lim[0], z_lim[1], z_step)

        mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')

        c_points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)

        if visualize:
            self.plot_3d_points(x=c_points[:, 0], y=c_points[:, 1], z=c_points[:, 2])

        return c_points

    def plot_3d_points(self, x, y, z, color=None, title='Plot 3D of max correlation points'):
        """
        Plot 3D points as scatter points where color is based on Z value
        Parameters:
            x: array of x positions
            y: array of y positions
            z: array of z positions
            color: Vector of point intensity grayscale
        """
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

    def load_camera_params(self, yaml_file):
        # Load the YAML file
        with open(yaml_file) as file:  # Replace with your file path
            params = yaml.safe_load(file)

            # Parse the matrices
        Kl = np.array(params['camera_matrix_left'], dtype=np.float64)
        Dl = np.array(params['dist_coeffs_left'], dtype=np.float64)
        Rl = np.array(params['rot_matrix_left'], dtype=np.float64)
        Tl = np.array(params['t_left'], dtype=np.float64)

        Kr = np.array(params['camera_matrix_right'], dtype=np.float64)
        Dr = np.array(params['dist_coeffs_right'], dtype=np.float64)
        Rr = np.array(params['rot_matrix_right'], dtype=np.float64)
        Tr = np.array(params['t_right'], dtype=np.float64)

        # R = np.array(params['R'], dtype=np.float64)
        # T = np.array(params['T'], dtype=np.float64)

        return Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr

    def load_array_from_csv(self, filename):
        """
        Load a 2D NumPy array from a CSV file.

        :param filename: Input CSV filename
        :return: 2D numpy array
        """
        # Load the array from the CSV file
        array = np.loadtxt(filename, delimiter=',')
        return array

    def bi_interpolation(self, images, uv_points, batch_size=10000):
        """
        Perform bilinear interpolation on a stack of images at specified uv_points, optimized for memory.

        Parameters:
        images: (height, width, num_images) array of images, or (height, width) for a single image.
        uv_points: (2, N) array of points where N is the number of points.
        batch_size: Maximum number of points to process at once (default 10,000 for memory efficiency).

        Returns:
        interpolated: (N, num_images) array of interpolated pixel values, or (N,) for a single image.
        std: Standard deviation of the corner pixels used for interpolation.
        """
        if len(images.shape) == 2:
            # Convert 2D image to 3D for consistent processing
            images = images[:, :, np.newaxis]

        height, width, num_images = images.shape
        N = uv_points.shape[1]

        # Initialize the output arrays
        interpolated = np.zeros((N, num_images))
        std = np.zeros(N)

        # Process points in batches
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            uv_batch = uv_points[:, i:end]

            x = uv_batch[0].astype(float)
            y = uv_batch[1].astype(float)

            # Ensure x and y are within bounds
            x1 = np.clip(np.floor(x).astype(int), 0, width - 1)
            y1 = np.clip(np.floor(y).astype(int), 0, height - 1)
            x2 = np.clip(x1 + 1, 0, width - 1)
            y2 = np.clip(y1 + 1, 0, height - 1)

            # Calculate the differences
            x_diff = x - x1
            y_diff = y - y1

            # Bilinear interpolation in batches (vectorized)
            for n in range(num_images):
                p11 = images[y1, x1, n]  # Top-left corner
                p12 = images[y2, x1, n]  # Bottom-left corner
                p21 = images[y1, x2, n]  # Top-right corner
                p22 = images[y2, x2, n]  # Bottom-right corner

                # Bilinear interpolation formula (for each batch)
                interpolated_batch = (
                        p11 * (1 - x_diff) * (1 - y_diff) +
                        p21 * x_diff * (1 - y_diff) +
                        p12 * (1 - x_diff) * y_diff +
                        p22 * x_diff * y_diff
                )
                interpolated[i:end, n] = interpolated_batch

        # Return 1D interpolated result if the input was a 2D image
        if images.shape[2] == 1:
            interpolated = interpolated[:, 0]
        std = np.zeros_like((uv_points.shape[0], images.shape[2]))
        return interpolated, std

    def phase_map(self, left_Igray, right_Igray, points_3d):
        z_step = np.unique(points_3d[:, 2]).shape[0]
        phi_map = []
        phi_min = []
        phi_min_id = []
        for k in range(points_3d.shape[0] // z_step):
            diff_phi = np.abs(left_Igray[z_step * k:(k + 1) * z_step] - right_Igray[z_step * k:(k + 1) * z_step])
            phi_map.append(diff_phi)
            phi_min.append(np.nanmin(diff_phi))
            phi_min_id.append(np.argmin(diff_phi) + k * z_step)

        return phi_map, phi_min, phi_min_id

    def undistorted_points(self, norm_points, kc):
        r2 = norm_points[:, 0] ** 2 + norm_points[:, 1] ** 2
        k1, k2, p1, p2, k3 = kc
        factor = (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)
        x_corrected = norm_points[:, 0] * factor + 2 * p1 * norm_points[:, 0] * norm_points[:, 1] + p2 * (
                r2 + 2 * norm_points[:, 0] ** 2)
        y_corrected = norm_points[:, 1] * factor + p1 * (r2 + 2 * norm_points[:, 1] ** 2) + 2 * p2 * norm_points[:, 0] * norm_points[:, 1]

        return np.hstack((np.stack([x_corrected, y_corrected], axis=-1), np.ones((norm_points.shape[0], 1))))

    def gcs2ccs(self, xyz_gcs, k, dist, rot, tran):
        """
           Transform Global Coordinate System (GCS) to Camera Coordinate System (CCS).
           Parameters:
               xyz_gcs (array): Global coordinate system coordinates [X, Y, Z]
               k: Intrinsic matrix
               dist: Distortion vector [k1, k2, p1, p2, k3]
               rot: Rotation matrix
               tran: Translation vector
           Returns:
               uv_points: Image points
           """
        # add one extra linhe of ones
        xyz_gcs_1 = np.hstack((xyz_gcs, np.ones((xyz_gcs.shape[0], 1))))
        # rot matrix and trans vector from gcs to ccs
        rt_matrix = np.vstack(
            (np.hstack((rot, tran[:, None])), [0, 0, 0, 1]))
        # Multiply rotation and translation matrix to global points [X; Y; Z; 1]
        xyz_ccs = np.dot(rt_matrix, xyz_gcs_1.T)
        # Normalize by dividing by Z to get normalized image coordinates
        epsilon = 1e-10  # Small value to prevent division by zero
        xyz_ccs_norm = np.hstack((xyz_ccs[:2, :].T / np.maximum(xyz_ccs[2, :, np.newaxis], epsilon),
                                  np.ones((xyz_ccs.shape[1], 1)))).T
        # remove distortion from lens
        xyz_ccs_norm_undist = self.undistorted_points(xyz_ccs_norm.T, dist)

        # Compute image's point as intrinsic K to XYZ CCS points normalized and undistorted
        uv_points = np.dot(k, xyz_ccs_norm_undist.T)
        return uv_points

    def plot_zscan_phi(self, phi_map):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
        plt.title('Z step for abs(phase_left - phase_right)')
        for j in range(len(phi_map)):
            if j < len(phi_map) // 2:
                ax1.plot(phi_map[j], label="{}".format(j))
                circle = plt.Circle((np.argmin(phi_map[j]), np.min(phi_map[j])), 0.05, color='r', fill=False, lw=2)
                ax1.add_patch(circle)  # Add circle to the plot
                ax1.set_ylabel('correlation [%]')
                ax1.set_xlabel('z steps')
                ax1.grid(True)
                # ax1.legend()
            if j >= len(phi_map) // 2:
                ax2.plot(phi_map[j], label="{}".format(j))
                circle = plt.Circle((np.argmin(phi_map[j]), np.min(phi_map[j])), 0.05, color='r', fill=False, lw=2)
                ax2.add_patch(circle)  # Add circle to the plot
                ax2.set_xlabel('z steps')
                ax2.set_ylabel('correlation [%]')
                ax2.grid(True)
                # ax2.legend()
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()

    def plot_points_3d_on_image(self, image, points, color=(0, 255, 0), radius=5, thickness=2):
        output_image = cv2.cvtColor(np.uint8(image), cv2.COLOR_GRAY2BGR)
        for (u, v, _) in points.T:
            cv2.circle(output_image, (int(u), int(v)), radius, color, thickness)

        return output_image

    def show_stereo_image(self, left, right, name='Rectified image'):
        combined_image = np.concatenate((left, right), axis=1)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, int(combined_image.shape[1] / 4), int(combined_image.shape[0] / 4))
        cv2.imshow(name, combined_image)
        cv2.waitKey(0)

    def fringe_masks(self, img_l, img_r, uv_l, uv_r, std_l, std_r, phi_id, min_thresh=0, max_thresh=1):
        valid_u_l = (uv_l[0, :] >= 0) & (uv_l[0, :] < img_l.shape[1])
        valid_v_l = (uv_l[1, :] >= 0) & (uv_l[1, :] < img_l.shape[0])
        valid_u_r = (uv_r[0, :] >= 0) & (uv_r[0, :] < img_r.shape[1])
        valid_v_r = (uv_r[1, :] >= 0) & (uv_r[1, :] < img_r.shape[0])
        valid_uv = valid_u_l & valid_u_r & valid_v_l & valid_v_r
        phi_mask = np.zeros(uv_l.shape[1], dtype=bool)
        phi_mask[phi_id] = True
        valid_l = (min_thresh < std_l) & (std_l < max_thresh)
        valid_r = (min_thresh < std_r) & (std_r < max_thresh)

        valid_std = valid_r & valid_l

        print("valid_uv shape:", valid_uv.shape)
        print("valid_std shape:", valid_std.shape)
        print("phi_mask shape:", phi_mask.shape)
        print("std_l shape:", std_l.shape)
        print("std_r shape:", std_r.shape)

        return valid_uv & valid_std & phi_mask

    def fringe_zscan(self, left_images, right_images, points_3d, yaml_file, DEBUG=False, SAVE=True):
    # def fringe_zscan(self, points_3d, yaml_file, DEBUG=False, SAVE=True):

        # left_images = []
        # right_images = []
        #
        # for abs_image_left_32_20241016, abs_image_right_32_20241016 in zip(sorted(os.listdir('csv/left')), sorted(os.listdir('csv/right'))):
        #     left_images.append(self.load_array_from_csv(os.path.join('csv/left', abs_image_left_32_20241016)))
        #     right_images.append(self.load_array_from_csv(os.path.join('csv/right', abs_image_right_32_20241016)))
        #
        # left_images = np.stack(left_images, axis=-1).astype(np.float32)
        # right_images = np.stack(right_images, axis=-1).astype(np.float32)

        # Read file containing all calibration parameters from stereo system
        Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr = self.load_camera_params(yaml_file=yaml_file)

        # Project points on Left and right
        uv_points_L = self.gcs2ccs(points_3d, Kl, Dl, Rl, Tl)
        uv_points_R = self.gcs2ccs(points_3d, Kr, Dr, Rr, Tr)

        # Interpolate reprojected points to image bounds (return pixel intensity)
        inter_points_L, std_interp_L = self.bi_interpolation(left_images, uv_points_L)
        inter_points_R, std_interp_R = self.bi_interpolation(right_images, uv_points_R)

        phi_map, phi_min, phi_min_id = self.phase_map(inter_points_L, inter_points_R, points_3d)

        fringe_mask = self.fringe_masks(img_l=left_images, img_r=right_images, uv_l=uv_points_L, uv_r=uv_points_R,
                               std_l=std_interp_L, std_r=std_interp_R, phi_id=phi_min_id)

        filtered_3d_phi = points_3d[np.asarray(phi_min_id, np.int32)]

        filtered_mask = points_3d[fringe_mask]

        if DEBUG:
            self.plot_zscan_phi(phi_map=phi_map)

            reproj_l = self.plot_points_3d_on_image(left_images[:, :, 0], uv_points_L)
            reproj_r = self.plot_points_3d_on_image(right_images[:, :, 0], uv_points_R)
            self.show_stereo_image(reproj_l, reproj_r, 'Reprojected points o image')
            cv2.destroyAllWindows()

        if SAVE:
            np.savetxt('./fringe_points.txt', filtered_3d_phi, delimiter='\t', fmt='%.3f')

        self.plot_3d_points(filtered_3d_phi[:, 0], filtered_3d_phi[:, 1], filtered_3d_phi[:, 2], color=None,
                                      title="Point Cloud of min phase diff")

        self.plot_3d_points(filtered_mask[:, 0], filtered_mask[:, 1], filtered_mask[:, 2], color=None,
                            title="Fringe Mask")
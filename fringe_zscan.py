import os
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import fringe_process
import Distortion_correction

def bi_interpolation(images, uv_points, batch_size=10000):
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

            # # Compute standard deviation across the four corners for each point
            # std_batch = np.std(np.vstack([p11, p12, p21, p22]), axis=0)
            # std[i:end] = std_batch

    # Return 1D interpolated result if the input was a 2D image
    if images.shape[2] == 1:
        interpolated = interpolated[:, 0]
    std = np.zeros_like((uv_points.shape[0], images.shape[2]))
    return interpolated, std

def phase_map(left_Igray, right_Igray, points_3d):
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

def load_array_from_csv(filename):
    """
    Load a 2D NumPy array from a CSV file.

    :param filename: Input CSV filename
    :return: 2D numpy array
    """
    # Load the array from the CSV file
    array = np.loadtxt(filename, delimiter=',')
    return array

def plot_zscan_phi(phi_map):
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

def fringe_zscan(points_3d, yaml_file, DEBUG=False, SAVE=True):
    t0 = time.time()

    left_images = []
    right_images = []

    for abs_image_left_32, abs_image_right_32 in zip(sorted(os.listdir('csv/left')), sorted(os.listdir('csv/right'))):
        left_images.append(load_array_from_csv(os.path.join('csv/left', abs_image_left_32)))
        right_images.append(load_array_from_csv(os.path.join('csv/right', abs_image_right_32)))

    left_images = np.stack(left_images, axis=-1).astype(np.float32)
    right_images = np.stack(right_images, axis=-1).astype(np.float32)

    t1 = time.time()
    print('Got {} left and right images: \n dt: {} s'.format(right_images.shape[2] + left_images.shape[2], round((t1 - t0), 2)))

    # Read file containing all calibration parameters from stereo system
    Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr, R, T = fringe_process.load_camera_params(yaml_file=yaml_file)

    # Project points on Left and right
    uv_points_L = Distortion_correction.gcs2ccs(points_3d, Kl, Dl, Rl, Tl)
    uv_points_R = Distortion_correction.gcs2ccs(points_3d, Kr, Dr, Rr, Tr)

    t3 = time.time()
    print('Project points \n dt: {} s'.format(round((t3 - t1), 2)))

    # Interpolate reprojected points to image bounds (return pixel intensity)
    inter_points_L, std_interp_L = bi_interpolation(left_images, uv_points_L)
    inter_points_R, std_interp_R = bi_interpolation(right_images, uv_points_R)

    t4 = time.time()
    print("Interpolate \n dt: {} s".format(round((t4 - t3), 2)))

    phi_map, phi_min, phi_min_id = phase_map(inter_points_L, inter_points_R, points_3d)

    filtered_3d_phi = points_3d[np.asarray(phi_min_id, np.int32)]

    if DEBUG:
        plot_zscan_phi(phi_map=phi_map)

        reproj_l = fringe_process.plot_points_3d_on_image(left_images[:, :, 0], uv_points_L)
        reproj_r = fringe_process.plot_points_3d_on_image(right_images[:, :, 0], uv_points_R)
        fringe_process.show_stereo_image(reproj_l, reproj_r, 'Reprojected points o image')
        cv2.destroyAllWindows()

    if SAVE:
        np.savetxt('./fringe_points.txt', filtered_3d_phi, delimiter='\t', fmt='%.3f')

    print('Total time: {} s'.format(round(time.time() - t0, 2)))
    print('wait')

    fringe_process.plot_3d_points(filtered_3d_phi[:, 0], filtered_3d_phi[:, 1], filtered_3d_phi[:, 2], color=None, title="Point Cloud of min phase diff")

def main():
    yaml_file = '/home/bianca/PycharmProjects/fringe_projection/Params/SM4_20241004_bianca.yaml'

    points_3d = fringe_process.points3d(x_lim=(-250, 500), y_lim=(-100, 400), z_lim=(-200, 200), xy_step=7, z_step=0.1, visualize=False)

    fringe_zscan(points_3d=points_3d, yaml_file=yaml_file, DEBUG=False, SAVE=True)

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml

def points3d(x_lim, y_lim, z_lim, xy_step, z_step, visualize=True):
    x_lin = np.arange(x_lim[0], x_lim[1], xy_step)
    y_lin = np.arange(y_lim[0], y_lim[1], xy_step)
    z_lin = np.arange(z_lim[0], z_lim[1], z_step)

    mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')

    c_points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)

    if visualize:
        plot_3d_points(x=c_points[:, 0], y=c_points[:, 1], z=c_points[:, 2])

    return c_points

def plot_3d_points(x, y, z, color=None, title='Plot 3D of max correlation points'):
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

def plot_points_3d_on_image(image, points, color=(0, 255, 0), radius=5, thickness=2):
    output_image = cv2.cvtColor(np.uint8(image), cv2.COLOR_GRAY2BGR)
    for (u, v, _) in points.T:
        cv2.circle(output_image, (int(u), int(v)), radius, color, thickness)

    return output_image

def show_stereo_image(left, right, name='Rectified image'):
    combined_image = np.concatenate((left, right), axis=1)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, int(combined_image.shape[1] / 4), int(combined_image.shape[0] / 4))
    cv2.imshow(name, combined_image)
    cv2.waitKey(0)

def read_image(path, image):
    image_path = os.path.join(path, image)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    return image

def load_camera_params(yaml_file):
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

    R = np.array(params['R'], dtype=np.float64)
    T = np.array(params['T'], dtype=np.float64)

    return Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr, R, T
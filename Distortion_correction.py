import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat


def points3d(x_lim, y_lim, z_lim, xy_step, z_step, visualize=True):
    x_lin = np.arange(x_lim[0], x_lim[1], xy_step)
    y_lin = np.arange(y_lim[0], y_lim[1], xy_step)
    z_lin = np.arange(z_lim[0], z_lim[1], z_step)

    mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')

    c_points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)

    if visualize:
        plot_3d_points(x=c_points[:, 0], y=c_points[:, 1], z=c_points[:, 2])

    return c_points


def plot_3d_points(x, y, z, color=None):
    if color is None:
        color = z
        cmap = 'viridis'
    else:
        cmap = 'gray'
        # Plot the 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(x, y, z, c=color, cmap=cmap, marker='o')
        # ax.set_zlim(0, np.max(z))
        colorbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        colorbar.set_label('Z Value Gradient')

        # Add labels
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')

        plt.show()


def gcs_f_ccs(xyz_gcs, a_im, k, rot_m, tran):
    xyz_gcs_1 = np.hstack((xyz_gcs, np.ones((xyz_gcs.shape[0], 1))))

    rt_matrix = np.vstack((np.hstack((rot_m, tran)), [0, 0, 0, 1]))

    xyz_ccs = np.dot(rt_matrix, xyz_gcs_1.T)

    epsilon = 1e-10
    xyz_css_norm = np.hstack((xyz_ccs[:2, :].T / np.maximum(xyz_ccs[2, :, np.newaxis], epsilon), np.ones((xyz_ccs.shape[1], 1)))).T

    xyz_ccs_norm_undist = undistorted_points(xyz_css_norm.T, k)

    uv_points = np.dot(a_im, xyz_ccs_norm_undist.T)

    return uv_points


def undistorted_points(norm_points, kc):
    r2 = norm_points[:, 0] ** 2 + norm_points[:, 1] ** 2
    k1, k2, p1, p2, k3 = kc
    factor = (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)
    x_corrected = norm_points[:, 0] * factor + 2 * p1 * norm_points[:, 0] * norm_points[:, 1] + p2 * (
                r2 + 2 * norm_points[:, 0] ** 2)
    y_corrected = norm_points[:, 1] * factor + p1 * (r2 + 2 * norm_points[:, 1] ** 2) + 2 * p2 * norm_points[:,
                                                                                                 0] * norm_points[:, 1]

    return np.hstack((np.stack([x_corrected, y_corrected], axis=-1), np.ones((norm_points.shape[0], 1))))


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


def load_camera_params(path, mat_file):
    full_path = os.path.join(path, mat_file)

    params = loadmat(full_path)

    a_im_l = np.array(params['A_l'], dtype=np.float64)
    k_l = np.array(params['kc_l'], dtype=np.float64)
    rot_m_l = np.array(params['R_1_l'], dtype=np.float64)
    tran_l = np.array(params['Tc_1_l'], dtype=np.float64)

    a_im_r = np.array(params['A_r'], dtype=np.float64)
    k_r = np.array(params['kc_r'], dtype=np.float64)
    rot_m_r = np.array(params['R_1_r'], dtype=np.float64)
    tran_r = np.array(params['Tc_1_r'], dtype=np.float64)

    return a_im_l, k_l, rot_m_l, tran_l, a_im_r, k_r, rot_m_r, tran_r


def main():
    image_path = 'C:/Users/bianca.rosa/Documents/params_for_project_points'
    os.chdir(image_path)

    left_images = read_image(image_path, 'L000.png')
    right_images = read_image(image_path, 'R000.png')
    mat_file = 'Params.mat'

    a_im_l, k_l, rot_m_l, tran_l, a_im_r, k_r, rot_m_r, tran_r = load_camera_params(image_path, mat_file)

    xy_points = points3d(x_lim=(-300, 300), y_lim=(-300, 300), z_lim=(0, 1), xy_step=10, z_step=1, visualize=True)
    uv_points_L = gcs_f_ccs(xy_points, a_im_l, k_l, rot_m_l, tran_l)
    uv_points_R = gcs_f_ccs(xy_points, a_im_r, k_r, rot_m_r, tran_r)
    output_image_L = plot_points_3d_on_image(image=left_images, points=uv_points_L, color=(0, 255, 0),
                                             radius=5, thickness=2)
    output_image_R = plot_points_3d_on_image(image=right_images, points=uv_points_R, color=(0, 255, 0),
                                             radius=5, thickness=2)

    show_stereo_image(output_image_L, output_image_R, "Remaped points")
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()


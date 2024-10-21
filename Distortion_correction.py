import cv2
import numpy as np
import fringe_process

def gcs2ccs(xyz_gcs, k, dist, rot, tran):
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
    xyz_ccs_norm_undist = undistorted_points(xyz_ccs_norm.T, dist)

    # Compute image's point as intrinsic K to XYZ CCS points normalized and undistorted
    uv_points = np.dot(k, xyz_ccs_norm_undist.T)
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

def main():

    yaml_file = 'C:/Users/bianca.rosa/PycharmProjects/fringe_projection/Params/SM4_20241004_bianca.yaml'
    images_path = 'C:/Users/bianca.rosa/PycharmProjects/fringe_projection/Params'

    left_images = fringe_process.read_image(images_path, 'left000.png')
    right_images = fringe_process.read_image(images_path, 'right000.png')

    Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr, R, T = fringe_process.load_camera_params(yaml_file=yaml_file)

    xy_points = fringe_process.points3d(x_lim=(-300, 300), y_lim=(-300, 300), z_lim=(0, 1), xy_step=25, z_step=1, visualize=True)

    uv_points_L = gcs2ccs(xy_points, Kl, Dl, Rl, Tl)
    uv_points_R = gcs2ccs(xy_points, Kr, Dr, Rr, Tr)

    output_image_L = fringe_process.plot_points_3d_on_image(image=left_images, points=uv_points_L, color=(0, 255, 0),
                                             radius=5, thickness=2)
    output_image_R = fringe_process.plot_points_3d_on_image(image=right_images, points=uv_points_R, color=(0, 255, 0),
                                             radius=5, thickness=2)

    fringe_process.show_stereo_image(output_image_L, output_image_R, "Remaped points")
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
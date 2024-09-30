import numpy as np
import os
import matplotlib.pyplot as plt
import Distortion_correction
from scipy.ndimage import map_coordinates

def interpolate_phase_map(images, projected_points):

    projected_points_uv = np.vstack([projected_points[1, :], projected_points[0, :]])

    inter_Igray = map_coordinates(images, projected_points_uv, order=0, mode='constant', cval=10000)

    return inter_Igray

def calculate_phase_difference(intensity_left, intensity_right):
    # Calcula a diferença de fase entre o mapa esquerdo e direito
    DifFase = np.abs(intensity_left - intensity_right)

    return DifFase


def find_valid_points(xy_points, phase_difference, threshold=300):
    z_size = np.unique(xy_points[:, 2]).shape[0]
    # Encontra o Z de menor diferença de fase e filtra os pontos válidos
    # Fmin = np.min(phase_difference, axis=0)
    # Zsmin = np.argmin(phase_difference, axis=0)

    # Rever a validação de pixel (dividir a minha diferença de fase para encontar o zs correspondente de cada espaço no vetor intensidade)
    ho_zstep = []
    ho = phase_difference / z_size
    hmin = np.zeros(int(xy_points.shape[0] / z_size))
    hmax = np.zeros(int(xy_points.shape[0] / z_size))
    Imax = np.zeros(int(xy_points.shape[0] / z_size))
    Imin = np.zeros(int(xy_points.shape[0] / z_size))

    for k in range(xy_points.shape[0] // z_size):
        ho_range = ho[k * z_size:(k + 1) * z_size]
        ho_zstep.append(ho_range)
        # find the maximum correlation for this (X, Y) pair
        hmax[k] = np.nanmax(ho_range)
        hmin[k] = np.nanmin(ho_range)
        # Find the index of the maximum correlation value
        Imax[k] = np.nanmax(ho_range) + k * z_size
        Imin[k] = np.nanargmin(ho_range) + k * z_size

    return ho, hmax, hmin, Imax, Imin, ho_zstep

def plot_point_cloud(COP):
    # Exibir a nuvem de pontos 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(COP[:, 0], COP[:, 1], COP[:, 2], c=COP[:, 2], cmap='viridis', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.colorbar(scatter, label='Profundidade (Z)')
    plt.title('Nuvem de Pontos 3D Reconstruída a partir do Mapa de Fase')
    plt.show()


def main():
    image_path = 'C:/Users/bianca.rosa/Documents/params_for_project_points'
    os.chdir(image_path)

    phasemap_l = Distortion_correction.read_image(image_path, 'abs_phi_image_left.png')
    phasemap_r = Distortion_correction.read_image(image_path, 'abs_phi_image_right.png')
    mat_file = 'Params.mat'

    a_im_l, k_l, rot_m_l, tran_l, a_im_r, k_r, rot_m_r, tran_r = Distortion_correction.load_camera_params(image_path, mat_file)

    xy_points = Distortion_correction.points3d(x_lim=(0, 200), y_lim=(0, 200), z_lim=(-200, 200), xy_step=10, z_step=0.01, visualize=False)
    uv_points_L = Distortion_correction.gcs_f_ccs(xy_points, a_im_l, k_l, rot_m_l, tran_l)
    uv_points_R = Distortion_correction.gcs_f_ccs(xy_points, a_im_r, k_r, rot_m_r, tran_r)

    # Interpola os mapas de fase
    intensity_l = interpolate_phase_map(phasemap_l, uv_points_L)
    intensity_r = interpolate_phase_map(phasemap_r, uv_points_R)

    # Calcula a diferença de fase
    phase_diff = calculate_phase_difference(intensity_l, intensity_r)

    # Encontra os pontos válidos
    ho, hmax, hmin, imax, imin, ho_zstep = find_valid_points(xy_points, phase_diff)

    filtered_3d_ho = xy_points[np.asarray(imax, np.int32)]
    filtered_3d_ho_min = xy_points[np.asarray(imin, np.int32)]

    plot_point_cloud(filtered_3d_ho)
    plot_point_cloud(filtered_3d_ho_min)

    # Grava os pontos válidos em arquivo
    # np.savetxt('COP2.txt', COP, delimiter='\t', fmt='%.3f')

    # Plota a nuvem de pontos
    # plot_point_cloud(COP)


if __name__ == '__main__':
    main()

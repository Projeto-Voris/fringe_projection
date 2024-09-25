import cv2
import numpy as np
import matplotlib.pyplot as plt
import Distortion_correction
import os
from scipy.interpolate import griddata

def interpolate_points(undistorted_points_3d, image_shape):
    # Gerar uma grade com as coordenadas da imagem
    grid_x, grid_y = np.mgrid[0:image_shape[0], 0:image_shape[1]]

    # Extrair as coordenadas originais e os valores correspondentes
    points = undistorted_points_3d[:, :2]  # Pegando as coordenadas 2D (x, y)
    values_x = undistorted_points_3d[:, 0]  # Valores de x
    values_y = undistorted_points_3d[:, 1]  # Valores de y

    # Interpolação para remapear para a resolução da imagem
    m_x = griddata(points, values_x, (grid_x, grid_y), method='linear')
    m_y = griddata(points, values_y, (grid_x, grid_y), method='linear')

    return m_x, m_y
def temp_correlation(points_3d, undistorted_points_3d, image_left, image_right):
    zs = np.linspace(0, 1, points_3d.shape[0])  # Ajustado para gerar uma sequência de valores de z

    COP = []

    mg1, mg2, mg3 = (points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])

    # Interpolar os pontos 3D não distorcidos para corresponder à resolução da imagem
    m_x, m_y = interpolate_points(undistorted_points_3d, image_left.shape[:2])

    # Usando remap para mapear as coordenadas dos pontos não distorcidos
    intel_l = cv2.remap(image_left, m_x.astype(np.float32), m_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    intel_r = cv2.remap(image_right, m_x.astype(np.float32), m_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if len(intel_l.shape) == 2:
        intel_l = np.expand_dims(intel_l, axis=2)
        intel_r = np.expand_dims(intel_r, axis=2)

    Dif_Fase = np.abs(intel_l - intel_r)
    if Dif_Fase.ndim == 3:  # Imagem com 3 dimensões (ex: RGB)
        Fmin = np.min(Dif_Fase, axis=2)
        Zsmin = np.argmin(Dif_Fase, axis=2)
    else:  # Imagem com 2 dimensões (escala de cinza)
        Fmin = np.min(Dif_Fase)
        Zsmin = np.argmin(Dif_Fase)

    # Fmin = np.min(Dif_Fase, axis=2)
    # Zsmin = np.argmin(Dif_Fase, axis=2)

    validIndices = Fmin <= 300
    xValid, yValid = np.where(validIndices)

    for k in range(len(xValid)):
        COP.append([mg1[xValid[k]], mg2[xValid[k]], zs[Zsmin[xValid[k]]]])

    COP = np.array(COP)
    return COP


def show_point_cloud(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Nuvem de Pontos 3D reconstruida a partir do Mapa de Fase')
    plt.axis('equal')
    plt.show()


def main():
    image_path = 'C:/Users/bianca.rosa/Documents/params_for_project_points'
    os.chdir(image_path)

    left_image = Distortion_correction.read_image(image_path, 'abs_phi_image_left.png')  # Carrega uma imagem de cada vez
    right_image = Distortion_correction.read_image(image_path, 'abs_phi_image_right.png')

    mat_file = 'Params.mat'

    # Carregando os parâmetros da câmera a partir do arquivo
    a_im_l, k_l, rot_m_l, tran_l, a_im_r, k_r, rot_m_r, tran_r = Distortion_correction.load_camera_params(image_path,
                                                                                                          mat_file)

    # Geração dos pontos 3D
    xy_points = Distortion_correction.points3d(x_lim=(-300, 300), y_lim=(-300, 300), z_lim=(0, 1), xy_step=10, z_step=1,
                                               visualize=True)

    # Projeção dos pontos no espaço da imagem
    uv_points_L = Distortion_correction.gcs_f_ccs(xy_points, a_im_l, k_l, rot_m_l, tran_l)
    uv_points_R = Distortion_correction.gcs_f_ccs(xy_points, a_im_r, k_r, rot_m_r, tran_r)

    # Buscando os pontos correspondentes
    COP_l = temp_correlation(xy_points, uv_points_L, left_image, right_image)
    COP_r = temp_correlation(xy_points, uv_points_R, right_image, left_image)

    # Exibição da nuvem de pontos
    show_point_cloud(COP_l)
    show_point_cloud(COP_r)

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
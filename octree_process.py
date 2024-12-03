import numpy as np
import open3d as o3d

def filter_points_by_depth(points, depth_threshold=0.05):
    # Converte o numpy array para um objeto PointCloud do Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Cria a octree com base na profundidade, qunado maior mais divisão
    octree = o3d.geometry.Octree(max_depth=2)
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
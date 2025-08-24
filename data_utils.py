import numpy as np
import torch
from skimage import measure


def create_simple_3d_shapes(grid_size=32):
    """创建简单的3D形状数据集"""
    # 创建立方体
    cube = np.zeros((grid_size, grid_size, grid_size))
    margin = grid_size // 4
    cube[margin:-margin, margin:-margin, margin:-margin] = 1

    # 创建球体
    sphere = np.zeros((grid_size, grid_size, grid_size))
    center = grid_size // 2
    radius = grid_size // 4
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                if (i - center) ** 2 + (j - center) ** 2 + (k - center) ** 2 <= radius ** 2:
                    sphere[i, j, k] = 1

    # 创建桌子形状
    table = np.zeros((grid_size, grid_size, grid_size))
    # 桌面
    table[10:22, 10:22, 20:24] = 1
    # 桌腿
    table[12:14, 12:14, 0:20] = 1
    table[12:14, 18:20, 0:20] = 1
    table[18:20, 12:14, 0:20] = 1
    table[18:20, 18:20, 0:20] = 1

    # 创建椅子形状
    chair = np.zeros((grid_size, grid_size, grid_size))
    # 座位
    chair[10:22, 10:22, 16:20] = 1
    # 椅背
    chair[10:14, 10:22, 20:28] = 1
    # 椅腿
    chair[12:14, 12:14, 0:16] = 1
    chair[12:14, 18:20, 0:16] = 1
    chair[18:20, 12:14, 0:16] = 1
    chair[18:20, 18:20, 0:16] = 1

    data = np.stack([cube, sphere, table, chair])
    return torch.FloatTensor(data)


def voxel_to_mesh(voxels, threshold=0.5):
    """将体素转换为网格"""
    voxels = np.squeeze(voxels)

    # 确保网格大小足够
    if voxels.shape[0] < 8:
        padded = np.zeros((32, 32, 32))
        start = (32 - voxels.shape[0]) // 2
        end = start + voxels.shape[0]
        padded[start:end, start:end, start:end] = voxels
        voxels = padded

    try:
        verts, faces, normals, values = measure.marching_cubes(voxels, threshold)

        # 确保面索引是整数类型
        faces = faces.astype(np.int32)

        return verts, faces
    except Exception as e:
        print(f"Marching cubes failed: {e}")
        # 返回一个简单的立方体作为备用
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # 底面
            [4, 5, 6], [4, 6, 7],  # 顶面
            [0, 1, 5], [0, 5, 4],  # 前面
            [2, 3, 7], [2, 7, 6],  # 后面
            [0, 3, 7], [0, 7, 4],  # 左面
            [1, 2, 6], [1, 6, 5]  # 右面
        ])
        return verts, faces
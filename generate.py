import torch
import pyvista as pv
import numpy as np
import os
import sys
import importlib.util

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 动态导入模块
def import_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 导入自定义模块
models = import_module_from_file('models.py', 'models')
data_utils = import_module_from_file('data_utils.py', 'data_utils')

# 检查模型是否存在
model_path = 'models/generator_final.pth'
if not os.path.exists(model_path):
    print("Model not found. Please train the model first by running 'python train.py'")
    exit()

# 初始化生成器
latent_dim = 100
generator = models.Generator(latent_dim)

# 加载模型
generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
generator.eval()

print("Generating 3D shape...")

# 生成样本
z = torch.randn(1, latent_dim)
with torch.no_grad():
    gen_voxels = generator(z).numpy()

# 转换为网格
verts, faces = data_utils.voxel_to_mesh(gen_voxels[0])

# 创建PyVista网格 - 修复面索引格式问题
# PyVista需要面数组以每个面的顶点数开头
faces_pv = np.insert(faces, 0, 3, axis=1)
faces_pv = faces_pv.astype(np.int32).flatten()

# 创建网格
mesh = pv.PolyData(verts, faces_pv)

print("Rendering 3D shape...")

# 可视化
plotter = pv.Plotter()
plotter.add_mesh(mesh, color='lightblue', show_edges=True)
plotter.show()
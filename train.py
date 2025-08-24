import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
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

# 创建模型保存目录
os.makedirs('models', exist_ok=True)

# 超参数
latent_dim = 100
lr = 0.0002
b1 = 0.5
b2 = 0.999
batch_size = 4
n_epochs = 1000
sample_interval = 100

# 初始化模型
generator = models.Generator(latent_dim)
discriminator = models.Discriminator()

# 损失函数
adversarial_loss = nn.BCELoss()

# 优化器
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)

# 创建数据集
data = data_utils.create_simple_3d_shapes()
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练循环
for epoch in range(n_epochs):
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')
    for i, (real_imgs,) in enumerate(progress_bar):
        batch_size = real_imgs.shape[0]

        # 真实和假标签
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # 真实数据
        real_imgs = real_imgs.to(device)

        # ---------------------
        #  训练判别器
        # ---------------------
        optimizer_D.zero_grad()

        # 真实图像的损失
        real_loss = adversarial_loss(discriminator(real_imgs), valid)

        # 生成假图像
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_imgs = generator(z)

        # 假图像的损失
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

        # 总判别器损失
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        #  训练生成器
        # ---------------------
        optimizer_G.zero_grad()

        # 生成器试图欺骗判别器
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        progress_bar.set_postfix(
            D_loss=d_loss.item(),
            G_loss=g_loss.item()
        )

    # 定期保存模型
    if epoch % sample_interval == 0:
        torch.save(generator.state_dict(), f'models/generator_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'models/discriminator_{epoch}.pth')
        print(f"Saved models at epoch {epoch}")

# 保存最终模型
torch.save(generator.state_dict(), 'models/generator_final.pth')
torch.save(discriminator.state_dict(), 'models/discriminator_final.pth')
print("Training completed! Final models saved.")
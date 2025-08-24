import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, output_shape=(32, 32, 32)):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        self.init_size = output_shape[0] // 4
        self.l1 = nn.Linear(latent_dim, 128 * self.init_size ** 3)

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(32, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img.view(img.shape[0], *self.output_shape)


class Discriminator(nn.Module):
    def __init__(self, input_shape=(32, 32, 32)):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv3d(in_filters, out_filters, 4, 2, 1)]
            if bn:
                block.append(nn.BatchNorm3d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128)
        )

        # 计算全连接层输入尺寸
        ds_size = input_shape[0] // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 3, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = img.unsqueeze(1)  # 添加通道维度
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
import torch.nn as nn
import torch
import torch.nn.functional as F

ngf = 32

class Dis_z(nn.Module):
    def __init__(self, latent_dim):
        super(Dis_z, self).__init__()
        # self.ngpu = ngpu
        self.z_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(self.z_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        output = self.net(z)
        return output

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        # self.ngpu = ngpu
        self.z_dim = latent_dim
        self.net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 1 x 32 x 32
        )

    def forward(self, z):
        z_ = z.view(z.shape[0], z.shape[1], 1, 1)
        output = self.net(z_)
        return output

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(1, ngf, padding = 1, kernel_size = 4, stride=2),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # ngf x 16 x 16

            nn.Conv2d(ngf, 2 * ngf, padding = 1, kernel_size = 4, stride=2),
            nn.BatchNorm2d(2 * ngf),
            nn.ReLU(True),
            # 2*ngf x 8 x 8

            nn.Conv2d(2 * ngf, 4 * ngf, padding = 1, kernel_size = 4, stride=2),
            nn.BatchNorm2d(4 * ngf),
            nn.ReLU(True),
            # 4*ngf x 4 x 4
            nn.Conv2d(4 * ngf, 8 * ngf, padding = 1, kernel_size = 4, stride=2),
            nn.BatchNorm2d(8 * ngf),
            nn.ReLU(True),
            # 8*ngf x 2 x 2

            nn.Conv2d(8 * ngf, self.latent_dim, padding = 1, kernel_size = 4, stride=2),
            # latent_dim x 1 x 1
        )
    def forward(self, x):
        return torch.squeeze(self.net(x))

class Discriminator(nn.Module):
    def __init__(self, color_channel = 1, pretrained = False):
        super(Discriminator, self).__init__()
        self.x_net = nn.Sequential(
            nn.Conv2d(color_channel, ngf, padding = 1, kernel_size = 4, stride=2),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # ngf * 16 * 16

            nn.Conv2d(ngf, ngf, padding = 1, kernel_size = 4, stride=2),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # ngf * 8 * 8
        )

        self.net = nn.Sequential(
            nn.Linear(ngf * 8 * 8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print ("z dim:", z.shape)
        x_ = torch.flatten(self.x_net(x), start_dim = 1)
        logits = self.net(x_)
        return logits

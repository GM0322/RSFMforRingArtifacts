import math
import torch
import numpy as np

class conv3(torch.nn.Module):
    def __init__(self,in_feature=32,base_feature=32):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_feature,base_feature,kernel_size=3,padding=1),
            # torch.nn.BatchNorm2d(base_feature),
            torch.nn.ReLU(),
            torch.nn.Conv2d(base_feature, base_feature, kernel_size=3, padding=1),
            # torch.nn.BatchNorm2d(base_feature),
            torch.nn.ReLU(),
            torch.nn.Conv2d(base_feature, base_feature, kernel_size=3, padding=1),
            # torch.nn.BatchNorm2d(base_feature),
            torch.nn.ReLU())

    def forward(self,x):
        return self.conv(x)

class convt3(torch.nn.Module):
    def __init__(self,in_feature=32,base_feature=32):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_feature, base_feature, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(base_feature, base_feature, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(base_feature, base_feature, kernel_size=3, padding=1),
            torch.nn.ReLU())

    def forward(self,x):
        return self.conv(x)


class SinusoidalTimeEmbedding(torch.nn.Module):
    """
    正弦余弦时间嵌入模块

    参数:
        embedding_dim: 嵌入维度 (d_model)
        base: 频率计算的基数 (默认为10000.0)
        scaled: 是否对输出进行归一化 (默认为True)

    输入:
        positions: 位置索引张量, 形状任意 (将展平处理)

    输出:
        嵌入向量, 形状为 (input_shape, embedding_dim)
    """

    def __init__(self, embedding_dim, base=10000.0, scaled=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.base = base
        self.scaled = scaled

        # 计算频率常数
        exponents = 2 * torch.arange(0, embedding_dim // 2) / embedding_dim
        freqs = 1.0 / (base ** exponents)
        self.register_buffer('freqs', freqs)  # 注册为缓冲区

        # 初始化缩放因子
        self.scale = torch.nn.Parameter(torch.ones(1)) if scaled else 1.0

    def forward(self, positions):
        orig_shape = positions.shape
        positions = positions.flatten().float()
        angles = positions.unsqueeze(1) * self.freqs.unsqueeze(0)
        angles = angles * (2 * torch.pi)  # 转换为完整周期

        # 创建正弦余弦嵌入
        embeddings = torch.zeros(positions.shape[0], self.embedding_dim, device=positions.device)
        embeddings[:, 0::2] = torch.sin(angles)  # 偶数列
        embeddings[:, 1::2] = torch.cos(angles)  # 奇数列
        if self.scaled:
            embeddings = self.scale * embeddings
        return embeddings#.unflatten(0, orig_shape)


class CNN(torch.nn.Module):
    def __init__(self,in_feature=7, base_feature=32):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(base_feature)
        self.conv1 = conv3(in_feature, base_feature)
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(base_feature,base_feature),
            torch.nn.Tanh()
        )
        self.conv2 = conv3(base_feature, base_feature)
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(base_feature,base_feature),
            torch.nn.Tanh()
        )
        self.conv3 = conv3(base_feature, base_feature)
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(base_feature,base_feature),
            torch.nn.Tanh()
        )
        self.conv4 = conv3(base_feature, base_feature)
        self.fc4 = torch.nn.Sequential(
            torch.nn.Linear(base_feature,base_feature),
            torch.nn.Tanh()
        )
        self.conv5 = conv3(base_feature, base_feature)

        self.tconv1 = convt3(base_feature, base_feature)
        self.tconv2 = convt3(base_feature, base_feature)
        self.tconv3 = convt3(base_feature, base_feature)
        self.tconv4 = convt3(base_feature, base_feature)
        self.tconv5 = convt3(base_feature, base_feature)
        self.out = torch.nn.Conv2d(base_feature, 1, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x, t):
        time = self.time_embedding(t)
        out = self.conv1(x)
        residual_1 = out
        out = self.conv2(out*(self.fc1(time)[..., None, None]))
        residual_2 = out
        out = self.conv3(out*(self.fc2(time)[..., None, None]))
        residual_3 = out
        out = self.conv4(out*(self.fc3(time)[..., None, None]))
        residual_4 = out
        out = self.conv5(out*(self.fc4(time)[..., None, None]))
        # decoder
        out = self.tconv5(out)
        out = out + residual_4
        out = self.tconv4(out)
        out = out + residual_3
        out = self.tconv3(out)
        out = out + residual_2
        out = self.tconv2(out)
        out = out + residual_1
        out = self.tconv1(out)
        out = self.relu(self.out(out))
        return out


class GatingSignal(torch.nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super(GatingSignal, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        return self.activation(x)

class Attention_Gate(torch.nn.Module):
    def __init__(self, in_channels):
        super(Attention_Gate, self).__init__()
        self.conv_theta_x = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 1), stride=(2, 2)
        )
        self.conv_phi_g = torch.nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.att = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, 1, kernel_size=(1, 1)),
            torch.nn.Sigmoid(),
            torch.nn.Upsample(scale_factor=2),
        )

    def forward(self, x, gat):
        theta_x = self.conv_theta_x(x)
        phi_g = self.conv_phi_g(gat)
        res = torch.add(phi_g, theta_x)
        res = self.att(res)
        # print(res.size(), x.size())
        return res, torch.mul(res, x)

class DoubleConv(torch.nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels,isBn=True):
        super().__init__()
        self.isBn = isBn
        if isBn:
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn1 = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.ReLU(inplace=True)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.relu = torch.nn.ReLU(inplace=True)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        if self.isBn:
            out = self.relu(self.bn1(self.conv1(x)))
            weight = torch.exp(torch.linspace(0,1,out.shape[1],requires_grad=False).to(x.device))[None, :, None, None]
            out = out * weight
            return self.relu(self.bn2(self.conv2(out)))
        else:
            out = self.relu(self.conv1(x))
            weight = torch.exp(torch.linspace(0,1,out.shape[1],requires_grad=False).to(x.device))[None, :, None, None]
            out = out * weight
            return self.relu(self.conv2(out))

class Down_Block(torch.nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, drop=0.5):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        # self.down = nn.Sequential(nn.MaxPool2d(2), nn.Dropout(drop))

    def forward(self, x):
        c = self.conv(x)
        return c#, self.down(c)

class Bridge(torch.nn.Module):
    def __init__(self, in_channels, out_channels, drop):
        super().__init__()
        self.conv = torch.nn.Sequential(
            DoubleConv(in_channels, out_channels), torch.nn.Dropout(drop)
        )

    def forward(self, x):
        return self.conv(x)

class Up_Block(torch.nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, drop=0.5, attention=False,isBn=True):
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2)
        )
        if isBn:
            self.conv = torch.nn.Sequential(
                DoubleConv(in_channels, out_channels), torch.nn.Dropout(p=drop)
            )
        else:
            self.conv = torch.nn.Sequential(
                DoubleConv(in_channels, out_channels,isBn=isBn), torch.nn.Dropout(p=drop)
            )
        self.attention = attention
        if attention:
            self.gating = GatingSignal(in_channels, out_channels)
            self.att_gat = Attention_Gate(out_channels)

    def forward(self, x, conc):
        x1 = self.up(x)
        if self.attention:
            gat = self.gating(x)
            map, att = self.att_gat(conc, gat)
            x = torch.cat([x1, att], dim=1)
            return map, self.conv(x)
        else:
            x = torch.cat([conc, x1], dim=1)
            return None, self.conv(x)

class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)

class UnetAttnCondition(torch.nn.Module):
    def __init__(self,
                 in_channels=7,
                 out_channels=1,
                 base_features=16,
                 drop_r=0.0,
                 attention=True,
                 noise_dim=1
                 ):
        super().__init__()

        self.down1 = Down_Block(in_channels, base_features)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.noise_encoder1 = torch.nn.Sequential(
            torch.nn.Linear(noise_dim,base_features),
            torch.nn.Sigmoid()
        )
        self.down2 = Down_Block(base_features, base_features * 2, drop_r)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.noise_encoder2 = torch.nn.Sequential(
            torch.nn.Linear(noise_dim,base_features*2),
            torch.nn.Sigmoid()
        )
        self.down3 = Down_Block(base_features * 2, base_features * 4, drop_r)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.noise_encoder3 = torch.nn.Sequential(
            torch.nn.Linear(noise_dim,base_features*4),
            torch.nn.Sigmoid()
        )
        self.down4 = Down_Block(base_features * 4, base_features * 8, drop_r)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.noise_encoder4 = torch.nn.Sequential(
            torch.nn.Linear(noise_dim,base_features*8),
            torch.nn.Sigmoid()
        )
        self.bridge = Bridge(base_features * 8, base_features * 16, drop_r)

        self.up1 = Up_Block(base_features * 16, base_features * 8, drop_r, attention)
        self.up2 = Up_Block(base_features * 8, base_features * 4, drop_r, attention)
        self.up3 = Up_Block(base_features * 4, base_features * 2, drop_r, attention)
        self.up4 = Up_Block(base_features * 2, base_features, drop_r, attention,isBn=False)

        self.outc = OutConv(base_features, out_channels)

    # @profile
    def forward(self, x,t):
        x1 = self.down1(x)
        weight = self.noise_encoder1(t)[:,:,None,None]
        # x1_ = torch.chunk(x1,chunks=2,dim=1)
        # x1 = torch.cat([weight*x1_[0],x1_[1]],dim=1)
        x1 = weight*x1
        x2 = self.down2(self.pool1(x1))

        weight = self.noise_encoder2(t)[:,:,None,None]
        # x2_ = torch.chunk(x2, chunks=2, dim=1)
        # x2 = torch.cat([weight * x2_[0], x2_[1]], dim=1)
        x2 = weight*x2
        x3 = self.down3(self.pool2(x2))

        weight = self.noise_encoder3(t)[:,:,None,None]
        # x3_ = torch.chunk(x3, chunks=2, dim=1)
        # x3 = torch.cat([weight * x3_[0], x3_[1]], dim=1)
        x3 = weight*x3
        x4 = self.down4(self.pool3(x3))

        weight = self.noise_encoder4(t)[:,:,None,None]
        # x4_ = torch.chunk(x4, chunks=2, dim=1)
        # x4 = torch.cat([weight * x4_[0], x4_[1]], dim=1)
        x4 = weight*x4
        bridge = self.bridge(self.pool4(x4))
        _, x = self.up1(bridge, x4)
        _, x = self.up2(x, x3)
        att, x = self.up3(x, x2)
        _, x = self.up4(x, x1)
        out = self.outc(x)
        return out



class RectifiedFlow():
    def __init__(self,model, num_steps=100, dim=5):
        self.model = model
        self.num_steps = num_steps
        self.dim=dim

    def get_train_tuple(self,z0=None,z1=None):
        t = torch.rand((z1.shape[0]))[:,None].to(z0.device)
        t_ = t[:,:,None,None]
        z_t = z0.clone()
        z_t[:,self.dim:self.dim+1,...] = t_ * z1 + (1. - t_) * z0[:,self.dim:self.dim+1,...]
        target = z1 - z0[:,self.dim:self.dim+1,...]
        return z_t, t, target

    @torch.no_grad()
    def sample_ode(self, z0=None, N=None):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.num_steps
        dt = 1. / N
        traj = []  # to store the trajectory
        z = z0.detach().clone()
        batchsize = z.shape[0]

        traj.append(z[:,self.dim:self.dim+1,...].detach().clone())
        for i in range(N):
            t = torch.ones((batchsize, 1)).to(z0.device) * i / N
            pred = self.model(z, t)
            z[:,self.dim:self.dim+1,...] = z[:,self.dim:self.dim+1,...] + pred * dt
            traj.append(z[:,self.dim:self.dim+1,...].detach().clone())
        return traj,z
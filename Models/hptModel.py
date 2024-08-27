import torch.nn as nn
import torch
import torch.nn.functional as F

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        # 深度卷积部分
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, groups=in_channels)
        # 逐点卷积部分
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 深度卷积
        x = self.depthwise_conv(x)
        # 逐点卷积
        x = self.pointwise_conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, ni, nf, SEreduction=16, patch_size=300):
        super(ResBlock, self).__init__()
        self.patch_size = patch_size

        self.convblock1 = nn.Sequential(
            nn.Conv1d(ni, nf, 5, padding='same', padding_mode='reflect', dilation=3),
            nn.BatchNorm1d(nf),
            nn.ReLU()
        )
        self.convblock2 = nn.Sequential(
            nn.Conv1d(nf, nf, 5, padding='same', padding_mode='reflect', dilation=3),
            nn.BatchNorm1d(nf),
            nn.ReLU()
        )
        self.convblock3 = nn.Sequential(
            nn.Conv1d(nf, nf, 5, padding='same', padding_mode='reflect', dilation=3),
            nn.BatchNorm1d(nf),
            nn.ReLU()
        )

        self.convblock_sec1 = nn.Sequential(
            nn.Conv1d(ni, nf, 3, padding='same', padding_mode='reflect'),
            nn.BatchNorm1d(nf),
            nn.ReLU()
        )
        self.convblock_sec2 = nn.Sequential(
            nn.Conv1d(nf, nf, 3, padding='same', padding_mode='reflect'),
            nn.BatchNorm1d(nf),
            nn.ReLU()
        )
        self.convblock_sec3 = nn.Sequential(
            nn.Conv1d(nf, nf, 3, padding='same', padding_mode='reflect'),
            nn.BatchNorm1d(nf),
            nn.ReLU()
        )

        self.to_onechannel1 = nn.Conv1d(nf, 1, 1)
        self.to_onechannel2 = nn.Conv1d(nf, 1, 1)
        # 交叉注意力
        self.to_fir_q_depthwise_separable_conv = DepthwiseSeparableConv1d(in_channels=nf, out_channels=1, kernel_size=3,
                                                                          padding=1)
        self.to_fir_k_depthwise_separable_conv = DepthwiseSeparableConv1d(in_channels=nf, out_channels=1, kernel_size=3,
                                                                          padding=1)
        self.to_sec_q_depthwise_separable_conv = DepthwiseSeparableConv1d(in_channels=nf, out_channels=1, kernel_size=3,
                                                                          padding=1)
        self.to_sec_k_depthwise_separable_conv = DepthwiseSeparableConv1d(in_channels=nf, out_channels=1, kernel_size=3,
                                                                          padding=1)

        self.bn = nn.BatchNorm1d(nf)
        self.fc1 = nn.Linear(nf, nf // SEreduction)
        self.fc2 = nn.Linear(nf // SEreduction, nf * 2)
        self.softmax = nn.Softmax(dim=1)

        self.shortcut = nn.BatchNorm1d(ni) if ni == nf else nn.Conv1d(ni, nf, 1)
        self.act = nn.ReLU()

        self.SEfc1 = nn.Linear(in_features=nf * 2, out_features=nf * 2 // SEreduction)
        self.SEfc2 = nn.Linear(in_features=nf * 2 // SEreduction, out_features=nf * 2)
        self.SErelu = nn.ReLU()
        self.SEsigmoid = nn.Sigmoid()
        self.SEavgpool = nn.AdaptiveAvgPool1d(1)

        self.maxpooling = nn.MaxPool1d(2)

    def forward(self, x):
        res = x
        x_fir = x
        x_sec = x

        x_fir = self.convblock1(x_fir)
        x_fir = self.convblock2(x_fir)

        x_sec = self.convblock_sec1(x_sec)
        x_sec = self.convblock_sec2(x_sec)

        x_fir_ori = x_fir
        x_sec_ori = x_sec
        x_fir_ori = x_fir_ori.view(x_fir_ori.size(0), x_fir_ori.size(1), -1, self.patch_size)
        x_sec_ori = x_sec_ori.view(x_sec_ori.size(0), x_sec_ori.size(1), -1, self.patch_size)

        # 交叉注意力
        x_fir_q = self.to_fir_q_depthwise_separable_conv(x_fir)
        x_fir_q = x_fir_q.squeeze(1).view(x_fir_q.size(0), -1, self.patch_size)
        x_fir_k = self.to_fir_k_depthwise_separable_conv(x_fir)
        x_fir_k = x_fir_k.squeeze(1).view(x_fir_k.size(0), -1, self.patch_size)
        x_sec_q = self.to_sec_q_depthwise_separable_conv(x_sec)
        x_sec_q = x_sec_q.squeeze(1).view(x_sec_q.size(0), -1, self.patch_size)
        x_sec_k = self.to_sec_k_depthwise_separable_conv(x_sec)
        x_sec_k = x_sec_k.squeeze(1).view(x_sec_k.size(0), -1, self.patch_size)

        x_fir_qdotkt = torch.softmax(torch.matmul(x_fir_q, x_fir_k.permute(0, 2, 1)) / (x_fir_q.shape[1] ** 0.5),
                                     dim=-1).unsqueeze(1)
        x_sec_qdotkt = torch.softmax(torch.matmul(x_sec_q, x_sec_k.permute(0, 2, 1)) / (x_fir_q.shape[1] ** 0.5),
                                     dim=-1).unsqueeze(1)
        x_fir = (torch.matmul(x_sec_qdotkt, x_fir_ori) + x_fir_ori).view(x_fir_ori.size(0), x_fir_ori.size(1),
                                                                         -1)
        x_sec = (torch.matmul(x_fir_qdotkt, x_sec_ori) + x_sec_ori).view(x_sec_ori.size(0), x_sec_ori.size(1),
                                                                         -1)
        x_ca = x_fir + x_sec
        x_ca = F.relu(self.bn(x_ca))

        s = F.adaptive_avg_pool1d(x_ca, 1).squeeze(-1)
        z = self.fc1(s)
        z = F.relu(z)
        z = self.fc2(z)
        z = z.view(-1, 2, x_ca.size(1))

        a_b = self.softmax(z)
        a_b = torch.split(a_b, 1, dim=1)
        a_b = [torch.squeeze(a, dim=1) for a in a_b]

        # Select
        x = a_b[0].unsqueeze(2) * x_fir + a_b[1].unsqueeze(2) * x_sec

        x = torch.add(x, self.shortcut(res))

        x = self.maxpooling(x)
        x = self.act(x)
        return x


class FrequencySeparationNet(nn.Module):
    def __init__(self):
        super(FrequencySeparationNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, 201, padding="same")
        self.conv2 = nn.Conv1d(1, 1, 51, padding="same")
        self.conv3 = nn.Conv1d(1, 1, 11, padding="same")
        self.conv11 = nn.Conv1d(1, 1, 201, padding="same")
        self.conv22 = nn.Conv1d(1, 1, 51, padding="same")
        self.conv33 = nn.Conv1d(1, 1, 11, padding="same")

    def forward(self, x):
        x_re1 = self.conv1(x) + self.conv2(x) + self.conv3(x)
        x_low = self.conv11(x_re1)
        x_mid = self.conv22(x_re1)
        x_high = self.conv33(x_re1)
        x_re2 = x_low + x_mid + x_high

        return x_re2, x_low, x_mid, x_high

class ResNet(nn.Module):
    def __init__(self, c_in, n_classes, patch_sizes=[300, 150, 75], FSnet=None, length=3000):
        super(ResNet, self).__init__()

        self.FSnet = FSnet

        self.linear1 = nn.Sequential(nn.Linear(length // 8, 2),
                                     nn.ReLU(),
                                     nn.Linear(2, length // 8))

        self.linear2 = nn.Sequential(nn.Linear(length // 8, 2),
                                     nn.ReLU(),
                                     nn.Linear(2, length // 8))

        self.linear3 = nn.Sequential(nn.Linear(length // 8, 2),
                                        nn.ReLU(),
                                        nn.Linear(2, length // 8))

        nf = 64
        self.res_block_low1 = ResBlock(c_in, nf, patch_size=patch_sizes[0])
        self.res_block_low2 = ResBlock(nf, nf * 2, patch_size=patch_sizes[1])
        self.res_block_low3 = ResBlock(nf * 2, nf * 2, patch_size=patch_sizes[2])
        self.res_block_mid1 = ResBlock(c_in, nf, patch_size=patch_sizes[0])
        self.res_block_mid2 = ResBlock(nf, nf * 2, patch_size=patch_sizes[1])
        self.res_block_mid3 = ResBlock(nf * 2, nf * 2, patch_size=patch_sizes[2])
        self.res_block_high1 = ResBlock(c_in, nf, patch_size=patch_sizes[0])
        self.res_block_high2 = ResBlock(nf, nf * 2, patch_size=patch_sizes[1])
        self.res_block_high3 = ResBlock(nf * 2, nf * 2, patch_size=patch_sizes[2])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear((nf * 2) * 3, nf * 2)
        self.fc2 = nn.Linear(nf * 2, n_classes)

    def cal_energy(self, data, pool_size):
        squared_data = data ** 2
        pooled_data = F.avg_pool1d(squared_data, kernel_size=pool_size, stride=pool_size, padding=0)
        return pooled_data

    def forward(self, x):
        x_re, x_low, x_mid, x_high = self.FSnet(x)

        low_energy = self.cal_energy(x_low, 8)
        mid_energy = self.cal_energy(x_mid, 8)
        high_energy = self.cal_energy(x_high, 8)
        low_energy = self.linear1(low_energy)
        mid_energy = self.linear2(mid_energy)
        high_energy = self.linear3(high_energy)

        concat_tensor = torch.cat((low_energy, mid_energy, high_energy), dim=1)

        softmax_tensor = F.softmax(concat_tensor, dim=1)

        low_energy, mid_energy, high_energy = torch.split(softmax_tensor, 1, dim=1)

        # 再时域
        x_low = self.res_block_low1(x_low)
        x_low = self.res_block_low2(x_low)
        x_low = self.res_block_low3(x_low)

        x_mid = self.res_block_mid1(x_mid)
        x_mid = self.res_block_mid2(x_mid)
        x_mid = self.res_block_mid3(x_mid)

        x_high = self.res_block_high1(x_high)
        x_high = self.res_block_high2(x_high)
        x_high = self.res_block_high3(x_high)

        x_mid_layer = torch.cat([x_low * low_energy, x_mid * mid_energy, x_high * high_energy], dim=1)
        x_mid_layer = self.gap(x_mid_layer).squeeze()
        x_mid_layer = self.fc1(x_mid_layer)
        x = self.fc2(x_mid_layer)
        return x, x_mid_layer


if __name__ == '__main__':
    x = torch.randn(16, 1, 500)
    FSnet = FrequencySeparationNet()
    model = ResNet(1, 2, [100, 50, 25], FSnet, 500)
    x_out, x_mid = model(x)
    print(x_out.shape)
    print(x_mid.shape)

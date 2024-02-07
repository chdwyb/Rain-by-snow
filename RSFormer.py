import torch
import torch.nn as nn

一个 LayerNorm 层用于输入张量 x 的归一化。
一个 1x1 卷积层 (self.fc1)，将输入的特征维度降低到一个较小的维度（即 hidden_features）。
一个 3x3 深度可分离卷积层 (self.dwconv)，对特征进行深度方向的卷积操作。
另一个 1x1 卷积层 (self.fc2)，将特征维度恢复到原始维度。
一个 GELU 激活函数用于激活每个卷积层的输出。
class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden_features = int(dim * mlp_ratio)
        self.norm = LayerNorm(dim)
        self.fc1 = nn.Conv2d(dim, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, dim, 1)
        self.act = nn.GELU()
在 forward 方法中，输入先经过归一化处理，然后通过第一个卷积层和 GELU 激活函数。
接着将输出保存到 res 中以备后用，然后经过深度可分离卷积层，再次经过激活函数。
最后将之前保存的 res 与当前的输出相加，得到最终的输出，再通过第二个卷积层得到最终的特征表示。
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        res = x
        x = self.dwconv(x)
        x = self.act(x) + res
        x = self.fc2(x)
        return x
这个结构可以有效地捕捉输入特征之间的关系，并且在一些深度学习模型中被广泛使用。


一个 LayerNorm 层用于输入张量 x 的归一化。
一个 1x1 卷积层 (self.qk) 用于计算 Query 和 Key。
一个 GELU 激活函数用于激活 Query 和 Key。
一个 11x11 的深度可分离卷积层 (self.dwconv) 用于计算注意力权重。
一个 1x1 卷积层 (self.v) 用于计算 Value。
一个 1x1 卷积层 (self.proj) 用于将注意力权重应用到 Value 上。
class Attention(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.qk = nn.Conv2d(dim, dim, 1, bias=bias)
        self.act = nn.GELU()
        self.dwconv = nn.Conv2d(dim, dim, 11, padding=5, groups=dim, bias=bias)

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
在 forward 方法中，输入先经过归一化处理，然后通过 Query 和 Key 的卷积层，再经过 GELU 激活函数。
接着通过深度可分离卷积层计算注意力权重，并再次经过激活函数。然后通过 Value 的卷积层计算 Value。
最后将注意力权重应用到 Value 上，并通过 self.proj 卷积层得到最终的输出。
    def forward(self, x):
        x = self.norm(x)
        qk = self.qk(x)
        attn = self.act(qk)
        attn = self.dwconv(attn)
        attn = self.act(attn)
        v = self.v(x)
        x = attn * v
        x = self.proj(x)
        return x
自注意力机制允许模型在处理序列数据时动态地分配不同位置的注意力权重，是 Transformer 模型的核心组件之一。


这段代码定义了一个卷积块（ConvolutionBlock），它由注意力模块（Attention）和前馈神经网络模块（FeedForward）组成。
一个注意力模块 self.attn，用于处理输入张量 x。
一个前馈神经网络模块 self.ffn，也用于处理输入张量 x。
两个参数 self.layer_scale_1 和 self.layer_scale_2，用于对两个模块的输出进行调节。
class ConvolutionBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.attn = Attention(dim)
        self.ffn = FeedForward(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
在 forward 方法中，输入先经过注意力模块，然后通过参数 self.layer_scale_1 进行缩放，并与输入相加。
接着，将输出再次经过前馈神经网络模块，然后通过参数 self.layer_scale_2 进行缩放，并再次与之前的输出相加。最终的输出就是卷积块的输出。
    def forward(self, x):
        x = x + self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x)
        x = x + self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.ffn(x)
        return x
这种结构允许模型在处理序列数据时通过注意力机制捕捉输入之间的关系，并通过前馈神经网络进行非线性变换和特征提取。

self.weight：用于缩放输入的可学习权重参数。
self.bias：用于偏移输入的可学习偏置参数。
self.eps：用于稳定计算的小值。
层归一化！
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = 1e-6

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

一个 2D 卷积层 self.proj，其作用是将输入图像分割成小的图像块，并将每个图像块映射到一个低维的向量空间中。这个卷积层的输出维度为 embed_dim，即每个图像块被映射到的向量的维度。
class PatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
PatchEmbed 模块通常用于 Vision Transformer 等模型中，将输入图像转换为序列数据，以便于后续的自注意力操作。






class Downsample(nn.Module):
    def __init__(self, dim, num_head=8, bias=False):
        super().__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1))

        self.v = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=bias),
            LayerNorm(dim),
            nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        )
        self.v_hp = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qk = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=bias)
        self.proj = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        out_shape = B, C * 2, H // 2, W // 2

        qk = self.qk(x).reshape(B, 2, self.num_head, (C * 2) // self.num_head, -1).transpose(0, 1)
        q, k = qk[0], qk[1]

        v = self.v(x)
        v_hp = self.v_hp(v)
        v = v.reshape(B, self.num_head, (C * 2) // self.num_head, -1)

        attn = (q @ k.transpose(-1, -2)) * self.temperature
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(out_shape) + v_hp
        x = self.proj(x)
        return x
该 Downsample 模块通常用于自注意力模型（如 Transformer）的下采样操作，用于减小特征图的尺寸并增加通道数。

class Upsample(nn.Module):
    def __init__(self, dim, num_head=8, bias=False):
        super().__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1))

        self.v = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1, bias=bias),
            LayerNorm(dim),
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False)
        )
        self.v_hp = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, stride=1, padding=1, groups=dim // 2, bias=False)
        self.qk = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        out_shape = B, C // 2, H * 2, W * 2

        qk = self.qk(x).reshape(B, 2, self.num_head, (C // 2) // self.num_head, -1).transpose(0, 1)
        q, k = qk[0], qk[1]

        v = self.v(x)
        v_hp = self.v_hp(v)
        v = v.reshape(B, self.num_head, (C // 2) // self.num_head, -1)

        attn = (q @ k.transpose(-1, -2)) * self.temperature
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(out_shape) + v_hp
        x = self.proj(x)
        return x


class RSFormer(nn.Module):
    def __init__(self,
                 in_channels=3,
                 dim=48,
                 num_blocks=(4, 6, 6, 8),
                 num_heads=(2, 4, 8),  # sampling head
                 num_refinement_blocks=4,
                 mlp_ratios=(4, 4, 4, 4),
                 bias=False,
                 ):

        super(RSFormer, self).__init__()
        PatchEmbed 模块： 将输入图像分块并映射到低维向量空间中。
        编码器（Encoder）： 由多个卷积块组成的序列，用于逐步提取图像特征。每个阶段都包括一个下采样模块和多个卷积块。
        解码器（Decoder）： 由多个卷积块组成的序列，用于逐步将特征向量还原为图像。每个阶段都包括一个上采样模块和多个卷积块。
        精化层（Refinement）： 由多个卷积块组成的序列，用于进一步改进重建图像的质量。
        输出层（Output）： 将最终的特征图映射回原始图像的通道数。
        self.patch_embed = PatchEmbed(in_channels, dim)
        self.encoder1 = nn.Sequential(*[
            ConvolutionBlock(dim=dim, mlp_ratio=mlp_ratios[0]) for i in range(num_blocks[0])])

        self.down1 = Downsample(dim, num_head=num_heads[0])
        self.encoder2 = nn.Sequential(*[
            ConvolutionBlock(dim=int(dim * 2 ** 1), mlp_ratio=mlp_ratios[1]) for i in range(num_blocks[1])])

        self.down2 = Downsample(int(dim * 2 ** 1), num_head=num_heads[1])
        self.encoder3 = nn.Sequential(*[
            ConvolutionBlock(dim=int(dim * 2 ** 2), mlp_ratio=mlp_ratios[2]) for i in range(num_blocks[2])])

        self.down3 = Downsample(int(dim * 2 ** 2), num_head=num_heads[2])
        self.latent = nn.Sequential(*[
            ConvolutionBlock(dim=int(dim * 2 ** 3), mlp_ratio=mlp_ratios[3]) for i in range(num_blocks[3])])

        self.up3 = Upsample(int(dim * 2 ** 3), num_head=num_heads[2])
        self.reduce3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder3 = nn.Sequential(*[
            ConvolutionBlock(dim=int(dim * 2 ** 2), mlp_ratio=mlp_ratios[2]) for i in range(num_blocks[2])])

        self.up2 = Upsample(int(dim * 2 ** 2), num_head=num_heads[1])
        self.reduce2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder2 = nn.Sequential(*[
            ConvolutionBlock(dim=int(dim * 2 ** 1), mlp_ratio=mlp_ratios[1]) for i in range(num_blocks[1])])

        self.up1 = Upsample(int(dim * 2 ** 1), num_head=num_heads[0])
        self.decoder1 = nn.Sequential(*[
            ConvolutionBlock(dim=int(dim * 2 ** 1), mlp_ratio=mlp_ratios[0]) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[ConvolutionBlock(dim=int(dim * 2 ** 1), mlp_ratio=mlp_ratios[0]) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim * 2 ** 1), in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
在 forward 方法中，模型首先通过 PatchEmbed 模块将输入图像分块并映射到低维向量空间中。然后通过编码器逐步提取图像特征，再通过解码器逐步将特征向量还原为图像。
最后，通过精化层进一步改进重建图像的质量，并通过输出层映射回原始图像的通道数。整个过程中，输入图像的信息被逐步压缩和解压缩，从而实现图像的重建。
    def forward(self, x):
        input_ = x
        x = self.patch_embed(x)  # stage 1
        x0 = self.encoder1(x)

        x = self.down1(x0)  # stage 2
        x1 = self.encoder2(x)

        x = self.down2(x1)  # stage 3
        x2 = self.encoder3(x)

        x = self.down3(x2)  # stage 4
        x = self.latent(x)

        x = self.up3(x)
        x2 = torch.cat([x, x2], 1)
        x2 = self.reduce3(x2)
        x2 = self.decoder3(x2)

        x = self.up2(x2)
        x1 = torch.cat([x, x1], 1)
        x1 = self.reduce2(x1)
        x1 = self.decoder2(x1)

        x = self.up1(x1)
        x0 = torch.cat([x, x0], 1)
        x0 = self.decoder1(x0)

        x = self.refinement(x0)
        x = self.output(x) + input_
        return x


if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256)).cuda()
    net = RSFormer().cuda()

    from thop import profile, clever_format
    flops, params = profile(net, (x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden_features = int(dim * mlp_ratio)
        self.norm = LayerNorm(dim)
        self.fc1 = nn.Conv2d(dim, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        res = x
        x = self.dwconv(x)
        x = self.act(x) + res
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.qk = nn.Conv2d(dim, dim, 1, bias=bias)
        self.act = nn.GELU()
        self.dwconv = nn.Conv2d(dim, dim, 11, padding=5, groups=dim, bias=bias)

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

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

    def forward(self, x):
        x = x + self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x)
        x = x + self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.ffn(x)
        return x


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


class PatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


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
import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    """卷积块类
    包含一个卷积层、ReLU激活函数和批归一化层的基础模块
    
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小，默认3
        pad (int): 填充大小，默认1
        stride (int): 步长，默认1
        bias (bool): 是否使用偏置，默认True
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class BallTrackerNet(nn.Module):
    """球体追踪网络
    基于U-Net架构的深度神经网络，用于追踪视频中的球体运动
    
    参数:
        out_channels (int): 输出通道数，默认256
    """
    def __init__(self, out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        # 编码器部分（下采样路径）
        # 第一层：9通道输入（3个RGB帧叠加）
        self.conv1 = ConvBlock(in_channels=9, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2倍下采样
        
        # 第二层
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4倍下采样
        
        # 第三层
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8倍下采样

        # 瓶颈层
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)

        # 解码器部分（上采样路径）
        # 第一层上采样
        self.ups1 = nn.Upsample(scale_factor=2)  # 2倍上采样
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        
        # 第二层上采样
        self.ups2 = nn.Upsample(scale_factor=2)  # 4倍上采样
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        
        # 第三层上采样
        self.ups3 = nn.Upsample(scale_factor=2)  # 8倍上采样，恢复原始大小
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()
                  
    def forward(self, x, testing=False): 
        """前向传播函数
        
        参数:
            x (Tensor): 输入张量，形状为 [batch_size, 9, height, width]
            testing (bool): 是否为测试模式，如果是则使用softmax
            
        返回:
            out (Tensor): 输出张量，形状为 [batch_size, out_channels, height*width]
        """
        batch_size = x.size(0)
        
        # 编码器路径
        x = self.conv1(x)
        x = self.conv2(x)    
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        
        # 瓶颈层
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        
        # 解码器路径
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        
        # 重塑输出并根据测试模式决定是否使用softmax
        out = x.reshape(batch_size, self.out_channels, -1)
        if testing:
            out = self.softmax(out)
        return out                       
    
    def _init_weights(self):
        """初始化模型权重
        - 卷积层权重使用均匀分布初始化 [-0.05, 0.05]
        - BatchNorm层权重初始化为1，偏置初始化为0
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)    
    
    
if __name__ == '__main__':
    # 测试代码
    device = 'cpu'
    model = BallTrackerNet().to(device)
    # 创建测试输入：1个样本，9通道(3帧RGB图像)，360x640分辨率
    inp = torch.rand(1, 9, 360, 640)
    out = model(inp)
    print('out = {}'.format(out.shape))
    
    

import torch.nn as nn

import torch

class CustomOp(nn.Module):
    def __init__(self):
        super(CustomOp, self).__init__()
        # 可以在这里初始化需要用到的参数（如 nn.Parameter）

    def forward(self, x):
        # 实现自定义算子的前向逻辑
        # 这里只是举例：简单地对输入进行平方处理
        return x ** 2
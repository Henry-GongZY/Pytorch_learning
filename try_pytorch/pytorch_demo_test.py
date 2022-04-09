import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_demo_model import LeNet

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 标准化 output = (input- 0.5)/0.5
)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

net = LeNet()

net.load_state_dict(torch.load('Lenet.pth'))  # 载入权重文件

im = Image.open('img.png')
im = transform(im)  # [C, H, W] 转成Pytorch的Tensor格式
im = torch.unsqueeze(im, dim=0)  # [N, C, H, W] 对数据增加一个新维度

with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].data.numpy()
print(classes[int(predict)])

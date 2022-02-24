import torchvision
import torch
import torchvision.models as models
import onnx
import onnxruntime

resnet18 = models.resnet18(pretrained=True).cuda() #下载pytorch模型，如果环境没有gpu,就把cuda()去掉
 
input_name = ['input']
output_name = ['output']
input = torch.randn(1, 3, 224, 224).cuda()  #可以自定义输入数据的shape，如果环境没有gpu,就把cuda()去掉
onnx_resnet18=torch.onnx.export(resnet18, input, 'resnet18.onnx', input_names=input_name, output_names=output_name, verbose=True)  #把pytorch模型转换成onnx模型（x.pth转成x.onnx）

test = onnx.load('resnet18.onnx')
onnx.checker.check_model(test)
print("==> Passed")
#没有报错并打印出“==> Passed”，这表示你第一步已经成功完成了
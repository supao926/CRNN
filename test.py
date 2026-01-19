import torch
print(torch.__version__)
print(torch.cuda.is_available())  # 應該是 True
print(torch.version.cuda)  # 應該顯示 12.8
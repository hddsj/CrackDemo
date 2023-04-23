# 作者: hxd
# 2023年04月18日14时57分05秒
import torch
import torch.nn as nn

if __name__ == '__main__':
    correct = 0
    total = 0
    a = torch.tensor([[0.6721, 0.3279],
            [0.2018, 0.7982],
            [0.7320, 0.2680]])
    b = torch.tensor([0, 1, 1])
    _, predicted = torch.max(a, 1)
    total += b.size(0)
    correct += (predicted == b).sum().item()
    accuracy = 100 * correct / total
    print('测试集准确率: {:.2f}%'.format(accuracy))

# 作者: hxd
# 2023年04月18日17时26分15秒

# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(unet.parameters(), lr=learning_rate)
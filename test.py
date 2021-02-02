import torch
torch.set_printoptions(precision=32)

a = torch.randn((1,5))
b = torch.randn((5,1024))
def prints(num):
    in_a = torch.stack((a,)*int(num)).squeeze(1)
    out = torch.matmul(in_a,b)
    print(out[0,1])

for i in range(129):
    print("{} test".format(i+1), sep=' ')
    prints(int(i+1))
    torch.bmm

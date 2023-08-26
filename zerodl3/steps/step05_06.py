import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None # 逆伝搬で受け取った勾配を保持

class Function:
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x) # 具体的な計算はforwardメソッドで行う
        output = Variable(y)
        self.input = input # 入力を記憶
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def forward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
if __name__ == '__main__':
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    # 順方向のネットワーク構築
    a = A(x)
    b = B(a)
    y = C(b)

    # 逆伝搬
    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad) # dy / db
    a.grad = B.backward(b.grad) # dy / da
    x.grad = A.backward(a.grad) # dy / dx

    print(x.grad) # これがstep4と一致する



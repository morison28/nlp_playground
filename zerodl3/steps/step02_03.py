import numpy as np

from step01 import Variable


class Function:
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x) # 具体的な計算はforwardメソッドで行う
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
if __name__ == '__main__':
    print('----------- step2 -----------')
    x = Variable(np.array(10))
    f = Square()
    print(type(f))
    y = f(x)
    print(type(y))
    print(y.data)

    print('----------- step3 -----------')
    # 関数の連結を試してみる
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    y = C(B(A(x)))
    print(y.data)


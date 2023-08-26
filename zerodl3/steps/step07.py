import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None # 逆伝搬で受け取った勾配を保持
        self.creator = None # 自身を生み出した関数を記憶

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator # 自身を生み出した関数
        if f is not None:
            x = f.input # その関数の入力
            x.grad = f.backward(self.grad) # 勾配はその関数の微分に自身の勾配をかけたもの
            x.backward() # これを再帰的に呼ぶことで自動化する


class Function:
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x) # 具体的な計算はforwardメソッドで行う
        output = Variable(y)
        output.set_creator(self) # 出力に自分を教える
        self.input = input # 入力を記憶
        self.output = output # 出力も記憶
        
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
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

    a = A(x)
    b = B(a)
    y = C(b)

    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x

    # Try 逆伝搬
    y.grad = np.array(1.0)
    C = y.creator # 関数の取得
    b = C.input # 関数の入力の取得
    b.grad = C.backward(y.grad)
    B = b.creator
    a = B.input
    a.grad = B.backward(b.grad)
    A = a.creator
    x = A.input
    x.grad = A.backward(a.grad)
    print(x.grad) # OK!
    # 同じ処理を繰り返しているのでここも自動化する(Variableにbackwardを与える)

    # 逆伝搬の自動化
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad) # OK!
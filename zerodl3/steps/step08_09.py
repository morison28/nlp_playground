import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")
        self.data = data
        self.grad = None # 逆伝搬で受け取った勾配を保持
        self.creator = None # 自身を生み出した関数を記憶

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        # 勾配の初期化を省略するためのコード
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        # 再帰を使わないstep8
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # リストの末尾の取得
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)

        # 再帰を使うstep7 
        # f = self.creator # 自身を生み出した関数
        # if f is not None:
        #     x = f.input # その関数の入力
        #     x.grad = f.backward(self.grad) # 勾配はその関数の微分に自身の勾配をかけたもの
        #     x.backward() # これを再帰的に呼ぶことで自動化する

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x) # 具体的な計算はforwardメソッドで行う
        output = Variable(as_array(y))
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

# step9 関数をインスタンスとして呼び出すのは不格好なので以下を実装
def square(x: Variable):
    return Square()(x)

def exp(x: Variable):
    return Exp()(x)

if __name__ == '__main__':
    # 入力さえ与えれば逆伝搬をすべてに求めることができた
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    print(x.grad) # OK!

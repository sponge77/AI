import sys
import os
from collections import OrderedDict
import pickle
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class Relu:
    def __init__(self):
        self.mask = None #인스턴스 변수
        #mask는 True/False로 구성된 넘파이 배열이다

    def forward(self, x):
        self.mask=(x<=0)
        out=x.copy()
        out[self.mask]=0
        #x의 원소 값이 0 이하인 인덱스는 True, 그 외는 False로 유지한다
        return out

    def backward(self, dout):
        dout[self.mask]=0
        dx=dout
        #순전파 때의 입력 값이 0 이하면 역전파 때의 값은 0이 되어야 한다
        return dx

#Sigmoid 계층
class CustomActivation:
    def __init__(self):
        self.out=None
        

    def forward(self, x): #순전파
        out=1/(1+np.exp(-x))
        self.out=out
       
        return out

    def backward(self, dout): #역전파
        dx=dout*(1.0-self.out)*self.out #순전파의 출력을 통해 계산한다

        return dx
   


class Affine:
    def __init__(self, W, b):
        self.W=W
        self.b=b
        self.x=None
        self.dW=None
        self.db=None

    def forward(self, x): #순전파
        self.x=x
        out=np.dot(x,self.W)+self.b #Affine계층의 계산

        return out

    def backward(self, dout): #역전파
        dx=np.dot(dout,self.W.T) #행렬 간의 계산을 수행한다
        self.dW=np.dot(self.x.T,dout) #행렬 간의 계산을 수행한다
        self.db=np.sum(dout,axis=0) #axis=0이므로 행을 기준으로 더한다

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss=None
        self.y=None
        self.t=None

    def forward(self, x, t):
        self.t=t
        self.y=softmax(x) #softmax 계층을 사용한다
        self.loss=cross_entropy_error(self.y,self.t) #cross entropy error 계층을 사용한다

        return self.loss

    def backward(self, dout=1):
        batch_size=self.t.shape[0] #데이터 1개당 오차를 앞 계층으로 전파하기 위해 배치의 수가 필요
        dx=(self.y-self.t)/batch_size #softmax 계층의 출력과 정답 레이블의 차분

        return dx


class SGD: #확률적 경사 하강법, 지그재그 모양을  그리며 이동하므로 비효율적이다
    def __init__(self, lr=0.01):
        self.lr = lr #learning rate

    def update(self, params, grads): #params와 grads는 딕셔너리 변수이다
        for key in params.keys():
                params[key]-=self.lr*grads[key] #딕셔너리 형태

#class AdaGrad
class CustomOptimizer: #개별 매개변수에 적응적으로 학습률을 조정하면서 학습을 진행한다
    def __init__(self,lr=0.01):
        self.lr=lr
        self.h=None
    def update(self,params,grads):
        if self.h is None:
            self.h={}
            for key,val in params.items():
                self.h[key]=np.zeros_like(val) #해당 배열의 값은 모두 0으로 생성
        for key in params.keys():
            self.h[key]+=grads[key]*grads[key] #h는 기존 기울기 값을 제곱하여 계속 더해줌
            params[key]-=self.lr*grads[key]/(np.sqrt(self.h[key])+1e-7) #학습률을 조정한다


class Model:
    """
    네트워크 모델 입니다.

    """
    def __init__(self, lr=0.01):
        """
        클래스 초기화
        """

        self.params = {}
        self.__init_weight()
        self.__init_layer()
        self.optimizer = CustomOptimizer(lr) #AdaGrad를 사용한다

    def __init_layer(self):
        """
        레이어를 생성하시면 됩니다.
        """
        self.layers=OrderedDict()
        self.layers['Affine1']=Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1']=Relu()
        self.layers['Affine2']=Affine(self.params['W2'],self.params['b2'])
        
        self.last_layer=SoftmaxWithLoss() #마지막 계층은 SoftmaxWithLoss를 사용한다

    def __init_weight(self,):
        """
        레이어에 탑재 될 파라미터들을 초기화 하시면 됩니다.
        """
        input_size=6 #입력층의 뉴런 수는 6개이다
        hidden_size=10 #은닉층의 뉴런 수는 10으로 설정했다
        output_size=6 #출력층의 뉴런 수는 6개이다
        weight_init_std=0.01 #초깃값 정해주기
        self.params={} #매개변수 초기화
        self.params['W1']=weight_init_std*np.random.randn(input_size, hidden_size) #난수를 생성하여 가중치 초기값에 곱한다
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std*np.random.randn(hidden_size,output_size) #난수를 생성하여 가중치 초기값에 곱한다
        self.params['b2']=np.zeros(output_size)

    def update(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구한 뒤
         옵티마이저 클래스를 사용해서 네트워크 파라미터를 업데이트 해주는 함수입니다.

        :param x: train_data
        :param t: test_data
        """
        grads = self.gradient(x, t)
        self.optimizer.update(self.params, grads) 

    def predict(self, x): #예측(추론)을 수행한다
        """
        데이터를 입력받아 정답을 예측하는 함수입니다.

        :param x: data
        :return: predicted answer
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t): #손실 함수를 구한다
        """
        데이터와 레이블을 입력받아 로스를 구하는 함수입니다.
        :param x: data
        :param t: data_label
        :return: loss
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t) #순전파


    def gradient(self, x, t): #가중치 매개변수의 기울기를 오차역전파법으로 구한다
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구하는 함수입니다.
        첫번째로 받은데이터를 forward propagation 시키고,
        두번째로 back propagation 시켜 grads에 미분값을 리턴합니다.
        :param x: data
        :param t: data_label
        :return: grads
        """
        # forward
        self.loss(x,t)
        # backward
        dout=1
        dout=self.last_layer.backward(dout)

        layers=list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout=layer.backward(dout)
        # 결과 저장
        grads = {}
        grads['W1']=self.layers['Affine1'].dW
        grads['b1']=self.layers['Affine1'].db
        grads['W2']=self.layers['Affine2'].dW
        grads['b2']=self.layers['Affine2'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        """
        네트워크 파라미터를 피클 파일로 저장하는 함수입니다.

        :param file_name: 파라미터를 저장할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        """
        저장된 파라미터를 읽어와 네트워크에 탑재하는 함수입니다.

        :param file_name: 파라미터를 로드할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        for i,key in enumerate(['Affine1','Affine2']):
            self.layers[key].W=self.params['W'+str(i+1)]
            self.layers[key].b=self.params['b'+str(i+1)]

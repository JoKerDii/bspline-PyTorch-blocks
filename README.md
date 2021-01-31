# BSpline PyTorch Blocks

A customized PyTorch Layer and a customized PyTorch Activation Function using B-spline transformation. They can be easily incorporated into any neural network architecture in PyTorch. Both CPU and GPU are supported.



## B-Spline Layer

BSpline Layer consists of two steps: B-spline expansion and weighted summation. The shape of input could be (N, L, *, C). The shape of output is (N, L, *, C, n_bases). Plus, combining B-Spline Layer with any activation function (e.g. ReLU or Sigmoid) is not recommended, since B-spline is already adding pretty much non-linearity into the model.

Implemented in:

* BSpline.py
* BSplineLayer.py
* EncodeSplines.py
* helper.py



## B-Spline Activation Function

B-Spline Activation Function consists of forward and backward computation applying [De Boor algorithm](https://en.wikipedia.org/wiki/De_Boor%27s_algorithm). The shape of input could be (N, C) or (N, C, H, W) from `nn.Linear()` or `nn.Conv2d` respectively. Any other shape of input is doable if the tensor is reshaped properly before the activation.

Implemented in:

* BSplineActivation.py
* BSplineActivationFunc.py



## Example

A simplest example of using B-Spline Layer:

```python
import torch

m = BSplineLayer(4, 4, n_bases=6, shared_weights=True,bias=False, weighted_sum=False)
input = torch.randn(100, 162, 4)
output = m(input)
print(output.size())
```

An MLP example of using B-Spline Layer:

```python
import torch

class BSpline_MLP(nn.Module): 
    def __init__(self):
        super(BSpline_MLP, self).__init__()
        self.in_channels = 4 # 4 channels (C)
        self.readlength = 162 # length of 162 (L)
        self.batch_size = 64 # batch_size of 64 (N)
        self.output = 1 # a regression problem
        self.bs = BSplineLayer(in_features=self.in_channels) # use default values for other arguments
        self.fc1 = nn.Linear(self.in_channels * self.readlength, self.output)

    def forward(self, input):
        x = self.bs(input)  # （N, L, C）
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))  # (N, L * C)
        x = self.fc1(x)  # (N, 1)
        return x
```



A simplest example of using B-Spline Activation Function:

```python
m = BSplineActivation(num_activations=4)
input = torch.randn((4,4,5,5))
output = m(input)
```



An MLP example of using B-Spline Activation Function:

```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)  # (N, 28 * 28) -> (N, 256)
        self.fc2 = nn.Linear(256, 128)  # -> (N, 128)
        self.fc3 = nn.Linear(128, 64)  # -> (N, 64)
        self.fc4 = nn.Linear(64, 10)  # -> (N, 10)
        self.a1 = BSplineActivation(num_activations=256,
                                    mode='linear', device='cuda:0')
        self.a2 = torch.nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.a1(self.fc1(x))
        x = self.a2(self.fc2(x))
        x = self.a2(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x
```



An CNN example of using B-Spline Activation Function:

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.c1 = 6
        self.conv1 = nn.Conv2d(1, self.c1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self.c1 * 12 * 12, 512)  # 864
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.a1 = BSplineActivation(
            num_activations=self.c1, device='cuda:0')
        self.a2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.a1(x)
        x = self.pool(x)
        x = x.view(-1, self.c1 * 12 * 12)
        x = self.a2(self.fc1(x))
        x = self.a2(self.fc2(x))
        x = self.a2(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x
```

Please see test_af.py



## Environment

python 3.8

PyTorch 1.7.0



## References

Thanks to the ideas and code from:

- Concise: Keras extension for regulatory genomics [[doc&code](https://www.cmm.in.tum.de/public/docs/concise/)] [[paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5905632/)]
- DeepSpline: customized activation functions [[code](https://github.com/joaquimcampos/DeepSplines/tree/32e54e5de5a20e6c45ebf14e7501562170277715)]

* B-spline in machine learning [[code](https://github.com/AndreDouzette/BsplineNetworks)] [[paper](https://www.duo.uio.no/bitstream/handle/10852/61162/thesisDouzette.pdf?sequence=1)]
* B-spline formula [[scipy.interpolate.BSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html#scipy.interpolate.BSpline)] 

 
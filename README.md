# BSplineLayer

This is a customized PyTorch Layer using B-spline transformation. It can be easily incorporated into any neural network architecture in PyTorch. Both CPU and GPU are supported.



## Example

A simplest example:

```python
import torch

m = BSplineLayer(4, 4, n_bases=6, shared_weights=True,bias=False, weighted_sum=False)
input = torch.randn(100, 162, 4)
output = m(input)
print(output.size())
```

An MLP example:

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
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        x = self.relu(self.bs(input))  # （N, L, C）
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))  # (N, L * C)
        x = self.relu(self.fc1(x))  # (N, 1)
        return x
```



## Environment

python 3.8

pytorch 1.7.0



## References

Thanks to the ideas and code from:

- Concise: Keras extension for regulatory genomics [[doc&code](https://www.cmm.in.tum.de/public/docs/concise/)] [[paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5905632/)]
- DeepSpline: customized activation functions [[code](https://github.com/joaquimcampos/DeepSplines/tree/32e54e5de5a20e6c45ebf14e7501562170277715)]

* B-spline in machine learning [[code](https://github.com/AndreDouzette/BsplineNetworks)] [[paper](https://www.duo.uio.no/bitstream/handle/10852/61162/thesisDouzette.pdf?sequence=1)]
* B-spline formula [[scipy.interpolate.BSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html#scipy.interpolate.BSpline)] 

 
# BSplineLayer

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from EncodeSplines import *
from helper import *


class BSplineLayer(nn.Module):
    """
    Applies a B-spline transformation to the input data.

    # BSpline formula
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html#scipy.interpolate.BSpline

    # Shape:
        - Input: (N, *, in_features)
        - weight: (n_basis, in_features)
            only when the weights are shared, the in_features are set to be 1.
        - Bias: (out_features, )
        - Output: (N, *, out_features)
            if the weighted_sum is True, the out_features equals to in_features
            else the out_features equals to n_basis * in_features

    # Example:
    >>> m = BSplineLayer(4, 4, n_bases=6, shared_weights=True,bias=False, weighted_sum=False)
    >>> input = torch.randn(100, 162, 4)
    >>> output = m(input)
    >>> print(output.size())

    """

    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        n_bases: int = 5,
        shared_weights: bool = False,
        bias: bool = True,
        weighted_sum=True,
    ):
        super(BSplineLayer, self).__init__()
        self.n_bases = n_bases
        # self.inp_shape = None  # 4
        self.in_features = in_features  # channels
        self.hidden_features = self.in_features * self.n_bases
        self.weighted_sum = weighted_sum
        if out_features == None:
            self.out_features = self.in_features
        else:
            self.out_features = out_features
        self.shared_weights = shared_weights
        if self.shared_weights:
            self.weight = Parameter(torch.Tensor(self.n_bases, 1), requires_grad=True)
        else:
            self.weight = Parameter(
                torch.Tensor(self.n_bases, self.in_features), requires_grad=True
            )
        if bias:
            self.bias = Parameter(
                torch.Tensor(
                    self.out_features,
                ),
                requires_grad=True,
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias != None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        produce output of BSplineLayer
        """
        N = input.shape[0]
        L = input.shape[1]
        expanded_features = self.expanded_hidden_basis_layer(input)

        if self.weighted_sum:
            return self.shrunk_hidden_basis_layer(
                expanded_features, self.weight, self.bias
            )
        else:
            return expanded_features.reshape((N, L, -1))

    def expanded_hidden_basis_layer(self, input):
        """
        input should be of the shape (batch size, length, channal/feature)
        can be the output of BSpline Layer, depending on the 'weighted_sum'
        argument.
        """
        cuda = False
        if input.is_cuda:
            cuda = True
        expanded_features = EncodeSplines(n_bases=self.n_bases).fit_transform(
            input
        )  # need import that class
        if cuda:
            return expanded_features.to("cuda")
        else:
            return expanded_features

    def shrunk_hidden_basis_layer(self, input, weight, bias=None):
        N = len(input.shape)

        # print("type(input of shrunk function): ", type(input))
        # if weights are shared
        if self.shared_weights:
            if bias != None:
                out = torch.add(bias, (torch.squeeze(input.matmul(weight), -1)))
                return out
            else:
                return torch.squeeze(input.matmul(weight), -1)
        # if weights are not shared
        else:
            # torch.Size([4, 10, 1])
            weight = weight.unsqueeze(-1).permute(1, 0, 2)
            # transpose: put feature/channel (-2) to the front
            # permute: # (2, 0, 1, 3) when N = 4
            # torch.Size([4, 100, 162, 10])
            out = input.permute((N - 2,) + tuple(range(N - 2)) + (N - 1,))
            if bias != None:
                output = torch.add(bias, corr2d_stack(out, weight))
                return output
            else:
                return corr2d_stack(out, weight)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


def corr2d_stack(X, K):
    """iterate through the 0th dimension (channel dimension) of `X` and
    `K`. multiply them and stack together
    """
    corr2d_stack = torch.stack([x.matmul(k) for x, k in zip(X, K)]).squeeze(-1)
    corr2d_stack = corr2d_stack.permute((1, 2, 0))
    return corr2d_stack


if __name__ == "__main__":

    # file_name = "ERR188021Aligned.sortedByCoord.out"  # should be assigend here
    # seq = "star_RNAseq"  # should be assiged here
    # root = "/data/zhendi/wei"  # should be fixed here
    # readlength = 162  # should be assigned in config
    # label_name = "count_5"  # should be assigned in config |"count_overlap"
    # model_name = "CNN"  # should be assigned in config
    # print("load data...")
    # # df = pd.read_pickle("/home/zhendi/wei/scripts/baseline/rnaseq021_0.01.pkl")
    # with open("/home/zhendi/wei/scripts/baseline/rnaseq021_0.01.pkl", 'rb') as fh:
    #     df = pickle.load(fh)
    # # one-hot encoding
    # cube = np.eye(4)[np.asarray(df.iloc[:, 1: 1 + readlength])]
    # print("Shape of one-hot features: ", cube.shape)

    # # append label of interest
    # labels = np.array(df[label_name]).astype("float32")
    # print("type(labels): ", type(labels))

    # cube_tensor = torch.tensor(cube)

    m = BSplineLayer(
        4, 4, n_bases=6, shared_weights=True, bias=False, weighted_sum=False
    )
    # (batch size, length, channel, basis functions)
    input = torch.randn(100, 162, 4)
    output = m(input)
    print(output.size())

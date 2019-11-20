from mxnet import nd
from mxnet.gluon import nn

layer = nn.Dense(2)
print(layer)

layer.intializer()

x = nd.random.uniform(-1,1,(3,4))

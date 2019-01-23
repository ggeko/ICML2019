import numpy as np
import chainer
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.dataset import concat_examples


############################################################################
# from chainer.backends import cuda
# cpu = cuda.to_cpu
############################################################################


############################################################################
############################################################################
class Base(chainer.Chain):
    def __init__(self, n_input, n_output, stride, dropout):
        w = chainer.initializers.HeNormal()
        super(Base, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                n_input, n_output, 3, stride, 1, nobias=True, initialW=w)
            self.conv2 = L.Convolution2D(
                n_output, n_output, 3, 1, 1, nobias=True, initialW=w)
            self.bn1 = L.BatchNormalization(n_input)
            self.bn2 = L.BatchNormalization(n_output)
            if n_input != n_output:
                self.shortcut = L.Convolution2D(
                    n_input, n_output, 1, stride, nobias=True, initialW=w)
        self.dropout = dropout

    def __call__(self, x):
        h = self.bn1(x)
        h = F.relu(h)
        h = self.conv1(h)
        h = self.bn2(h)
        h = F.relu(h)
        if self.dropout:
            h = F.dropout(h)
        h = self.conv2(h)
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        return h + shortcut

    def get_act(self, x, act):
        ####################################################################
        h = self.bn1(x)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.conv1(h)
        h = self.bn2(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.conv2(h)
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        ####################################################################
        return h + shortcut, act

    def cutoff(self, x, cut, cid):
        ####################################################################
        h = self.bn1(x)
        h = h * cut[cid]
        cid += 1
        h = self.conv1(h)
        h = self.bn2(h)
        h = h * cut[cid]
        cid += 1
        h = self.conv2(h)
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        ####################################################################
        return h + shortcut, cid

    
class Block(chainer.Chain):
    def __init__(self, n_input, n_output, count, stride, dropout):
        super(Block, self).__init__()
        with self.init_scope():
            self.base0 = Base(n_input, n_output, stride, dropout)
            self._forward = ['base0']
            for i in range(1, count):
                name = 'base{}'.format(i)
                base =  Base(n_output, n_output, 1, dropout) 
                setattr(self, name, base)
                self._forward.append(name)
            # end for

    def __call__(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
        return x

    def get_act(self, x, act):
        for name in self._forward:
            l = getattr(self, name)
            x, act = l.get_act(x, act)
        return x, act

    def cutoff(self, x, cut, cid):
        for name in self._forward:
            l = getattr(self, name)
            x, cid = l.cutoff(x, cut, cid)
        return x, cid

    
class ResNet(chainer.Chain):
    def __init__(self, w_factor=10, depth=28, num_classes=10, dropout=True):
        k = w_factor
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        n_stages = [16, 16 * k, 32 * k, 64 * k]
        hw = chainer.initializers.HeNormal()
        lw = chainer.initializers.LeCunNormal()
        super(ResNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                None, n_stages[0], 3, 1, 1, nobias=True, initialW=hw)
            self.res2 = Block(n_stages[0], n_stages[1], n, 1, dropout)
            self.res3 = Block(n_stages[1], n_stages[2], n, 2, dropout)
            self.res4 = Block(n_stages[2], n_stages[3], n, 2, dropout)
            self.bn5 = L.BatchNormalization(n_stages[3])
            self.fin = L.Linear(n_stages[3], num_classes, initialW=lw)
        # end with
        self.output = None

    def __call__(self, x):
        h = self.conv1(x)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = F.relu(self.bn5(h))
        h = F.average_pooling_2d(h, (h.shape[2], h.shape[3]))
        return self.fin(h)

    def get_act(self, x):
        act = []
        h = self.conv1(x)
        h, act = self.res2.get_act(h, act)
        h, act = self.res3.get_act(h, act)
        h, act = self.res4.get_act(h, act)
        h = self.bn5(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, (h.shape[2], h.shape[3]))
        self.output = (self.fin(h)).data
        return act

    def cutoff(self, x, cut):
        cid = 0
        h = self.conv1(x)
        h, cid = self.res2.cutoff(h, cut, cid)
        h, cid = self.res3.cutoff(h, cut, cid)
        h, cid = self.res4.cutoff(h, cut, cid)
        h = self.bn5(h)
        h = h * cut[cid]
        h = F.average_pooling_2d(h, (h.shape[2], h.shape[3]))
        return self.fin(h)
############################################################################
############################################################################
class ResNet10(ResNet):
    def __init__(self, w_factor=1, num_classes=10):
        super(ResNet10, self).__init__(w_factor, 10, num_classes)
############################################################################
############################################################################
class ResNet16(ResNet):
    def __init__(self, w_factor=1, num_classes=10):
        super(ResNet16, self).__init__(w_factor, 16, num_classes)
############################################################################
############################################################################
class ResNet22(ResNet):
    def __init__(self, w_factor=1, num_classes=10):
        super(ResNet22, self).__init__(w_factor, 22, num_classes)
############################################################################
############################################################################
class ResNet28(ResNet):
    def __init__(self, w_factor=1, num_classes=10):
        super(ResNet28, self).__init__(w_factor, 28, num_classes)
############################################################################
############################################################################
class ResNet34(ResNet):
    def __init__(self, w_factor=1, num_classes=10):
        super(ResNet34, self).__init__(w_factor, 34, num_classes)
############################################################################
############################################################################
class ResNet40(ResNet):
    def __init__(self, w_factor=1, num_classes=10):
        super(ResNet40, self).__init__(w_factor, 40, num_classes)
############################################################################
############################################################################
class ResNet46(ResNet):
    def __init__(self, w_factor=1, num_classes=10):
        super(ResNet46, self).__init__(w_factor, 46, num_classes)
############################################################################
############################################################################
class ResNet52(ResNet):
    def __init__(self, w_factor=1, num_classes=10):
        super(ResNet52, self).__init__(w_factor, 52, num_classes)
############################################################################
############################################################################
class MLP(chainer.Chain):
    def __init__(self, nlayers, nunits, nclasses):
        hw = chainer.initializers.HeNormal()
        lw = chainer.initializers.HeNormal(scale=1.0/np.sqrt(2.0))
        # lw = chainer.initializers.LeCunNormal()
        super(MLP, self).__init__()
        with self.init_scope():
            ################################################################
            self._forward = []
            ################################################################
            for i in range(nlayers - 1):
                name = 'l{}'.format(i)
                proj = L.Linear(None, nunits, initialW=hw)
                setattr(self, name, proj)
                self._forward.append(name)
            # end for
            ################################################################
            self.fin = L.Linear(None, nclasses, initialW=lw)
            ################################################################
        # end with
        self.output = None

    def __call__(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
            x = F.relu(x)
        return self.fin(x)

    def get_act(self, x):
        act = []
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
            act.append(x.data > 0)
            x = F.relu(x)
        self.output = (self.fin(x)).data
        return act

    def cutoff(self, x, cut):
        for cid, name in enumerate(self._forward):
            l = getattr(self, name)
            x = l(x) * cut[cid]
        return self.fin(x)
############################################################################
############################################################################
class ConvNet2(chainer.Chain):
    def __init__(self, w_factor=1, num_classes=10, drop_rate=None):
        k = w_factor
        hw = chainer.initializers.HeNormal()
        lw = chainer.initializers.LeCunNormal()
        super(ConvNet2, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(3, 16 * k, 3, 2, 1, initialW=hw)
            self.conv1 = L.Convolution2D(None, 32 * k, 3, 2, 1, initialW=hw)
            self.fin   = L.Linear(None, num_classes, initialW=lw)
            
        self.output = None

    def __call__(self, x):
        h = self.conv0(x)
        h = F.relu(h)
        h = self.conv1(h)
        h = F.relu(h)
        return self.fin(h)

    def get_act(self, x):
        act = []
        h = self.conv0(x)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.conv1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        self.output = (self.fin(h)).data
        return act

    def cutoff(self, x, cut):
        cid = 0
        h = self.conv0(x) * cut[cid]
        cid = 1
        h = self.conv1(h) * cut[cid]
        return self.fin(h)
############################################################################
############################################################################
class ConvNet4(chainer.Chain):
    def __init__(self, w_factor=1, num_classes=10, drop_rate=None):
        k = w_factor
        hw = chainer.initializers.HeNormal()
        lw = chainer.initializers.LeCunNormal()
        super(ConvNet4, self).__init__()
        with self.init_scope():
            self._forward = ['conv0', 'conv1', 'conv2', 'conv3']
            self.conv0 = L.Convolution2D(None, 16 * k, 3, 2, 1, initialW=hw)
            self.conv1 = L.Convolution2D(None, 16 * k, 3, 1, 1, initialW=hw)
            self.conv2 = L.Convolution2D(None, 32 * k, 3, 2, 1, initialW=hw)
            self.conv3 = L.Convolution2D(None, 32 * k, 3, 1, 1, initialW=hw)
            self.fin   = L.Linear(None, num_classes, initialW=lw)
            
        self.output = None

    def __call__(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
            x = F.relu(x)
        return self.fin(x)
        
    def get_act(self, x):
        act = []
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
            act.append(x.data > 0)
            x = F.relu(x)
        self.output = (self.fin(x)).data
        return act
        
    def cutoff(self, x, cut):
        for cid, name in enumerate(self._forward):
            l = getattr(self, name)
            x = l(x) * cut[cid]
        return self.fin(x)  
############################################################################
############################################################################
class ConvNet6(chainer.Chain):
    def __init__(self, w_factor=1, num_classes=10):
        k = w_factor
        hw = chainer.initializers.HeNormal()
        lw = chainer.initializers.LeCunNormal()
        super(ConvNet6, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(None,  16 * k, 3, 2, 1, initialW=hw)
            self.conv1 = L.Convolution2D(None,  16 * k, 3, 1, 1, initialW=hw)
            self.conv2 = L.Convolution2D(None,  32 * k, 3, 2, 1, initialW=hw)
            self.conv3 = L.Convolution2D(None,  32 * k, 3, 1, 1, initialW=hw)
            self.conv4 = L.Convolution2D(None,  64 * k, 3, 2, 1, initialW=hw)
            self.conv5 = L.Convolution2D(None,  64 * k, 3, 1, 1, initialW=hw)
            self.fin   = L.Linear(None, num_classes, initialW=lw)
            self._forward = ['conv0', 'conv1', 'conv2',
                             'conv3', 'conv4', 'conv5']
            
        self.output = None

    def __call__(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
            x = F.relu(x)
        # end for
        # h = F.average_pooling_2d(x, x.shape[2:], stride=1)
        return self.fin(x)
        
    def get_act(self, x):
        act = []
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
            act.append(x.data > 0)
            x = F.relu(x)
        # end for
        # h = F.average_pooling_2d(x, x.shape[2:], stride=1)
        self.output = (self.fin(x)).data
        return act
        
    def cutoff(self, x, cut):
        for cid, name in enumerate(self._forward):
            l = getattr(self, name)
            x = l(x) * cut[cid]
        # end for
        # h = F.average_pooling_2d(x, x.shape[2:], stride=1)
        return self.fin(x) 
############################################################################
############################################################################
class VGG(chainer.Chain):
    def __init__(self, num_classes=10):
        hw = chainer.initializers.HeNormal()
        lw = chainer.initializers.LeCunNormal()
        super(VGG, self).__init__()
        with self.init_scope():
            ################################################################
            self._forward = ['conv']
            self.conv1_1 = L.Convolution2D(None, 64, 3, 1, 1, initialW=hw)
            self.bn1_1 = L.BatchNormalization(64)
            self.conv1_2 = L.Convolution2D(None, 64, 3, 1, 1, initialW=hw)
            self.bn1_2 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D(None, 128, 3, 1, 1, initialW=hw)
            self.bn2_1 = L.BatchNormalization(128)
            self.conv2_2 = L.Convolution2D(None, 128, 3, 1, 1, initialW=hw)
            self.bn2_2 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(None, 256, 3, 1, 1, initialW=hw)
            self.bn3_1 = L.BatchNormalization(256)
            self.conv3_2 = L.Convolution2D(None, 256, 3, 1, 1, initialW=hw)
            self.bn3_2 = L.BatchNormalization(256)
            self.conv3_3 = L.Convolution2D(None, 256, 3, 1, 1, initialW=hw)
            self.bn3_3 = L.BatchNormalization(256)
            self.conv3_4 = L.Convolution2D(None, 256, 3, 1, 1, initialW=hw)
            self.bn3_4 = L.BatchNormalization(256)

            self.fc4 = L.Linear(None, 1024, initialW=hw)
            self.fc5 = L.Linear(None, 1024, initialW=hw)
            self.fin = L.Linear(None, num_classes, initialW=lw)
            
        self.output = None


    def __call__(self, x):
        h = self.bn1_1(self.conv1_1(x))
        h = F.relu(h)
        h = self.bn1_2(self.conv1_2(h))
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.bn2_1(self.conv2_1(h))
        h = F.relu(h)
        h = self.bn2_2(self.conv2_2(h))
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.bn3_1(self.conv3_1(h))
        h = F.relu(h)
        h = self.bn3_2(self.conv3_2(h))
        h = F.relu(h)
        h = self.bn3_3(self.conv3_3(h))
        h = F.relu(h)
        h = self.bn3_4(self.conv3_4(h))
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fc5(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fin(h)
        return h

    def get_act(self, x):
        act = []
        h = self.bn1_1(self.conv1_1(x))
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.bn1_2(self.conv1_2(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.bn2_1(self.conv2_1(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.bn2_2(self.conv2_2(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.bn3_1(self.conv3_1(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.bn3_2(self.conv3_2(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.bn3_3(self.conv3_3(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.bn3_4(self.conv3_4(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        self.output = (self.fin(h)).data
        return act

    def cutoff(self, x, cut):
        h = self.bn1_1(self.conv1_1(x)) * cut[0]
        h = self.bn1_2(self.conv1_2(h)) * cut[1]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.bn2_1(self.conv2_1(h)) * cut[2]
        h = self.bn2_2(self.conv2_2(h)) * cut[3]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.bn3_1(self.conv3_1(h)) * cut[4]
        h = self.bn3_2(self.conv3_2(h)) * cut[5]
        h = self.bn3_3(self.conv3_3(h)) * cut[6]
        h = self.bn3_4(self.conv3_4(h)) * cut[7]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h) * cut[8]
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h) * cut[9]
        # h = F.dropout(h, ratio=0.5)
        return self.fin(h)
############################################################################
############################################################################
class VGG_WOB(chainer.Chain):
    def __init__(self, wide_factor=1, num_classes=10):
        k = wide_factor
        hw = chainer.initializers.HeNormal()
        lw = chainer.initializers.LeCunNormal()
        super(VGG_WOB, self).__init__()
        with self.init_scope():
            ################################################################
            self.conv1_1 = L.Convolution2D(None, 64*k, 3, 1, 1, initialW=hw)
            # self.bn1_1 = L.BatchNormalization(64)
            self.conv1_2 = L.Convolution2D(None, 64*k, 3, 1, 1, initialW=hw)
            # self.bn1_2 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D(None, 128*k, 3, 1, 1, initialW=hw)
            # self.bn2_1 = L.BatchNormalization(128)
            self.conv2_2 = L.Convolution2D(None, 128*k, 3, 1, 1, initialW=hw)
            # self.bn2_2 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(None, 256*k, 3, 1, 1, initialW=hw)
            # self.bn3_1 = L.BatchNormalization(256)
            self.conv3_2 = L.Convolution2D(None, 256*k, 3, 1, 1, initialW=hw)
            # self.bn3_2 = L.BatchNormalization(256)
            self.conv3_3 = L.Convolution2D(None, 256*k, 3, 1, 1, initialW=hw)
            # self.bn3_3 = L.BatchNormalization(256)
            self.conv3_4 = L.Convolution2D(None, 256*k, 3, 1, 1, initialW=hw)
            # self.bn3_4 = L.BatchNormalization(256)

            self.fc4 = L.Linear(None, 1024, initialW=hw)
            self.fc5 = L.Linear(None, 1024, initialW=hw)
            self.fin = L.Linear(None, num_classes, initialW=lw)
            
        self.output = None


    def __call__(self, x):
        h = self.conv1_1(x)
        # h = self.bn1_1(h)
        h = F.relu(h)
        h = self.conv1_2(h)
        # h = self.bn1_2(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h)
        # h = self.bn2_1(h)
        h = F.relu(h)
        h = self.conv2_2(h)
        # h = self.bn2_2(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h)
        # h = self.bn3_1(h)
        h = F.relu(h)
        h = self.conv3_2(h)
        # h = self.bn3_2(h)
        h = F.relu(h)
        h = self.conv3_3(h)
        # h = self.bn3_3(h)
        h = F.relu(h)
        h = self.conv3_4(h)
        # h = self.bn3_4(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        h = self.fc5(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        return self.fin(h)

    def get_act(self, x):
        act = []
        h = self.conv1_1(x)
        # h = self.bn1_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.conv1_2(h)
        # h = self.bn1_2(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h)
        # h = self.bn2_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.conv2_2(h)
        # h = self.bn2_2(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h)
        # h = self.bn3_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.conv3_2(h)
        # h = self.bn3_2(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.conv3_3(h)
        # h = self.bn3_3(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.conv3_4(h)
        # h = self.bn3_4(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        self.output = (self.fin(h)).data
        return act

    def cutoff(self, x, cut):
        h = self.conv1_1(x) * cut[0]
        h = self.conv1_2(h) * cut[1]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h) * cut[2]
        h = self.conv2_2(h) * cut[3]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h) * cut[4]
        h = self.conv3_2(h) * cut[5]
        h = self.conv3_3(h) * cut[6]
        h = self.conv3_4(h) * cut[7]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h) * cut[8]
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h) * cut[9]
        # h = F.dropout(h, ratio=0.5)
        return self.fin(h)
############################################################################
############################################################################
class VGG_C6L3(chainer.Chain):
    def __init__(self, num_classes=10):
        hw = chainer.initializers.HeNormal()
        lw = chainer.initializers.LeCunNormal()
        super(VGG_C6L3, self).__init__()
        with self.init_scope():
            ################################################################
            self._forward = ['conv']
            self.conv1_1 = L.Convolution2D(None, 64, 3, 1, 1, initialW=hw)
            self.bn1_1 = L.BatchNormalization(64)
            self.conv1_2 = L.Convolution2D(None, 64, 3, 1, 1, initialW=hw)
            self.bn1_2 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D(None, 128, 3, 1, 1, initialW=hw)
            self.bn2_1 = L.BatchNormalization(128)
            self.conv2_2 = L.Convolution2D(None, 128, 3, 1, 1, initialW=hw)
            self.bn2_2 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(None, 256, 3, 1, 1, initialW=hw)
            self.bn3_1 = L.BatchNormalization(256)
            self.conv3_2 = L.Convolution2D(None, 256, 3, 1, 1, initialW=hw)
            self.bn3_2 = L.BatchNormalization(256)

            self.fc4 = L.Linear(None, 1024, initialW=hw)
            self.fc5 = L.Linear(None, 1024, initialW=hw)
            self.fin = L.Linear(None, num_classes, initialW=lw)
            
        self.output = None


    def __call__(self, x):
        h = self.bn1_1(self.conv1_1(x))
        h = F.relu(h)
        h = self.bn1_2(self.conv1_2(h))
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.bn2_1(self.conv2_1(h))
        h = F.relu(h)
        h = self.bn2_2(self.conv2_2(h))
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.bn3_1(self.conv3_1(h))
        h = F.relu(h)
        h = self.bn3_2(self.conv3_2(h))
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fc5(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fin(h)
        return h

    def get_act(self, x):
        act = []
        h = self.bn1_1(self.conv1_1(x))
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.bn1_2(self.conv1_2(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.bn2_1(self.conv2_1(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.bn2_2(self.conv2_2(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.bn3_1(self.conv3_1(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.bn3_2(self.conv3_2(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        self.output = (self.fin(h)).data
        return act

    def cutoff(self, x, cut):
        h = self.bn1_1(self.conv1_1(x)) * cut[0]
        h = self.bn1_2(self.conv1_2(h)) * cut[1]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.bn2_1(self.conv2_1(h)) * cut[2]
        h = self.bn2_2(self.conv2_2(h)) * cut[3]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.bn3_1(self.conv3_1(h)) * cut[4]
        h = self.bn3_2(self.conv3_2(h)) * cut[5]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h) * cut[6]
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h) * cut[7]
        # h = F.dropout(h, ratio=0.5)
        return self.fin(h)
############################################################################
############################################################################
class VGG_C6L3_S(chainer.Chain):
    def __init__(self, num_classes=10):
        hw = chainer.initializers.HeNormal()
        lw = chainer.initializers.LeCunNormal()
        super(VGG_C6L3_S, self).__init__()
        with self.init_scope():
            ################################################################
            self._forward = ['conv']
            self.conv1_1 = L.Convolution2D(None, 37, 3, 1, 1, initialW=hw)
            self.bn1_1 = L.BatchNormalization(37)
            self.conv1_2 = L.Convolution2D(None, 58, 3, 1, 1, initialW=hw)
            self.bn1_2 = L.BatchNormalization(58)

            self.conv2_1 = L.Convolution2D(None, 114, 3, 1, 1, initialW=hw)
            self.bn2_1 = L.BatchNormalization(114)
            self.conv2_2 = L.Convolution2D(None, 117, 3, 1, 1, initialW=hw)
            self.bn2_2 = L.BatchNormalization(117)

            self.conv3_1 = L.Convolution2D(None, 229, 3, 1, 1, initialW=hw)
            self.bn3_1 = L.BatchNormalization(229)
            self.conv3_2 = L.Convolution2D(None, 236, 3, 1, 1, initialW=hw)
            self.bn3_2 = L.BatchNormalization(236)

            self.fc4 = L.Linear(None, 10, initialW=hw)
            self.fc5 = L.Linear(None, 10, initialW=hw)
            self.fin = L.Linear(None, num_classes, initialW=lw)
            
        self.output = None


    def __call__(self, x):
        h = self.bn1_1(self.conv1_1(x))
        h = F.relu(h)
        h = self.bn1_2(self.conv1_2(h))
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.bn2_1(self.conv2_1(h))
        h = F.relu(h)
        h = self.bn2_2(self.conv2_2(h))
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.bn3_1(self.conv3_1(h))
        h = F.relu(h)
        h = self.bn3_2(self.conv3_2(h))
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fc5(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fin(h)
        return h

    def get_act(self, x):
        act = []
        h = self.bn1_1(self.conv1_1(x))
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.bn1_2(self.conv1_2(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.bn2_1(self.conv2_1(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.bn2_2(self.conv2_2(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.bn3_1(self.conv3_1(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.bn3_2(self.conv3_2(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        self.output = (self.fin(h)).data
        return act

    def cutoff(self, x, cut):
        h = self.bn1_1(self.conv1_1(x)) * cut[0]
        h = self.bn1_2(self.conv1_2(h)) * cut[1]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.bn2_1(self.conv2_1(h)) * cut[2]
        h = self.bn2_2(self.conv2_2(h)) * cut[3]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.bn3_1(self.conv3_1(h)) * cut[4]
        h = self.bn3_2(self.conv3_2(h)) * cut[5]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h) * cut[6]
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h) * cut[7]
        # h = F.dropout(h, ratio=0.5)
        return self.fin(h)
############################################################################
############################################################################
class VGG_C6L3_WOB(chainer.Chain):
    def __init__(self, wide_factor=1, num_classes=10):
        k = wide_factor
        hw = chainer.initializers.HeNormal()
        lw = chainer.initializers.LeCunNormal()
        super(VGG_C6L3_WOB, self).__init__()
        with self.init_scope():
            ################################################################
            self._forward = ['conv']
            self.conv1_1 = L.Convolution2D(None, 64*k, 3, 1, 1, initialW=hw)
            # self.bn1_1 = L.BatchNormalization(64)
            self.conv1_2 = L.Convolution2D(None, 64*k, 3, 1, 1, initialW=hw)
            # self.bn1_2 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D(None, 128*k, 3, 1, 1, initialW=hw)
            # self.bn2_1 = L.BatchNormalization(128)
            self.conv2_2 = L.Convolution2D(None, 128*k, 3, 1, 1, initialW=hw)
            # self.bn2_2 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(None, 256*k, 3, 1, 1, initialW=hw)
            # self.bn3_1 = L.BatchNormalization(256)
            self.conv3_2 = L.Convolution2D(None, 256*k, 3, 1, 1, initialW=hw)
            # self.bn3_2 = L.BatchNormalization(256)

            self.fc4 = L.Linear(None, 1024, initialW=hw)
            self.fc5 = L.Linear(None, 1024, initialW=hw)
            self.fin = L.Linear(None, num_classes, initialW=lw)
            
        self.output = None


    def __call__(self, x):
        h = self.conv1_1(x)
        # h = self.bn1_1(h)
        h = F.relu(h)
        h = self.conv1_2(h)
        # h = self.bn1_2(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h)
        # h = self.bn2_1(h)
        h = F.relu(h)
        h = self.conv2_2(h)
        # h = self.bn2_2(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h)
        # h = self.bn3_1(h)
        h = F.relu(h)
        h = self.conv3_2(h)
        # h = self.bn3_2(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fc5(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fin(h)
        return h

    def get_act(self, x):
        act = []
        h = self.conv1_1(x)
        # h = self.bn1_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.conv1_2(h)
        # h = self.bn1_2(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h)
        # h = self.bn2_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.conv2_2(h)
        # h = self.bn2_2(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h)
        # h = self.bn3_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.conv3_2(h)
        # h = self.bn3_2(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        self.output = (self.fin(h)).data
        return act

    def cutoff(self, x, cut):
        h = self.conv1_1(x) 
        # h = self.bn1_1(h)
        h = h * cut[0]
        h = self.conv1_2(h)
        # h = self.bn1_2(h)
        h = h * cut[1]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h)
        # h = self.bn2_1(h)
        h = h * cut[2]
        h = self.conv2_2(h)
        # h = self.bn2_2(h)
        h = h * cut[3]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h)
        # h = self.bn3_1(h)
        h = h * cut[4]
        h = self.conv3_2(h)
        # h = self.bn3_2(h)
        h = h * cut[5]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h) * cut[6]
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h) * cut[7]
        # h = F.dropout(h, ratio=0.5)
        return self.fin(h)
############################################################################
############################################################################
class VGG_C6L3_S_WOB(chainer.Chain):
    def __init__(self, num_classes=10):
        hw = chainer.initializers.HeNormal()
        lw = chainer.initializers.LeCunNormal()
        super(VGG_C6L3_S_WOB, self).__init__()
        with self.init_scope():
            ################################################################
            self._forward = ['conv']
            self.conv1_1 = L.Convolution2D(None, 37, 3, 1, 1, initialW=hw)
            # self.bn1_1 = L.BatchNormalization(37)
            self.conv1_2 = L.Convolution2D(None, 58, 3, 1, 1, initialW=hw)
            # self.bn1_2 = L.BatchNormalization(58)

            self.conv2_1 = L.Convolution2D(None, 114, 3, 1, 1, initialW=hw)
            # self.bn2_1 = L.BatchNormalization(114)
            self.conv2_2 = L.Convolution2D(None, 117, 3, 1, 1, initialW=hw)
            # self.bn2_2 = L.BatchNormalization(117)

            self.conv3_1 = L.Convolution2D(None, 229, 3, 1, 1, initialW=hw)
            # self.bn3_1 = L.BatchNormalization(229)
            self.conv3_2 = L.Convolution2D(None, 236, 3, 1, 1, initialW=hw)
            # self.bn3_2 = L.BatchNormalization(236)

            self.fc4 = L.Linear(None, 10, initialW=hw)
            self.fc5 = L.Linear(None, 10, initialW=hw)
            self.fin = L.Linear(None, num_classes, initialW=lw)
            
        self.output = None


    def __call__(self, x):
        h = self.conv1_1(x)
        # h = self.bn1_1(h)
        h = F.relu(h)
        h = self.conv1_2(h)
        # h = self.bn1_2(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h)
        # h = self.bn2_1(h)
        h = F.relu(h)
        h = self.conv2_2(h)
        # h = self.bn2_2(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h)
        # h = self.bn3_1(h)
        h = F.relu(h)
        h = self.conv3_2(h)
        # h = self.bn3_2(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fc5(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fin(h)
        return h

    def get_act(self, x):
        act = []
        h = self.conv1_1(x)
        # h = self.bn1_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.conv1_2(h)
        # h = self.bn1_2(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h)
        # h = self.bn2_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.conv2_2(h)
        # h = self.bn2_2(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h)
        # h = self.bn3_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.conv3_2(h)
        # h = self.bn3_2(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        self.output = (self.fin(h)).data
        return act

    def cutoff(self, x, cut):
        h = self.conv1_1(x) 
        # h = self.bn1_1(h)
        h = h * cut[0]
        h = self.conv1_2(h)
        # h = self.bn1_2(h)
        h = h * cut[1]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h)
        # h = self.bn2_1(h)
        h = h * cut[2]
        h = self.conv2_2(h)
        # h = self.bn2_2(h)
        h = h * cut[3]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h)
        # h = self.bn3_1(h)
        h = h * cut[4]
        h = self.conv3_2(h)
        # h = self.bn3_2(h)
        h = h * cut[5]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h) * cut[6]
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h) * cut[7]
        # h = F.dropout(h, ratio=0.5)
        return self.fin(h)
############################################################################
############################################################################
class VGG_C5L3_WOB(chainer.Chain):
    def __init__(self, wide_factor=1, num_classes=10):
        k = wide_factor
        hw = chainer.initializers.HeNormal()
        lw = chainer.initializers.LeCunNormal()
        super(VGG_C5L3_WOB, self).__init__()
        with self.init_scope():
            ################################################################
            self._forward = ['conv']
            self.conv1_1 = L.Convolution2D(None, 64*k, 3, 1, 1, initialW=hw)
            # self.bn1_1 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D(None, 128*k, 3, 1, 1, initialW=hw)
            # self.bn2_1 = L.BatchNormalization(128)
            self.conv2_2 = L.Convolution2D(None, 128*k, 3, 1, 1, initialW=hw)
            # self.bn2_2 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(None, 256*k, 3, 1, 1, initialW=hw)
            # self.bn3_1 = L.BatchNormalization(256)
            self.conv3_2 = L.Convolution2D(None, 256*k, 3, 1, 1, initialW=hw)
            # self.bn3_2 = L.BatchNormalization(256)

            self.fc4 = L.Linear(None, 1024, initialW=hw)
            self.fc5 = L.Linear(None, 1024, initialW=hw)
            self.fin = L.Linear(None, num_classes, initialW=lw)
            
        self.output = None


    def __call__(self, x):
        h = self.conv1_1(x)
        # h = self.bn1_1(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h)
        # h = self.bn2_1(h)
        h = F.relu(h)
        h = self.conv2_2(h)
        # h = self.bn2_2(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h)
        # h = self.bn3_1(h)
        h = F.relu(h)
        h = self.conv3_2(h)
        # h = self.bn3_2(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fc5(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fin(h)
        return h

    def get_act(self, x):
        act = []
        h = self.conv1_1(x)
        # h = self.bn1_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h)
        # h = self.bn2_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.conv2_2(h)
        # h = self.bn2_2(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h)
        # h = self.bn3_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.conv3_2(h)
        # h = self.bn3_2(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        self.output = (self.fin(h)).data
        return act

    def cutoff(self, x, cut):
        h = self.conv1_1(x) 
        # h = self.bn1_1(h)
        h = h * cut[0]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h)
        # h = self.bn2_1(h)
        h = h * cut[1]
        h = self.conv2_2(h)
        # h = self.bn2_2(h)
        h = h * cut[2]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h)
        # h = self.bn3_1(h)
        h = h * cut[3]
        h = self.conv3_2(h)
        # h = self.bn3_2(h)
        h = h * cut[4]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h) * cut[5]
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h) * cut[6]
        # h = F.dropout(h, ratio=0.5)
        return self.fin(h)
############################################################################
############################################################################
class VGG_C4L3(chainer.Chain):
    def __init__(self, num_classes=10):
        hw = chainer.initializers.HeNormal()
        lw = chainer.initializers.LeCunNormal()
        super(VGG_C4L3, self).__init__()
        with self.init_scope():
            ################################################################
            self._forward = ['conv']
            self.conv1_1 = L.Convolution2D(None, 64, 3, 1, 1, initialW=hw)
            self.bn1_1 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D(None, 128, 3, 1, 1, initialW=hw)
            self.bn2_1 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(None, 256, 3, 1, 1, initialW=hw)
            self.bn3_1 = L.BatchNormalization(256)
            self.conv3_2 = L.Convolution2D(None, 256, 3, 1, 1, initialW=hw)
            self.bn3_2 = L.BatchNormalization(256)

            self.fc4 = L.Linear(None, 1024, initialW=hw)
            self.fc5 = L.Linear(None, 1024, initialW=hw)
            self.fin = L.Linear(None, num_classes, initialW=lw)
            
        self.output = None


    def __call__(self, x):
        h = self.bn1_1(self.conv1_1(x))
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.bn2_1(self.conv2_1(h))
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.bn3_1(self.conv3_1(h))
        h = F.relu(h)
        h = self.bn3_2(self.conv3_2(h))
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fc5(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fin(h)
        return h

    def get_act(self, x):
        act = []
        h = self.bn1_1(self.conv1_1(x))
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.bn2_1(self.conv2_1(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.bn3_1(self.conv3_1(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.bn3_2(self.conv3_2(h))
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        self.output = (self.fin(h)).data
        return act

    def cutoff(self, x, cut):
        h = self.bn1_1(self.conv1_1(x)) * cut[0]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.bn2_1(self.conv2_1(h)) * cut[1]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.bn3_1(self.conv3_1(h)) * cut[2]
        h = self.bn3_2(self.conv3_2(h)) * cut[3]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h) * cut[4]
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h) * cut[5]
        # h = F.dropout(h, ratio=0.5)
        return self.fin(h)
############################################################################
############################################################################
class VGG_C4L3_WOB(chainer.Chain):
    def __init__(self, wide_factor=1, num_classes=10):
        k = wide_factor
        hw = chainer.initializers.HeNormal()
        lw = chainer.initializers.LeCunNormal()
        super(VGG_C4L3_WOB, self).__init__()
        with self.init_scope():
            ################################################################
            self._forward = ['conv']
            self.conv1_1 = L.Convolution2D(None, 64*k, 3, 1, 1, initialW=hw)
            # self.bn1_1 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D(None, 128*k, 3, 1, 1, initialW=hw)
            # self.bn2_1 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(None, 256*k, 3, 1, 1, initialW=hw)
            # self.bn3_1 = L.BatchNormalization(256)
            self.conv3_2 = L.Convolution2D(None, 256*k, 3, 1, 1, initialW=hw)
            # self.bn3_2 = L.BatchNormalization(256)

            self.fc4 = L.Linear(None, 1024, initialW=hw)
            self.fc5 = L.Linear(None, 1024, initialW=hw)
            self.fin = L.Linear(None, num_classes, initialW=lw)
            
        self.output = None


    def __call__(self, x):
        h = self.conv1_1(x)
        # h = self.bn1_1(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h)
        # h = self.bn2_1(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h)
        # h = self.bn3_1(h)
        h = F.relu(h)
        h = self.conv3_2(h)
        # h = self.bn3_2(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fc5(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fin(h)
        return h

    def get_act(self, x):
        act = []
        h = self.conv1_1(x)
        # h = self.bn1_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h)
        # h = self.bn2_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h)
        # h = self.bn3_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.conv3_2(h)
        # h = self.bn3_2(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        self.output = (self.fin(h)).data
        return act

    def cutoff(self, x, cut):
        h = self.conv1_1(x) 
        # h = self.bn1_1(h)
        h = h * cut[0]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h)
        # h = self.bn2_1(h)
        h = h * cut[1]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h)
        # h = self.bn3_1(h)
        h = h * cut[2]
        h = self.conv3_2(h)
        # h = self.bn3_2(h)
        h = h * cut[3]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h) * cut[4]
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h) * cut[5]
        # h = F.dropout(h, ratio=0.5)
        return self.fin(h)
############################################################################
############################################################################
class VGG_C3L3_WOB(chainer.Chain):
    def __init__(self, wide_factor=1, num_classes=10):
        k = wide_factor
        hw = chainer.initializers.HeNormal()
        lw = chainer.initializers.LeCunNormal()
        super(VGG_C3L3_WOB, self).__init__()
        with self.init_scope():
            ################################################################
            self._forward = ['conv']
            self.conv1_1 = L.Convolution2D(None, 64*k, 3, 1, 1, initialW=hw)
            # self.bn1_1 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D(None, 128*k, 3, 1, 1, initialW=hw)
            # self.bn2_1 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(None, 256*k, 3, 1, 1, initialW=hw)
            # self.bn3_1 = L.BatchNormalization(256)

            self.fc4 = L.Linear(None, 1024, initialW=hw)
            self.fc5 = L.Linear(None, 1024, initialW=hw)
            self.fin = L.Linear(None, num_classes, initialW=lw)
            
        self.output = None


    def __call__(self, x):
        h = self.conv1_1(x)
        # h = self.bn1_1(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h)
        # h = self.bn2_1(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h)
        # h = self.bn3_1(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fc5(h)
        h = F.dropout(F.relu(h), ratio=0.5)
        h = self.fin(h)
        return h

    def get_act(self, x):
        act = []
        h = self.conv1_1(x)
        # h = self.bn1_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h)
        # h = self.bn2_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h)
        # h = self.bn3_1(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        self.output = (self.fin(h)).data
        return act

    def cutoff(self, x, cut):
        h = self.conv1_1(x) 
        # h = self.bn1_1(h)
        h = h * cut[0]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv2_1(h)
        # h = self.bn2_1(h)
        h = h * cut[1]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.conv3_1(h)
        # h = self.bn3_1(h)
        h = h * cut[2]
        h = F.average_pooling_2d(h, 2, 2)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)

        h = self.fc4(h) * cut[3]
        # h = F.dropout(h, ratio=0.5)
        h = self.fc5(h) * cut[4]
        # h = F.dropout(h, ratio=0.5)
        return self.fin(h)
############################################################################
############################################################################
class LeNet5(chainer.Chain):

    def __init__(self, num_classes=10):
        hw = chainer.initializers.HeNormal()
        lw = chainer.initializers.LeCunNormal()
        super(LeNet5, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, 5, 1, 2, initialW=hw)
            self.conv2 = L.Convolution2D(None, 32, 5, 1, 2, initialW=hw)
            self.conv3 = L.Convolution2D(None, 64, 5, 1, 2, initialW=hw)
            self.fc4 = L.Linear(None, 4096, initialW=hw)
            self.fin = L.Linear(None, num_classes, initialW=lw)
        self.output = None
        
    def __call__(self, x):
        h = self.conv1(x)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 3, stride=2)
        h = self.conv2(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 3, stride=2)
        h = self.conv3(h)
        h = F.relu(h)
        h = self.fc4(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        return self.fin(h)

    def get_act(self, x):
        act = []
        h = self.conv1(x)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 3, stride=2)
        h = self.conv2(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 3, stride=2)
        h = self.conv3(h)
        act.append(h.data > 0)
        h = F.relu(h)
        h = self.fc4(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        self.output = (self.fin(h)).data
        return act
    
    def cutoff(self, x, cut):
        h = self.conv1(x) * cut[0]
        h = F.average_pooling_2d(h, 3, stride=2)
        h = self.conv2(h) * cut[1]
        h = F.average_pooling_2d(h, 3, stride=2)
        h = self.conv3(h) * cut[2]
        h = self.fc4(h) * cut[3]
        # h = F.dropout(h, ratio=0.5)
        return self.fin(h)
############################################################################
############################################################################
class SmallAlexnet(chainer.Chain):
    def __init__(self, num_classes=10):
        hw = chainer.initializers.HeNormal()
        lw = chainer.initializers.LeCunNormal()
        super(SmallAlexnet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 96, 5, 1, 2, initialW=hw)
            self.conv2 = L.Convolution2D(None, 256, 5, 1, 2, initialW=hw)
            self.fc3 = L.Linear(None, 1024, initialW=hw)
            self.fc4 = L.Linear(None, 1024, initialW=hw)
            self.fin = L.Linear(None, num_classes, initialW=lw)
        self.output = None

    def __call__(self, x):
        h = self.conv1(x)
        h = F.relu(h)
        h = F.local_response_normalization(h)
        h = F.average_pooling_2d(h, 2, stride=2)

        h = self.conv2(x)
        h = F.relu(h)
        h = F.local_response_normalization(h)
        h = F.average_pooling_2d(h, 2, stride=2)
        
        h = self.fc3(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        h = self.fc4(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        return self.fin(h)

    def get_act(self, x):
        act = []
        h = self.conv1(x)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.local_response_normalization(h)
        h = F.average_pooling_2d(h, 2, stride=2)

        h = self.conv2(x)
        act.append(h.data > 0)
        h = F.relu(h)
        h = F.local_response_normalization(h)
        h = F.average_pooling_2d(h, 2, stride=2)
        
        h = self.fc3(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        h = self.fc4(h)
        act.append(h.data > 0)
        h = F.relu(h)
        # h = F.dropout(h, ratio=0.5)
        self.output = (self.fin(h)).data
        return act

    def cutoff(self, x, cut):
        h = self.conv1(x) * cut[0]
        h = F.local_response_normalization(h)
        h = F.average_pooling_2d(h, 2, stride=2)

        h = self.conv2(x) * cut[1]
        h = F.local_response_normalization(h)
        h = F.average_pooling_2d(h, 2, stride=2)
        
        h = self.fc3(h) * cut[2]
        # h = F.dropout(h, ratio=0.5)
        h = self.fc4(h) * cut[3]
        # h = F.dropout(h, ratio=0.5)
        return self.fin(h)
############################################################################
############################################################################

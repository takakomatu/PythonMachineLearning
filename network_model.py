#coding:utf-8
#!/usr/bin/env python
__author__ = 'HidetomoKataoka'

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable 
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from sklearn.datasets import fetch_mldata
import sys
import matplotlib.pyplot as plt


"""
##################################
ネットワークモデルを定義
##################################
"""


class mnist_model(chainer.Chain):
	def __init__( self ):
		super(mnist_model, self).__init__(
			l1 = L.Linear( 784 , 100 ),
			l2 = L.Linear( 100 , 100 ),
			l3 = L.Linear( 100 , 10  ),
		)

	def __call__( self, x, y ):
		fv = self.fwd(x)
		return F.softmax_cross_entropy( fv, y ), F.accuracy( fv, y )

	def fwd( self, x ):
		h1 = F.dropout( F.relu( self.l1(x) ) )
		h2 = F.dropout( F.relu( self.l2(h1) ) )
		h3 = self.l3(h2)
		return h3


class CNN(chainer.Chain):
	def __init__( self ):
		super( CNN, self ).__init__(
			# input_channel_num, output_channel_num(filter_num), filter_size
			conv1 = L.Convolution2D( 1  , 32 , 5 ),
			conv2 = L.Convolution2D( 32 , 32 , 5 ),
			l3    = L.Linear( 288 , 100 ),
			l4    = L.Linear( 100 , 10 ),
		)
	
	def __call__( self, x, y ):
		fv = self.fwd( x )
		return F.softmax_cross_entropy( fv, y ), F.accuracy( fv, y )

	def fwd( self, x ):
		h1 = F.max_pooling_2d( F.relu( self.conv1(x) ), ksize = 2, stride = 2 )
		h2 = F.max_pooling_2d( F.relu( self.conv2(h1)), ksize = 3, stride = 3 )
		h3 = F.dropout( F.relu( self.l3(h2) ) )
		h4 = self.l4(h3)

		return h4


class SRCNN(chainer.Chain):
	def __init__( self ):
		super( SRCNN, self ).__init__(
			# input_channel_num, output_channel_num(filter_num), filter_size
			conv1 = L.Convolution2D( 1  , 64 , 9, pad = int(9) / 2),
			conv2 = L.Convolution2D( 64 , 32 , 3, pad = int(3) / 2),
			conv3 = L.Convolution2D( 32 , 1  , 5, pad = int(5) / 2),
		)
	
	def __call__( self, x, y ):
		fv = self.fwd( x )
		return F.mean_squared_error( fv, y )

	def fwd( self, x ):
		h1 = F.relu(self.conv1(x))
		h2 = F.relu(self.conv2(h1))
		h3 = F.relu(self.conv3(h2))

		return h3



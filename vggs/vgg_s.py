#!/usr/bin/env python

#from paddle.trainer_config_helpers import *
import  paddle.v2 as paddle


def VGG_S(img):
	net = paddle.layer.img_conv(
	    input=img,
	    filter_size=7,
	    num_channels=3,
	    num_filters=96,
	    stride=2, act=paddle.activation.Linear())
	net = paddle.layer.img_cmrnorm(input=net, size=5, scale=0.0005, power=0.75)
	net = paddle.layer.img_pool(input=net, pool_size=3, stride=3)

	# conv2
	net = paddle.layer.img_conv(
	    input=net, filter_size=5, num_filters=256, stride=1, act=paddle.activation.Linear())
	net = paddle.layer.img_pool(input=net, pool_size=2, stride=2)

	# conv3
	net = paddle.layer.img_conv(
	    input=net, filter_size=3, num_filters=512, stride=1, padding=1,act=paddle.activation.Linear())
	# conv4
	net = paddle.layer.img_conv(
	    input=net, filter_size=3, num_filters=512, stride=1, padding=1, groups=1,act=paddle.activation.Linear())

	# conv5
	net = paddle.layer.img_conv(
	    input=net, filter_size=3, num_filters=512, stride=1, padding=1, groups=1,act=paddle.activation.Linear())
	net = paddle.layer.img_pool(input=net, pool_size=3, stride=3)

	net = paddle.layer.fc(
	    input=net,
	    size=4096,
	    act=paddle.activation.Linear(),
	    layer_attr=paddle.attr.Extra(drop_rate=0.5))
	net = paddle.layer.fc(
	    input=net,
	    size=4096,
	    act=paddle.activation.Linear(),
	    layer_attr=paddle.attr.Extra(drop_rate=0.5))
	return net

# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import sys
import gzip

import paddle.v2 as paddle

from vgg_pruning import vgg_bn_drop
from resnet_pruning import resnet_cifar10
from paddle.v2.attr import Hook
from paddle.v2.attr import ParamAttr


def main():
    datadim = 3 * 32 * 32
    classdim = 10

    # PaddlePaddle init
    paddle.init(use_gpu=True, trainer_count=1, gpu_id=1)
    momentum_optimizer = paddle.optimizer.Momentum(
	    momentum=0.9,
	    regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
	    learning_rate=0.001 / 128.0,
			learning_rate_decay_a=0.1,
					learning_rate_decay_b=50000 * 100,
							learning_rate_schedule='discexp')

    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(datadim))

    # Add neural network config
    # option 1. resnet
    # net = resnet_cifar10(image, depth=32)
    # option 2. vgg
    #net = resnet_cifar10(image)
    net = vgg_bn_drop(image)


    out = paddle.layer.fc(
        input=net, size=classdim, act=paddle.activation.Softmax(),
		param_attr = ParamAttr(update_hooks=Hook('dynamic_pruning')))

    lbl = paddle.layer.data(
        name="label", type=paddle.data_type.integer_value(classdim))
    cost = paddle.layer.classification_cost(input=out, label=lbl)
    with gzip.open('params_120.tar.gz', 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    # Create parameters
    #parameters = paddle.parameters.create(cost)

    # Create optimizer

    # End batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            sys.stdout.write('.')
            sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with gzip.open('static_pruning_params_pass__%d.tar.gz' % event.pass_id, 'w') as f:
                parameters.to_tar(f)

            result = trainer.test(
                reader=paddle.batch(
                    paddle.dataset.cifar.test10(), batch_size=128),
                feeding={'image': 0,
                         'label': 1})
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

    # Create trainer
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=momentum_optimizer)
    trainer.train(
        reader=paddle.batch(
		#	paddle.reader.shuffle(
		#		        paddle.dataset.cifar.train10(), buf_size=50000),
            paddle.reader.buffered(
                paddle.dataset.cifar.train10(), size=100000),
            batch_size=128),
        num_passes=200,
        event_handler=event_handler,
        feeding={'image': 0,
                 'label': 1})

if __name__ == '__main__':
    main()

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
# limitations under the License.

import paddle.v2 as paddle

__all__ = ['vgg_bn_drop']

from paddle.v2.attr import  Hook
from paddle.v2.attr import  ParamAttr

def vgg_bn_drop(input):
    def conv_block(ipt, num_filter, groups, dropouts, num_channels=None, param_attr = None):
        return paddle.networks.img_conv_group(
            input=ipt,
            num_channels=num_channels,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act=paddle.activation.Relu(),
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type=paddle.pooling.Max(), param_attr = param_attr)
    pa0 = ParamAttr(update_hooks = Hook('dynamic_pruning', sparsity_upper_bound=0.6))
    conv1 = conv_block(input, 64, 2, [0.3, 0], 3, param_attr=pa0)
    pa_conv = ParamAttr(update_hooks = Hook('dynamic_pruning', sparsity_upper_bound=0.85))
    conv2 = conv_block(conv1, 128, 2, [0.4, 0], param_attr = pa_conv)
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0], param_attr = pa_conv)
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0], param_attr = pa_conv)
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0], param_attr = pa_conv)

    fc1 = paddle.layer.fc(input=conv5, size=512, act=paddle.activation.Linear(),
                            param_attr = ParamAttr(update_hooks=Hook('dynamic_pruning', sparsity_upper_bound=0.97)))
    bn = paddle.layer.batch_norm(
        input=fc1,
        act=paddle.activation.Relu())
    fc2 = paddle.layer.fc(input=bn, size=512, act=paddle.activation.Linear(),
                            param_attr = ParamAttr(update_hooks=Hook('dynamic_pruning', sparsity_upper_bound=0.97)))
    return fc2

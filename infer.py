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
from mobilenet import mobile_net
from paddle.v2.topology import Topology
import cv2
import numpy as np
from alexnet import Alexnet

def main():
    datadim = 3 * 224 * 224 
    classdim = 1000

    # PaddlePaddle init
    paddle.init(use_gpu=True, trainer_count=1)

    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(datadim))

    #net = mobile_net(image)
    net = Alexnet(image)

    out = paddle.layer.fc(
        input=net, size=classdim, act=paddle.activation.Softmax())
    
    '''
    out = paddle.layer.img_conv(
                         input=net,
                         filter_size=1,
                         num_filters=classdim,
                         stride=1,
                         act=paddle.activation.Linear())
    '''
    return out


def infer(out):
    #with gzip.open('Paddle_mobilenet.tar.gz', 'r') as f:
    with gzip.open('Paddle_vgg_s_pretrained.tar.gz', 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    test_data = []
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    for i in xrange(50):
    	test_data.append((load_image(cur_dir + '/image/cat.jpg'), ))
    for i in xrange(1):
    	probs = paddle.infer(
        	output_layer=out, parameters=parameters, input=test_data)
    
    lab = np.argsort(-probs)  # probs and lab are the results of one batch data
    print probs[0].argmax()
    print "Label of image/cat is: %d" % lab[0][0]

def load_image(file, resize_size=256, crop_size=224, mean_file=None):
    # load image
    im = cv2.imread(file)
    # resize
    h, w = im.shape[:2]
    h_new, w_new = resize_size, resize_size
    if h > w:
        h_new = resize_size * h / w
    else:
        w_new = resize_size * w / h
    im = cv2.resize(im, (h_new, w_new), interpolation=cv2.INTER_CUBIC)
    # crop
    h, w = im.shape[:2]
    h_start = (h - crop_size) / 2
    w_start = (w - crop_size) / 2
    h_end, w_end = h_start + crop_size, w_start + crop_size
    im = im[h_start:h_end, w_start:w_end, :]
    # transpose to CHW order
    mean = np.array([103.94,116.78,123.68])
    im = im - mean
    im = im.transpose((2, 0, 1))

    #im = im * 0.017 
    return im


if __name__ == '__main__':
	out = main()
	infer(out)

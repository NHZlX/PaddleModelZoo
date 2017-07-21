## Static pruning

> Static pruning是对一个已经训练好的模型任务上进行fine-tune。对于一个带参数的层(`conv or Fc`)指定一个稀疏度(`sparsity_ratio`)。在网络`fine-tune`开始前，将该层的参数的绝对值进行排序，根据稀疏度(`sparsity_ratio`)将较小的参数给剪切掉，然后进行`fine-tune`。整个`pruning`的过程分为多个阶段，因为`sparsity_ratio`是逐步递增的， 每变化一次`sparsity_ratio`， 网络都要重新启动。关于pruning的更多信息，可以查看 `https://arxiv.org/pdf/1506.02626.pdf`

### Usage：
`修改vgg_pruning.py中的带参数层的稀疏度`

`python train_pruning.py` 来进行第一轮训练， 收敛后，停止网络。 重新修改稀疏度，然后继续下一轮，直至达到期望的稀疏度
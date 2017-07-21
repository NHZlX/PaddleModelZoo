## Dynamic pruning
>还在PR中，https://github.com/PaddlePaddle/Paddle/pull/2603
>
>so，如果想试用的话，git clone https://github.com/NHZlX/Paddle.git  
>并且 git checkout auto_pruning 
>
> 和`Static pruning`不同的是，`Dynamic pruning`的整个`fine-tune`过程为一个阶段。`Dynamic pruning` 是对一个已经训练好的模型任务上进行fine-tune。首先为每层指定一个稀疏度（`sparsity_ratio`）的`sparsity_upper_bound`，表示该层最终要达到的稀疏度。从开始到训练结束，稀疏度从0逐渐增加到`sparsity_upper_ bound`。为了避免稀疏度变化过于频繁， 每interval_pass个pass 变化一次(`通过对一些数据集（flowers102 etc）测试，每次变化一次`sparsity_ratio`，`interval_pass`设置为3基本可以微调到好的效果，其他具体任务还得进一步测试`），一共变化`end_pass`/`interval_pass`次，稀疏度（`sparsity_ratio`）变化的曲线如下 ![](https://raw.githubusercontent.com/NHZlX/Auto_pruning/master/examples/photo/log.png)


### Usage：
`指定 vgg_pruning.py 没层最终达到的稀疏度`

[Download pretrained model from BaiduCloud](https://pan.baidu.com/s/1slyiDL7)

`python train_pruning.py` 

使用残差神经网络实现一个分类问题，家人们

图片数据集不好找，我们直接使用随机生成的一些数来当作数据集和测试集

因为生成的数都是单独的也就是1*1的所以咱们的分类任务足够简单只使用全连接层就够用

残差块是三层全连接

主网络由三个残差块组成

那就实现一个奇数和偶数的分类问题吧

介绍一下文件奥
main是主函数文件，训练模型和测试
NetWork是agen和网络的类
model文件夹存的当然是模型了

因为最开始发现只输入一个数进行训练不行

因为只有一个数的话特征太少了，对于网络来说没啥特征可以找

所以我把每一个数都换成了二进制文件传入智能体进行训练

结果表明，只需要十次训练就可以达到100%，效果很好

因为数据很少所有就没有写GPU训练，直接使用默认的cpu进行训练了

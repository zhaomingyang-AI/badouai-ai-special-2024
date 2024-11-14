import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 首先生成了随机的输入数据 X 和二分类的目标数据 y。
# 生成随机数据
# X = np.random.rand(1000, 10)和y = np.random.randint(2, size=(1000, 1))：
# X是一个形状为 (1000, 10) 的二维数组，代表 1000 个样本，每个样本有 10 个特征。通过np.random.rand函数生成随机数据，这些数据在 0 到 1 之间均匀分布。
# y是一个形状为 (1000, 1) 的二维数组，代表 1000 个样本的标签，每个标签是 0 或 1（二分类问题）。通过np.random.randint(2, size=(1000, 1))生成随机的 0 或 1 整数。
np.random.seed(0)
X = np.random.rand(1000, 10)
y = np.random.randint(2, size=(1000, 1))

# 然后创建了一个顺序模型，添加了两个隐藏层，第一个隐藏层有 64 个神经元，激活函数为 relu，第二个隐藏层有 32 个神经元，激活函数也为 relu。
# 输出层有一个神经元，激活函数为 sigmoid，适用于二分类问题。
# 创建模型

model = Sequential()
#创建一个顺序模型，即一种线性堆叠层的模型结构。
model.add(Dense(64, activation='relu', input_dim=10))
# Dense(64)表示添加一个全连接层，该层有 64 个神经元。
# activation='relu'指定该层的激活函数为relu（Rectified Linear Unit），它在正半轴是线性的，负半轴输出为 0，能够引入非线性因素，有助于模型学习复杂的模式。
# input_dim=10表示输入层的维度为 10，与生成的输入数据X的特征数量相对应。
model.add(Dense(32, activation='relu'))
#添加另一个全连接层，有 32 个神经元，激活函数同样为relu。
model.add(Dense(1, activation='sigmoid'))
#最后添加一个全连接层，只有 1 个神经元，激活函数为sigmoid。在二分类问题中，这个输出层的神经元的输出值在 0 到 1 之间，可以解释为属于某一类别的概率。
# 编译模型时使用 adam 优化器、binary_crossentropy 损失函数，并监控准确率。
# optimizer='adam'指定使用Adam优化器来更新模型的权重。Adam是一种自适应学习率的优化算法，结合了动量和自适应学习率的优点，在很多深度学习任务中表现良好。
# loss='binary_crossentropy'表示使用二元交叉熵作为损失函数，适用于二分类问题。它衡量了模型预测的概率分布与真实标签之间的差异。
# metrics=['accuracy']指定在训练和评估过程中监控准确率这个指标。
# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 最后使用 fit 方法训练模型，指定训练轮数为 50，批次大小为 32，并通过 evaluate 方法评估模型在训练集上的性能。
# 训练模型
model.fit(X, y, epochs=50, batch_size=32)
# X和y是训练数据和对应的标签。
# epochs=50表示进行 50 个训练轮次，即整个训练数据集在模型中经过 50 次迭代。
# batch_size=32表示每次训练时使用 32 个样本组成一个批次进行更新权重。
# 评估模型
score = model.evaluate(X, y)
#使用训练数据X和标签y对模型进行评估，返回损失值和指定的指标值（在这个例子中是准确率）。
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#打印出评估结果中的损失值和准确率。score[0]是损失值，score[1]是准确率。
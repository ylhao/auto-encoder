# coding: utf-8


from keras.datasets import mnist
import numpy as np
# 服务器端使用 matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras.utils import plot_model


(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
print('x_train.shape:', x_train.shape)
print('x_test.shape:', x_test.shape)


noise_factor = 0.5
# numpy.random.normal(loc=0.0, scale=1.0, size=None) 生成符合高斯分布的随机数
# loc: 均值
# scale: 标准差
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
# numpy.clip(a, a_min, a_max, out=None)
# 这个函数将数组中的元素限制在 a_min，a_max 之间，大于 a_max 的就使得它等于 a_max，小于 a_min 的就使得它等于 a_min
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
print('x_train_noisy.shape:', x_train_noisy.shape)
print('x_test_noisy.shape:', x_test_noisy.shape)


n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    # plt.subplot(2, 2, 1) 表示将整个图像窗口分为 2 行 2 列, 当前位置为 1.
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    # 单通道黑白图
    plt.gray()
    # 不显示坐标
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# plt.show()
plt.savefig('x_test_noisy_0_9.png')


# encode
input_img = Input(shape=(28, 28, 1,)) # N * 28 * 28 * 1
x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_img) # 28 * 28 * 32
x = MaxPooling2D((2, 2), padding='same')(x) # 14 * 14 * 32
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x) # 14 * 14 * 32
encoded = MaxPooling2D((2, 2), padding='same')(x) # 7 * 7 * 32

# decode
x = Conv2D(32, (3, 3), padding='same', activation='relu')(encoded) # 7 * 7 * 32
x = UpSampling2D((2, 2))(x) # 14 * 14 * 32
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x) # 14 * 14 * 32
x = UpSampling2D((2, 2))(x) # 28 * 28 * 32
decoded = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x) # 28 * 28 * 1


autoencoder = Model(input_img, decoded)
autoencoder.summary()
plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)  # 绘制模型图
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')  # compile
autoencoder.fit(x_train_noisy, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))
autoencoder.save('autoencoder.h5')  # 保存模型


model = load_model('autoencoder.h5')  # 加载模型
decoded_imgs = autoencoder.predict(x_test_noisy)
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # 显示加入了噪声的图片
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 显示去噪以后的图片
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# plt.show()
plt.savefig('result.png')


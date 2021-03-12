
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
train_path = r'D:\myGit\face_identity\12306-CAPTCHA\data\words\train'
test_path = r'D:\myGit\face_identity\12306-CAPTCHA\data\words\test'
npz_file = 'images.npz'


def get_images(path):
    labels = []
    images = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".jpg"):
                img_path = os.path.join(root, f)
                label = os.path.basename(root)
                img = plt.imread(img_path)
                images.append(img)
                labels.append(label)
    return np.array(labels), np.array(images)


def save_img_to_npz():
    train_labels, train_images = get_images(train_path)
    test_labels, test_images = get_images(test_path)
    np.savez(npz_file, train_labels, train_images, test_labels, test_images)


def load_img_from_npz():
    r = np.load(npz_file)
    return r['arr_0'], r['arr_1'], r['arr_2'], r['arr_3']


class MyModel(Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 2, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(240, activation='relu')
    self.d2 = Dense(80, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    print('-', x.shape)
    x = self.flatten(x)
    print('--', x.shape)
    x = self.d1(x)
    print('----', x.shape)
    x = self.d2(x)
    print('----', x.shape)
    return x


def train_my_model(train_ds, test_ds):
    model = MyModel()
    model_save_path = "test1.model"

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 100
    train_a = []
    test_a = []
    for epoch in range(EPOCHS):
        # 在下一个epoch开始时，重置评估指标
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            # print(images.shape, labels.shape)
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
        train_a.append(train_accuracy.result() * 100)
        test_a.append(test_accuracy.result() * 100)
        if epoch % 5 == 0:
            plt.plot(test_a)
            plt.savefig('rate.jpg')
            plt.show()
        if int(test_a[-1]) > 90:
            print("save: %s.model" % epoch)
            model.save(os.path.join("models", "model%s" % epoch))
    model.save(os.path.join("models", "model%s" % epoch))
    # model.save(model_save_path)


def get_y_label(label_index, labels):
    l = len(label_index.keys())
    y_labels = []
    y = []
    eye = np.eye(l, l)
    for label in labels:
        index = label_index[label]
        y_label = eye[index]
        y_labels.append(y_label)
        y.append(index)
    return np.array(y)


if __name__ == '__main__':
    # save_img_to_npz()
    train_labels, train_images, test_labels, test_images = load_img_from_npz()
    all_label = ['中国结', '仪表盘', '公交卡', '冰箱', '创可贴', '刺绣', '剪纸', '印章', '卷尺', '双面胶', '口哨', '啤酒',
                  '安全帽', '开瓶器', '手掌印', '打字机', '护腕', '拖把', '挂钟', '排风机', '文具盒', '日历', '本子', '档案袋',
                  '棉棒', '樱桃', '毛线', '沙包', '沙拉', '海报', '海苔', '海鸥', '漏斗', '烛台', '热水袋', '牌坊', '狮子',
                  '珊瑚', '电子秤', '电线', '电饭煲', '盘子', '篮球', '红枣', '红豆', '红酒', '绿豆', '网球拍', '老虎', '耳塞',
                  '航母', '苍蝇拍', '茶几', '茶盅', '药片', '菠萝', '蒸笼', '薯条', '蚂蚁', '蜜蜂', '蜡烛', '蜥蜴', '订书机',
                  '话梅', '调色板', '跑步机', '路灯', '辣椒酱', '金字塔', '钟表', '铃铛', '锅铲', '锣', '锦旗', '雨靴', '鞭炮',
                  '风铃', '高压锅', '黑板', '龙舟']
    label_index = {}
    for i, label in enumerate(all_label):
        label_index[label] = i
    y_train = get_y_label(label_index, train_labels)
    y_test = get_y_label(label_index, test_labels)
    # for label in train_labels:
    #     if label not in all_labals:
    #         all_labals.append(label)
    # print(all_labals)
    # sys.exit()
    # img = train_images[0]
    # print(train_labels.shape, train_images.shape, img.shape)
    # plt.imshow(img)
    # print("label %s" % train_labels[0])
    # img = test_images[0]
    # print(test_labels.shape, test_images.shape, img.shape)
    # plt.imshow(img)
    # print("label %s" % test_labels[0])
    # plt.show()

    x_train, x_test = train_images / 255.0, test_images / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(300)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(10000).batch(300)
    train_my_model(train_ds, test_ds)




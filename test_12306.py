#coding: utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import requests
import uuid
import os
import re
import base64
import time

plt.rcParams['font.sans-serif'] = ['SimHei']

model_save_path = r'D:\myGit\12306\12306.image.model.h5'
model_save_path = r'D:\myGit\12306\model.v2.0.h5'
model_save_path = os.path.join("models", "model13")#'10.model'
test_jpg = r'D:\myGit\face_identity\12306验证图片\xx.jpg'
test_jpg = r'D:\myGit\face_identity\12306验证图片\xx3.jpg'
img = plt.imread(test_jpg)
# img = cv2.imread(test_jpg, cv2.IMREAD_GRAYSCALE)
# with open(test_jpg, 'rb') as fh:
#     data = fh.read()

model = tf.keras.models.load_model(model_save_path)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
# x_train = x_train[..., tf.newaxis]
# x_test = x_test[..., tf.newaxis]

# train_ds = tf.data.Dataset.from_tensor_slices(
#     (x_train, y_train)).shuffle(10).batch(32)
# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

def test():
    offset = 0
    url = r'https://kyfw.12306.cn/otn/resources/login.html'
    url = r'https://kyfw.12306.cn/passport/captcha/captcha-image64?login_site=E&module=login&rand=sjrand&1615339291663&callback=jQuery1910048662488071605337_1615339239718&_=1615339239720'

    save_path = '12306验证图片'
    while True:
        try:
            r = requests.request('GET', url)
            text = r.text
            search_obj = re.search(r'"image":"([^"]+)', text)
            if search_obj:
                uname = '%s.jpg' % str(uuid.uuid4())
                img_base64 = search_obj.groups()[0]
                jpg = os.path.join(save_path, uname)
                with open(jpg, 'wb') as fh:
                    fh.write(base64.b64decode(img_base64))

                test_jpg = jpg
                img = plt.imread(test_jpg)
                new_img1 = img[3:27, 120 + offset:180 + offset]
                gray_img = tf.image.rgb_to_grayscale(img)
                new_img = gray_img[3:27, 120 + offset:180 + offset] / 255
                mm = np.array([new_img])
                print(new_img.shape)
                prediction = model(mm)
                verify_titles = ['中国结', '仪表盘', '公交卡', '冰箱', '创可贴', '刺绣', '剪纸', '印章', '卷尺', '双面胶', '口哨', '啤酒',
                                  '安全帽', '开瓶器', '手掌印', '打字机', '护腕', '拖把', '挂钟', '排风机', '文具盒', '日历', '本子', '档案袋',
                                  '棉棒', '樱桃', '毛线', '沙包', '沙拉', '海报', '海苔', '海鸥', '漏斗', '烛台', '热水袋', '牌坊', '狮子',
                                  '珊瑚', '电子秤', '电线', '电饭煲', '盘子', '篮球', '红枣', '红豆', '红酒', '绿豆', '网球拍', '老虎', '耳塞',
                                  '航母', '苍蝇拍', '茶几', '茶盅', '药片', '菠萝', '蒸笼', '薯条', '蚂蚁', '蜜蜂', '蜡烛', '蜥蜴', '订书机',
                                  '话梅', '调色板', '跑步机', '路灯', '辣椒酱', '金字塔', '钟表', '铃铛', '锅铲', '锣', '锦旗', '雨靴', '鞭炮',
                                  '风铃', '高压锅', '黑板', '龙舟']
                label = tf.argmax(prediction, 1)
                text = verify_titles[int(label)]
                plt.imshow(new_img)
                try:
                    os.makedirs(text)
                except:
                    pass
                plt.title("predict: %s" % text)
                plt.show()
                print(text)
                text_save_path = os.path.join(text, uname)
                plt.imsave(text_save_path, new_img1, cmap="gray")
                # time.sleep(1)
        except:
            print("Get data from 12306 error, wait....")
            time.sleep(2)

            # p = r'D:\myGit\face_identity\12306-CAPTCHA\data\words\train\日历\20171215095532_0.jpg'
            # p = r'D:\myGit\face_identity\12306-CAPTCHA\data\words\train\铃铛\20171215090438_0.jpg'
            # img = plt.imread(p)
            # img = img / 255.0
            # img = img[tf.newaxis, ..., tf.newaxis]
            # prediction = model(img)
            # label = tf.argmax(prediction, 1)
            # text1 = verify_titles[int(label)]
# plt.imshow(gray_img)
# plt.imshow(img)
# plt.show()
# for test_images, test_labels in test_ds:
#     for index, prediction in enumerate(predictions):
#         max = 0
#         value = -1
#         for i, v in enumerate(prediction):
#             if v > max:
#                 max = v
#                 value = i
#         vv = int(test_labels[index])
#         if vv != value:
#             print("%s prediction: %s" % (str(vv), value))
    #         plt.imshow(test_images[index])
    #         plt.title("%s prediction: %s" % (str(vv), value))
    # plt.show()
    # test_step(test_images, test_labels)

# print(text, text1)#, test_loss.result(), test_accuracy.result()*100)
npz_file = 'images.npz'


def load_img_from_npz():
    r = np.load(npz_file)
    return r['arr_0'], r['arr_1'], r['arr_2'], r['arr_3']

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


def test_1():
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
        (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    for images, labels in test_ds:
        for image in images:
            mm = np.array([image])
            prediction = model(mm)
            verify_titles = ['中国结', '仪表盘', '公交卡', '冰箱', '创可贴', '刺绣', '剪纸', '印章', '卷尺', '双面胶', '口哨', '啤酒',
                             '安全帽', '开瓶器', '手掌印', '打字机', '护腕', '拖把', '挂钟', '排风机', '文具盒', '日历', '本子', '档案袋',
                             '棉棒', '樱桃', '毛线', '沙包', '沙拉', '海报', '海苔', '海鸥', '漏斗', '烛台', '热水袋', '牌坊', '狮子',
                             '珊瑚', '电子秤', '电线', '电饭煲', '盘子', '篮球', '红枣', '红豆', '红酒', '绿豆', '网球拍', '老虎', '耳塞',
                             '航母', '苍蝇拍', '茶几', '茶盅', '药片', '菠萝', '蒸笼', '薯条', '蚂蚁', '蜜蜂', '蜡烛', '蜥蜴', '订书机',
                             '话梅', '调色板', '跑步机', '路灯', '辣椒酱', '金字塔', '钟表', '铃铛', '锅铲', '锣', '锦旗', '雨靴', '鞭炮',
                             '风铃', '高压锅', '黑板', '龙舟']
            label = tf.argmax(prediction, 1)
            text = verify_titles[int(label)]
            plt.imshow(image)
            plt.title("predict: %s" % text)
            plt.show()
            print(images.shape, labels.shape)


if __name__ == '__main__':
    test()
    # test_1()

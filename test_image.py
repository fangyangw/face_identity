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

model_save_path = os.path.join("models", "image_model")
text_model_save_path = os.path.join("models", "text_model2")

model = tf.keras.models.load_model(model_save_path)
text_model = tf.keras.models.load_model(text_model_save_path)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
npz_file = 'picture_images.npz'
verify_titles = [
    '中国结', '仪表盘', '公交卡', '冰箱', '创可贴', '刺绣', '剪纸', '印章', '卷尺', '双面胶', '口哨', '啤酒',
     '安全帽', '开瓶器', '手掌印', '打字机', '护腕', '拖把', '挂钟', '排风机', '文具盒', '日历', '本子', '档案袋',
     '棉棒', '樱桃', '毛线', '沙包', '沙拉', '海报', '海苔', '海鸥', '漏斗', '烛台', '热水袋', '牌坊', '狮子',
     '珊瑚', '电子秤', '电线', '电饭煲', '盘子', '篮球', '红枣', '红豆', '红酒', '绿豆', '网球拍', '老虎', '耳塞',
     '航母', '苍蝇拍', '茶几', '茶盅', '药片', '菠萝', '蒸笼', '薯条', '蚂蚁', '蜜蜂', '蜡烛', '蜥蜴', '订书机',
     '话梅', '调色板', '跑步机', '路灯', '辣椒酱', '金字塔', '钟表', '铃铛', '锅铲', '锣', '锦旗', '雨靴', '鞭炮',
     '风铃', '高压锅', '黑板', '龙舟']


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


def get_text(img, offset=0):
    return img[3:27, 120 + offset:180 + offset]


def _get_imgs(img):
    interval = 4
    length = 68
    for x in range(40, img.shape[0] - length, interval + length):
        for y in range(interval, img.shape[1] - length, interval + length):
            yield img[x:x + length, y:y + length]


def get_imgs(img):
    imgs = []
    for img in _get_imgs(img):
        imgs.append(img)
    return imgs


def get_image_from_12306():
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
                img = plt.imread(jpg)
                yield img
        except:
            print("Get data from 12306 error, wait....")
            time.sleep(2)


def load_image_from_disk():
    varify_code_path = os.path.join("12306验证图片")
    for varify_code_file in os.listdir(varify_code_path):
        if varify_code_file.endswith('.jpg'):
            code_file_image = plt.imread(os.path.join(varify_code_path, varify_code_file))
            yield code_file_image


def test(online=False, times=100):
    dataset_func = get_image_from_12306 if online else load_image_from_disk

    for test_number, code_file_image in enumerate(dataset_func()):
        imgs = None
        try:
            imgs = get_imgs(code_file_image)
            new_imgs = np.array(imgs) / 255
            gray_img = tf.image.rgb_to_grayscale(code_file_image)
            text_img = get_text(gray_img)
            new_text_img = np.array([text_img]) / 255
            # for i, img in enumerate(imgs):
            #     plt.subplot(4, 2, i+1)
            #     plt.imshow(img)
            # plt.show()
            # plt.imshow(code_file)
            # plt.show()
            # a = 1
        except:
            pass
        if imgs:
            # plt.imshow(text_img)
            # plt.show()
            text_prediction = text_model(new_text_img)
            prediction = model(new_imgs)
            text_labels = tf.argmax(text_prediction, 1)
            predict_text = verify_titles[int(text_labels)]
            print("prediction title: %s" % predict_text)
            labels = tf.argmax(prediction, 1)
            # plt.imshow(code_file_image)
            # plt.show()
            plt.subplot(3, 3, 1)
            plt.imshow(text_img)
            plt.title("题目为：%s" % predict_text)
            plt.axis("off")
            for i, label in enumerate(labels):
                text = verify_titles[int(label)]
                if text == predict_text:
                    text = '**%s**' % text
                plt.subplot(3, 3, i+2)
                plt.imshow(imgs[i])
                plt.title(text)
                plt.axis("off")
                print(text)
            plt.show()
            time.sleep(1)
            # text = verify_titles[int(label)]
            # plt.imshow(image)
            # plt.title("predict: %s" % text)
            # plt.show()
            # print(images.shape, labels.shape)
        if test_number == times:
            break


if __name__ == '__main__':
    # 再线下载图片预测
    test(True)
    # 离线图片预测
    # test()

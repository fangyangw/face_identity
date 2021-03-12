
import requests
import re
import base64
import sys
import numpy as np
import os

url = r'https://kyfw.12306.cn/otn/resources/login.html'
url = r'https://kyfw.12306.cn/passport/captcha/captcha-image64?login_site=E&module=login&rand=sjrand&1615339291663&callback=jQuery1910048662488071605337_1615339239718&_=1615339239720'


save_path = '12306验证图片'
for i in range(1):
    r = requests.request('GET', url)
    text = r.text
    search_obj = re.search(r'"image":"([^"]+)', text)
    if search_obj:
        img_base64 = search_obj.groups()[0]
        jpg = os.path.join(save_path, 'aa%s.jpg' % i)
        with open(jpg, 'wb') as fh:
            fh.write(base64.b64decode(img_base64))

    print(text)
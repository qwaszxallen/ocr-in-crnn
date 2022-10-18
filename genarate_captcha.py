# -*- coding: utf-8 -*-
import random
import PIL.Image as Image
import cv2
import matplotlib.pyplot as plt
import os
import string
import numpy as np
import math
import csv
from captcha.image import ImageCaptcha 

characters = string.digits + string.ascii_uppercase + '-'
width, height, n_len, n_class = 128, 64, 5, len(characters)
print(characters)
print(len(characters))

def strgen():
    generator = ImageCaptcha(width=width, height=height)
    while True:
        
        ans_str = ''.join([random.choice(characters) for j in range(n_len)])
        toImage = generator.generate_image(ans_str)
        toImage = np.array(toImage)
        yield toImage, ans_str
Genstr = strgen()

ans=[]

for i in range(20000):
    a,b=next(Genstr)
    cv2.imwrite('./captcha_train/%d.jpg'%i,a)
    ans.append(b)
    if i % 2000 == 0:
        print('genarate %d graph...'%i)
'''
for i in range(20000,22000):
    a=next(Genblank)
    cv2.imwrite('./genstr/%d.jpg'%i,a)
    ans.append('         ')


n = 0
for img in os.listdir('./22'):
    pic = cv2.imread('./22/'+img)
    for x in range(100):
        cv2.imwrite('./genstr/%d.jpg'%n,pic)
        n += 1
'''
with open('captcha_train.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image', 'label'])    
    for i in range(20000):        
        writer = csv.writer(csvfile)
        writer.writerow(["%d.jpg"%i ,ans[i]])

import cv2
import os
import numpy as np

all_num = 5790 / 2
fps = 25
size = (1280, 720)
name = 1
videowriter = cv2.VideoWriter("./video/result.mp4",-1, fps, size)
 
for i in range(int(all_num)):
    #生成图片视频
    print(f'./imgs/output/1/{i+1}.png')
    img=cv2.imread(f'./imgs/output/1/{i+1}.png')
    img=cv2.resize(img,(1280,720))
    videowriter.write(img)

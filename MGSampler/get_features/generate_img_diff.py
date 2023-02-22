import json
from skimage.measure import compare_ssim
import cv2

import imageio
import os

import multiprocessing

def read_lines(file):
    arr = []
    with open(file, 'r') as f:
        arr = f.readlines()
    return arr



from multiprocessing.dummy import Pool as ThreadPool
global root, dirs, files,tmp_num
img_diff = dict()
count = 0




def multiplication(video):
    video_name = video.split(' ')[0]
    #video_name = (video.split('/')[4]).split('.')[0]

    img = list()
    global count
    global img_diff
    img_diff[video_name] = list()
    
    # afm change
    root_path = "/workspace/k200/Mini_K200/train_val"
    video_path = root_path + '/' + video_name
    img_list = os.listdir(video_path)
    img_list.sort()
    for filename in img_list:
        filename = video_path + '/' + filename
        im = cv2.imread(filename)
        img.append(im)
    for i in range(len(img) - 1):
        tmp1 = cv2.cvtColor(img[i], cv2.COLOR_RGB2GRAY)
        tmp2 = cv2.cvtColor(img[i + 1], cv2.COLOR_RGB2GRAY)
        (score, diff) = compare_ssim(tmp1, tmp2, full=True)
        score = 1 - score

        img_diff[video_name].append(score)
    count = count + 1
    print(count)

video_list = read_lines('./lists/K200/val_K200_mp4.txt')[:]

pool = ThreadPool(processes=28)
re = pool.map(multiplication, video_list)
fileObject = open('img_diff_k200_val.json', 'a+')
jsonData = json.dumps(img_diff)
fileObject.write(jsonData)
fileObject.close()
pool.close()
pool.join()





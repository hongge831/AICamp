##���Դ��룬����json���
# -- coding: utf-8 --
import sys
sys.path.append('/home/ssw/caffe/python')
import matplotlib.pyplot as plt
import os
import caffe
import numpy as np
import pickle
import json
from tqdm import tqdm
root='/home/ssw/Desktop/YFF/faceTask/FER_image/'   #��Ŀ¼
deploy='/home/ssw/Desktop/YFF/faceTask/FER_image/deploy_alex.prototxt'    #deploy�ļ�����·��
caffe_model='/home/ssw/Desktop/YFF/faceTask/train_done/premodel/model_iter__iter_80000.caffemodel'   #ѵ���õ� caffemodel
net = caffe.Net(deploy,caffe_model,caffe.TEST)   #����model��network


#ͼƬԤ��������
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #�趨ͼƬ��shape��ʽ(1,3,28,28)
transformer.set_transpose('data', (2,0,1))    #�ı�ά�ȵ�˳����ԭʼͼƬ(28,28,3)��Ϊ(3,28,28)
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #��ȥ��ֵ��ǰ��ѵ��ģ��ʱû�м���ֵ������Ͳ���
transformer.set_raw_scale('data', 1)    # ���ŵ���0��255��֮��
transformer.set_channel_swap('data', (2,1,0))   #����ͨ������ͼƬ��RGB��ΪBGR
#����label�ֵ䣬���ϲ���Ҫ��������Ҫ
# char_dict = pick.load(open('./char_dict','rb'))
#�ֵ��ֵ�Ի���
src = '/home/ssw/Desktop/YFF/faceTask/test_done/dete2/'
fileName = os.listdir(src)
answer=[]
for imgName in tqdm(fileName):
    img = src+imgName
    im = caffe.io.load_image(img)  # ����ͼƬ
    net.blobs['data'].data[...] = transformer.preprocess('data', im)  # ִ���������õ�ͼƬԤ�������������ͼƬ���뵽blob��
    out = net.forward()
    top_k = net.blobs['prob'].data[0].flatten().argmax()
    save_dict = dict()
    save_dict['label'] = int(top_k)
    save_dict['filename'] = imgName
    r = json.dumps(save_dict)
    answer.append(r)
submit_name = './submit7.json'
with open(submit_name,'w')as f:
    wirte_str = '['+','.join(answer)+']'
    f.write(wirte_str)


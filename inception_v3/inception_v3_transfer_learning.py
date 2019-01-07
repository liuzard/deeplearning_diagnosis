#-*- coding: utf-8 -*-

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# 瓶颈层节点个数和迁移学习中的名字
BOTTLENECK_TENSOR_SIZE=2048
BOTTLENECK_TENSOR_NAME='pool_3/_reshape:0'

#迁移学习模型目录
MODEL_DIR='E:\\tensorflow_typical_dataset\inception_dec_2015'
MODEL_NAME='tensorflow_inception_graph.pb'

#迁移学习得到的特征向量保存地址
CATCH_DIR='/tmp/bottleneck'

#图像数据存放文件夹
INPUT_DATA='E:\\tensorflow_typical_dataset\\flower_photos'

#验证数据和测试数据的百分比
VALIDATION_PERCENTAGE=10
TEST_PERCENTAGE=10

#定义神经网络的设置
LEARNING_RATE=0.01
STEPS=4000
BATCH=100

#函数将文件夹中所有图片列表按训练、验证、测试数据分开
def create_image_lists(testing_percentage,validation_percentage):
    result={}
    sub_dirs=[x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir=True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir=False
            continue
        extensions=['jpg','jpeg','JPG','JPEG']
        file_list=[]
        dir_name=os.path.basename(sub_dir)
        for extension in extensions:
            file_glob=os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        label_name=dir_name.lower()

        training_iamges=[]
        testing_images=[]
        validation_images=[]

        for file_name in file_list:
            base_name=os.path.basename(file_name)
            chance=np.random.randint(100)
            if chance<validation_images:
                validation_images.append(base_name)
            elif chance<(testing_percentage+validation_percentage):
                testing_images.append(base_name)
            else:
                training_iamges.append(base_name)
        result[label_name]={'dir':dir_name,'training':training_iamges,'testing':testing_images,'validation':validation_images}
        return result


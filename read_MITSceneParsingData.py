__author__ = 'charlie'
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
#DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
#改为自己的训练数据路径
DATA_URL = "D:/FCN/tensorflow-FCNS/FCN.tensorflow/Data_zoo/MIT_SceneParsing/WaterSceneData"


def read_dataset(data_dir):
    #pickle_filename = "MITSceneParsing.pickle"
    #修改上一句，改成自己得数据名
    pickle_filename = "Water_SceneParsing.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        #如果已下载数据，注释下一行
        #utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
        #SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
        #修改上一句，取自己数据得文件名
        SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])
        result = create_image_lists(os.path.join(data_dir, SceneParsing_folder))   # Data_zoo / MIT_SceneParsing / ADEChallengeData2016
        # result =   {training: [{image: 图片全路径， annotation:标签全路径， filename:图片名字}] [][]
        #             validation:[{image:图片全路径， annotation:标签全路径， filename:图片名字}] [] []}

        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:   # 制作pickle文件
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:    # 打开pickle文件
        result = pickle.load(f)                 #读取
        training_records = result['training']
        validation_records = result['validation']
        del result

    return training_records, validation_records


def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        # Data_zoo/MIT_SceneParsing/ADEChallengeData2016/images/training/*.jpg
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
        # 加入文件列表  包含所有图片文件全路径+文件名字  如 Data_zoo/MIT_SceneParsing/ADEChallengeData2016/images/training/hi.jpg
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("\\")[-1])[0]
                annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

    return image_list

# -*- coding: utf-8 -*-
# @Time    : 18-5-6 下午4:14
# @Author  : Kim Luo
# @Email   : kim_luo_balabala@163.com
# @File    : Experiment.py
# @Software: PyCharm
'''
主要是想写一个实验的超类，提供一些小的接口，像txt读取啊，json存读啊这些

'''
import json
import os

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_txt_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list

def load_json(path=''):
        # get proposal data
        json_path = path
        file_obj = open(json_path, 'r')
        a = json.load(file_obj)
        file_obj.close()
        return a

def save_json(obj, json_path):
        with open(json_path, 'w') as file_obj:
            json.dump(obj, file_obj)
            file_obj.close()

if __name__ == '__main__':
    '''测试一下'''
    path='./data/ww/ww'
    check_dir(path)
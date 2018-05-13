# -*- coding: utf-8 -*-
'''
@author: Kim Luo
@license: I don't kown,don't ask me
@contact: kim_luo_balabala@163.com
@software: @.@
@file: test.py
@time: 18-4-17 下午5:19
@desc:
'''
import os

def print_cur_path():
    print( os.path.curdir)
    print(os.path.abspath(os.path.curdir))
    print __file__
    print os.path.dirname(__file__)
if __name__ == '__main__':
    pass
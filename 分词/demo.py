'''
@File       :   demo.py
@Author     :   Jiang Fubang
@Time       :   2020/7/22 11:12
@Version    :   1.0
@Contact    :   luckybang@163.com
@Dect       :   None
'''

import pkuseg

seg = pkuseg.pkuseg(postag=True, model_name="medicine")

text = seg.cut("我爱北京天安门")

print(text)
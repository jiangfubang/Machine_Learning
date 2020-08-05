'''
@File       :   fbnq.py
@Author     :   Jiang Fubang
@Time       :   2020/8/3 20:18
@Version    :   1.0
@Contact    :   luckybang@163.com
@Dect       :   None
'''
import chardet
with open('/Users/jiangfubang/Documents/Machine_Learning/github机器学习简单方式/basics-master/data/report.csv','rb') as f:
    fencoding=chardet.detect(f.read())
print(fencoding)
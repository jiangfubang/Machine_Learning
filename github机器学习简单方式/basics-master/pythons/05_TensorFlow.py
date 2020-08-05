'''
@File       :   05_TensorFlow.py
@Author     :   Jiang Fubang
@Time       :   2020/7/28 11:36
@Version    :   1.0
@Contact    :   luckybang@163.com
@Dect       :   None
'''
import numpy as np
import tensorflow as tf

SEED = 1234

np.random.seed(seed=SEED)
tf.random.set_seed(SEED)

# 常量
x = tf.constant(1)
# print(1)
x = tf.random.uniform((2, 3))
# print(f"Type: {x.dtype}")
# print(f"Size: {x.shape}")
# print(f"Values: \n{x}")
x = tf.zeros((2, 3))
# print(x)
x = tf.ones((2, 3))
# print(x)
x = tf.convert_to_tensor([[1, 2, 3], [4, 5, 6]], dtype='int32')
# print(f"Size: {x.shape}")
# print(f"Values: \n{x}")
x = tf.convert_to_tensor(np.random.rand(2, 3), dtype="float32")
print(f"Size: {x.shape}")
print(f"Values: \n{x}")
print (tf.config.list_physical_devices('GPU'))
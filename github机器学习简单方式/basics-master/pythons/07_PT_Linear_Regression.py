'''
@File       :   07_PT_Linear_Regression.py
@Author     :   Jiang Fubang
@Time       :   2020/7/28 12:04
@Version    :   1.0
@Contact    :   luckybang@163.com
@Dect       :   None
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 1234
NUM_SAMPLES = 50

np.random.seed(SEED)

def generate_data(num_samples):
    X = np.array(range(NUM_SAMPLES))
    random_noise = np.random.uniform(-10, 20, size=num_samples)
    y = 3.5 * X + random_noise
    return X, y

X, y = generate_data(NUM_SAMPLES)
data = np.vstack([X, y]).T
# print(data[:5])

df = pd.DataFrame(data, columns=['X', 'y'])
X = df[['X']].values
y = df[['y']].values
# print(df.head())
#
# plt.title("Generated data")
# plt.scatter(x=df['X'], y=df['y'])
# plt.show()

TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
SHUFFLE = True
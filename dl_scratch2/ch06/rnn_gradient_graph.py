import numpy as np
from nptyping import NDArray, Shape, Int, Float
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print

sns.set_style('whitegrid')


N = 2  # ミニバッチサイズ
H = 3  # 隠れ状態ベクトルの次元数
T = 20  # 時系列データの長さ

dh : NDArray[Shape['Batch, Hidden'], Float] = np.ones((N, H))

np.random.seed(3)

Wh : NDArray[Shape['Hidden, Hidden'], Float] = np.random.randn(H, H)
#Wh = np.random.randn(H, H) * 0.5

norm_list = []
for t in range(T):
    dh = dh @ Wh.T
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)

print(norm_list)

# グラフの描画
plt.plot(np.arange(len(norm_list)), norm_list)
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
plt.xlabel('time step')
plt.ylabel('norm')
plt.show()

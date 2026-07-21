import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# 设置参数
mu = np.array([0, 0])  # 均值
cov = np.array([[1, 0.5],
                [0.5, 1]])  # 协方差矩阵

# 创建网格
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# 创建多元正态分布
rv = multivariate_normal(mu, cov)
Z = rv.pdf(pos)

# 创建图形
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9,
                      linewidth=0, antialiased=True, rstride=2, cstride=2)

# 设置斜视角
ax.view_init(elev=30, azim=45)  # 仰角30度，方位角45度

# 添加标签和标题
ax.set_xlabel('X轴', fontsize=12, labelpad=10)
ax.set_ylabel('Y轴', fontsize=12, labelpad=10)
ax.set_zlabel('概率密度', fontsize=12, labelpad=10)
ax.set_title('二维正态分布三维图 (斜视角)', fontsize=16, pad=20)

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1)

# 设置坐标轴范围
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([0, 0.25])

# 添加网格
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 额外绘制等高线投影图来辅助理解
fig2, ax2 = plt.subplots(1, 2, figsize=(15, 6))

# 左侧：三维图的俯视图
ax2[0].contourf(X, Y, Z, levels=20, cmap='viridis')
ax2[0].set_xlabel('X轴')
ax2[0].set_ylabel('Y轴')
ax2[0].set_title('二维正态分布等高线图')
ax2[0].set_aspect('equal')
ax2[0].grid(True, alpha=0.3)

# 右侧：边缘分布
ax2[1].plot(x, rv.pdf(np.column_stack((x, np.zeros_like(x)))),
           'r-', linewidth=2, label='X边缘分布')
ax2[1].plot(y, rv.pdf(np.column_stack((np.zeros_like(y), y))),
           'b-', linewidth=2, label='Y边缘分布')
ax2[1].set_xlabel('值')
ax2[1].set_ylabel('概率密度')
ax2[1].set_title('边缘分布')
ax2[1].legend()
ax2[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

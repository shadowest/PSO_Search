# Rosenbrock函数极大值
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_Rosenbrock(p):          # 得到Rosenbrock函数值
    return 100 * (p[0] ** 2 - p[1]) ** 2 + (1 - p[0]) ** 2


if __name__ == '__main__':
    gbest_all_x = []            # 存储独立运行的PSO找到的pbest的x1的值
    gbest_all_y = []            # 存储独立运行的PSO找到的pbest的x2的值
    gbest_all_fn = []           # 存储独立运行的PSO找到的pbest的函数值
    total = 10                  # 独立运行的次数

    m = 30                      # 种群大小
    x_max = 2.048               # 每一维位移上限
    v_max = 0.3072              # 每一维速度上限
    run_max = 20                # PSO算法迭代寻找次数
    w_max = 1                   # 惯性权重上限
    w_min = 0.5                 # 惯性权重下限
    c1 = c2 = 2                 # 加速度常数

    for num in range(total):    # 开始total次独立寻找
        x = np.random.uniform(low=-x_max, high=x_max, size=(2, m))  # 初始化速度和位置
        v = np.random.uniform(low=-v_max, high=v_max, size=(2, m))
        pbest = x.copy()                # 初始化pbest
        pbest_fn = []
        for i in range(m):
            pbest_fn.append(get_Rosenbrock((x[0][i], x[1][i])))
        gbest_fn = max(pbest_fn)        # 初始化gbest
        gbest_fn_pos = pbest_fn.index(gbest_fn)
        gbest = (x[0][gbest_fn_pos], x[1][gbest_fn_pos])

        for i in range(run_max):        # 开始PSO迭代寻找
            w = w_max - (w_max - w_min) * (i / run_max)     # 更新该次迭代的惯性权重
            r1 = random.random()
            r2 = random.random()
            for j in range(m):
                for k in range(2):      # 更新每一维的速度
                    v[k][j] = w * v[k][j] + c1 * r1 * (pbest[k][j] - x[k][j]) + c2 * r2 * (gbest[k] - x[k][j])
                    if v[k][j] > v_max:
                        v[k][j] = v_max
                    elif v[k][j] < -v_max:
                        v[k][j] = -v_max
                    x[k][j] = x[k][j] + v[k][j]     # 更新每一维的位置
                    if x[k][j] > x_max:
                        x[k][j] = x_max
                    elif x[k][j] < -x_max:
                        x[k][j] = -x_max
                fn = get_Rosenbrock((x[0][j], x[1][j]))     # 评估该位置的适应度函数值
                if fn >= pbest_fn[j]:               # 更新pbest
                    pbest[0][j] = x[0][j]
                    pbest[1][j] = x[1][j]
                    pbest_fn[j] = fn
            gbest_fn = max(pbest_fn)                # 更新gbest
            gbest_fn_pos = pbest_fn.index(gbest_fn)
            gbest = (x[0][gbest_fn_pos], x[1][gbest_fn_pos])
        gbest_all_x.append(gbest[0])                # 存储每次独立运行找到的最优解
        gbest_all_y.append(gbest[1])
        gbest_all_fn.append(get_Rosenbrock(gbest))

    for i in range(len(gbest_all_x)):               # 输出最优解
        print("第 %d 组搜索结果坐标为：（%f，%f)，其函数值为：%f" % (i + 1, gbest_all_x[i], gbest_all_y[i], gbest_all_fn[i]))

    fig = plt.figure()                          # 定义新的三维坐标轴
    graph = plt.axes(projection='3d')
    xx = np.arange(-x_max, x_max, 0.01)         # 绘制Rosenbrock函数
    yy = np.arange(-x_max, x_max, 0.01)
    X, Y = np.meshgrid(xx, yy)
    Z = 100 * (X ** 2 - Y) ** 2 + (1 - X) ** 2
    graph.plot_surface(X, Y, Z, cmap='rainbow')
    graph.scatter3D(gbest_all_x, gbest_all_y, gbest_all_fn, marker='o', c='r')     # 标出所有找到的最优解
    plt.show()

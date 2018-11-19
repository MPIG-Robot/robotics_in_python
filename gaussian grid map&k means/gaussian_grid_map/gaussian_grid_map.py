"""

2D gaussian grid map sample

author: Atsushi Sakai (@Atsushi_twi)

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

EXTEND_AREA = 10    # [m] grid map extention length

show_animation = True

def generate_gaussian_grid_map(ox, oy, xyreso, std):#输入参数
    minx, miny, maxx, maxy, xw, yw = calc_grid_map_config(ox, oy, xyreso)
    gmap = [[0.0 for i in range(yw)] for i in range(xw)] #calc each potential?
    for ix in range(xw): #赋值
        for iy in range(yw):
            x = ix * xyreso + minx
            y = iy * xyreso + miny
            # Search minimum distance
            mindis = float("inf") #00
            for (iox, ioy) in zip(ox, oy):
                d = math.sqrt((iox - x)**2 + (ioy - y)**2)  #
                if mindis >= d:
                    mindis = d
            pdf = (1.0 - norm.cdf(mindis, 0.0, std)) #  cdf分布函数(x,u,sigma)
            gmap[ix][iy] = pdf
    return gmap, minx, maxx, miny, maxy

def calc_grid_map_config(ox, oy, xyreso):
    minx = round(min(ox) - EXTEND_AREA / 2.0)
    miny = round(min(oy) - EXTEND_AREA / 2.0)
    maxx = round(max(ox) + EXTEND_AREA / 2.0)
    maxy = round(max(oy) + EXTEND_AREA / 2.0)
    xw = int(round((maxx - minx) / xyreso))
    yw = int(round((maxy - miny) / xyreso))
    return minx, miny, maxx, maxy, xw, yw

def draw_heatmap(data, minx, maxx, miny, maxy, xyreso): #绘制热图
    x, y = np.mgrid[slice(minx - xyreso / 2.0, maxx + xyreso / 2.0, xyreso),
                    slice(miny - xyreso / 2.0, maxy + xyreso / 2.0, xyreso)]
    plt.pcolor(x, y, data, vmax=1.0, cmap=plt.cm.Blues) #colormap为蓝色 颜色取值最大值为1
    plt.axis("equal") #使x和y的相等增量


def main():
    print(__file__ + " start!!")

    xyreso = 0.5 # 网格分辨率
    STD = 5.0

    for i in range(5):
        ox = (np.random.rand(4) - 0.5) * 10.0 #np.random.rand()生成0~1随机值
        oy = (np.random.rand(4) - 0.5) * 10.0
        gmap, minx, maxx, miny, maxy = generate_gaussian_grid_map(
            ox, oy, xyreso, STD)

        if show_animation:
            plt.cla() #清空
            draw_heatmap(gmap, minx, maxx, miny, maxy, xyreso)
            plt.plot(ox, oy, "xr")
            plt.plot(0.0, 0.0, "ob")
            plt.pause(1.0) #暂停一秒



if __name__ == '__main__':
    main()

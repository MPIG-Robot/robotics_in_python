import math
import numpy as np
import matplotlib.pyplot as plt

EXTEND_AREA = 10.0  #原为10.0

show_animation = True


def calc_grid_map_config(ox, oy, xyreso):
    minx = round(min(ox) - EXTEND_AREA / 2.0)
    miny = round(min(oy) - EXTEND_AREA / 2.0)
    maxx = round(max(ox) + EXTEND_AREA / 2.0)
    maxy = round(max(oy) + EXTEND_AREA / 2.0)
    xw = int(round((maxx - minx) / xyreso))
    yw = int(round((maxy - miny) / xyreso))

    return minx, miny, maxx, maxy, xw, yw


class precastDB:

    def __init__(self):
        self.px = 0.0
        self.py = 0.0
        self.d = 0.0
        self.angle = 0.0
        self.ix = 0
        self.iy = 0

    def __str__(self):
        return str(self.px) + "," + str(self.py) + "," + str(self.d) + "," + str(self.angle)


def atan_zero_to_twopi(y, x): #点（x,y）与x轴的弧度夹角，在第三、四象限转为第一、二象限。（单位/弧度）
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += math.pi * 2.0

    return angle


def precasting(minx, miny, xw, yw, xyreso, yawreso):

    precast = [[] for i in range(int(round((math.pi * 2.0) / yawreso)) + 1)]#将绕原点一圈分成36块，()因为在main()函数定义的10°,
                                                                            #每块有自己的ID()，则每一块像素都有自己所在区域的编号。
                                                                            #每块像素的所有信息(实际坐标，距原点的距离、映射到网格图中的坐标、所在区域块)都存储在
                                                                            #precast[[] [] [] ...]中的list中


    for ix in range(xw):
        for iy in range(yw):
            px = ix * xyreso + minx
            py = iy * xyreso + miny

            d = math.sqrt(px**2 + py**2)
            angle = atan_zero_to_twopi(py, px)
            angleid = int(math.floor(angle / yawreso))#angle的编号

            pc = precastDB()

            pc.px = px
            pc.py = py
            pc.d = d
            pc.ix = ix
            pc.iy = iy
            pc.angle = angle

            precast[angleid].append(pc)

    return precast


def generate_ray_casting_grid_map(ox, oy, xyreso, yawreso):

    minx, miny, maxx, maxy, xw, yw = calc_grid_map_config(ox, oy, xyreso)

    pmap = [[0.0 for i in range(yw)] for i in range(xw)]#生成xw个((元素有yw个)的list) 这xw个list组成新的list: pmap

    precast = precasting(minx, miny, xw, yw, xyreso, yawreso)

    for (x, y) in zip(ox, oy):

        d = math.sqrt(x**2 + y**2) #点(x,y)到原点的距离
        angle = atan_zero_to_twopi(y, x)
        angleid = int(math.floor(angle / yawreso))#那四个点的所在区域

        gridlist = precast[angleid]  #新的gridlist存储那四个点所有信息，也包括除了这个点以外的这个区域的点的信息。因为angleid本来就是区域的号，有坐标、所在区域，跟(0,0)的距离、

        ix = int(round((x - minx) / xyreso)) #ix: 各个点在网格图上的位置，27是离左边边界的距离，52是离下面边界的距离。
        iy = int(round((y - miny) / xyreso))
        pmap[ix][iy] = 3.0      #将这四个点的网格点映射坐标：pmap[x1][y1]、pmap[x2][y2]、pmap[x3][y3]、pmap[x4][y4] 应该是变成黑色。80%把握 #原为1.0
        for grid in gridlist:   #如果网格中的某些点和那四个点是一个区域，并且距离大于这些点距离原点的距离，
                                #则将这些点的pmap[x][y]值变成0.5，应该是都变成一个色，即下面的都是0.5。
            if grid.d > d:
                pmap[grid.ix][grid.iy] = 0.5#控制“阴影”的颜色深浅 原为0.5比较浅， 2.0比较深



    return pmap, minx, maxx, miny, maxy, xyreso


def draw_heatmap(data, minx, maxx, miny, maxy, xyreso):
    x, y = np.mgrid[slice(minx - (xyreso / 2.0), maxx + (xyreso / 2.0), xyreso), #生成一个矩阵，从上到下记录绘画的点。
                    slice(miny - (xyreso / 2.0), maxy + (xyreso / 2.0), xyreso)]
    plt.pcolor(x, y, data, vmax=1.0, cmap=plt.cm.Blues)#原为 vamx=1.0  控制投射的‘阴影’颜色深浅，越小越深  plt.cm.Blues  #整体颜色
    plt.axis("equal")#保持x y轴的刻度的一致


def main():
    print(__file__ + " start!!")

    xyreso = 0.25  # x-y grid resolution [m]    原为0.25 #控制网格的密度。
    yawreso = np.deg2rad(10.0)  #度转为弧度 yaw angle resolution [rad] 原为10.0  #每10°分区。

    for i in range(5): #原为5  生成5张图
        ox = (np.random.rand(4) - 0.5) * 10.0 #生成[0,1]之间4个数字 那么ox其实就是一个列表
        oy = (np.random.rand(4) - 0.5) * 10.0
        pmap, minx, maxx, miny, maxy, xyreso = generate_ray_casting_grid_map(
            ox, oy, xyreso, yawreso)
        if show_animation:
            plt.cla()#即清除当前图形中的当前活动轴。其他轴不受影响。
            draw_heatmap(pmap, minx, maxx, miny, maxy, xyreso)
            plt.plot(ox, oy, "xr")  #以'x'且红色标记
            plt.plot(0.0, 0.0, "ob")#以'原点'且蓝色标记
            plt.pause(1.0) #图片停顿1s


if __name__ == '__main__':
    main()
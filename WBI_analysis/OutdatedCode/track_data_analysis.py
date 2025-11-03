import cv2 as cv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def main():
    fps = 20
    file_name = 'D:\\ZC_data\\track_data\\0217\\0217-02\\'
    videoFile = file_name + 'c1.mp4'
    data_stage = pd.read_csv(file_name + 'stage_data.txt', header = None).values
    t_stage =  data_stage[:, 0]
    f_data = np.loadtxt(file_name + 'c1.txt', delimiter=',')
    t_c = f_data[f_data[:,3] == 1, 1]

    x = data_stage[:, 3] / 1000
    y = data_stage[:, 4] / 1000
    plt.figure(figsize=(15, 15))
    plt.scatter(x, y, 1,c = range(len(x)))
    plt.savefig(file_name + '\\xy.jpg')
    plt.close()

    figure2, ax2 = plt.subplots(figsize = (15,15))
    ax2.set_aspect('equal', adjustable='box')
    ax2.scatter(x, y, 1)

    # 计算速度
    dx = np.diff(x)
    dy = np.diff(y)
    T = data_stage[:, 0] / 1000
    dt = np.diff(T)
    v = np.sqrt(dx ** 2 + dy ** 2) / dt
    v_m = np.convolve(v, np.ones(60)/60, mode='same')

    figure1, ax1 = plt.subplots(figsize = (15,10))
    ax1.scatter(T[1:], v_m, 1)

    # 计算激光照射时间
    s_data = pd.read_csv(file_name + 's-data.txt', header = None)
    s_start = s_data[s_data[1] == 2].index
    s_stop = s_data[s_data[1] == 3].index
    if s_stop.empty:
        s_s = [len(s_data[1])]
        print(s_s)
    else:
        s_s = []
        for m in s_start:
            data = (s_stop >= m)
            s_s.append(s_stop[data == 1][0])
    for i in range(len(s_start)):
        t_array = []
        t_l = int(round((s_data[0][s_start[i] + 1 : s_s[i] - 1]).diff().sum(skipna=True) / (s_s[i] - s_start[i] - 2)))
        t = s_data[0].values
        t_frame = np.concatenate([[t[s_start[i]] - t_l *2, t[s_start[i]] - t_l], t[s_start[i]:s_s[i]]])
        t_start = s_data[0][s_start[i]] - t_l *2
        t_stop = s_data[0][s_s[i] - 1] + t_l

        data1 = t_stage > t_start
        id_start = np.where(data1 == 1)[0][0] - 1
        data1 = t_stage > t_stop
        id_stop = np.where(data1 == 1)[0][0]
        np.savetxt(file_name + 's' + str(i + 1) + '.txt', data_stage[id_start:id_stop, [0,3,4]] / 1000, fmt='%.4f')
        np.savetxt(file_name + 't' + str(i + 1) + '.txt', t_frame / 1000, fmt='%.4f')
        ax1.scatter(T[id_start:id_stop], v_m[id_start:id_stop], 1)
        ax2.scatter(x[id_start:id_stop], y[id_start:id_stop], 1)

        data1 = t_c > t_start
        id_start = np.where(data1 == 1)[0][0] - 1
        f_start = id_start.astype(str)
        data1 = t_c > t_stop
        id_stop = np.where(data1 == 1)[0][0]
        f_stop = id_stop.astype(str)
        with open(file_name + 'f' + str(i + 1) + '.txt', 'a') as file:
            file.write('Start:\n')
            file.write(f_start)
            file.write('\n')
            file.write('Stop:\n')
            file.write(f_stop)

    # 显示图形
    figure1.savefig(file_name + 'v.jpg')
    ax2.scatter(x[0], y[0], 50, c='green')
    ax2.scatter(x[-1], y[-1], 50, c='red')
    figure2.savefig(file_name + 'xy_laser.jpg')

if __name__ == '__main__':
    main()

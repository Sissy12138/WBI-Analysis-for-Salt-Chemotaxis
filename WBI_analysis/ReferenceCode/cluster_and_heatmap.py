# import torch
# import time, os
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from datetime import datetime
# import glob
# import pandas as pd
# import math
# from holoviews.plotting.bokeh.styles import font_size
from matplotlib.gridspec import GridSpecFromSubplotSpec
import AnalysisMethod as am
# import cv2
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
# import glob
import numpy as np
import matplotlib.pyplot as plt
# import sys
# # sys.path.append('/mnt/c/gaochao/CODE/NeuronTrack/fDNC_Neuron_ID/src')
# # from src_utils import *
# from tqdm import tqdm
import matplotlib as mpl
# from multiprocessing import Pool
# import os
# from collections import Counter
# from scipy.io import loadmat
import cv2
# import imageio
# from io import BytesIO
# from PIL import Image
# from multiprocessing import Pool
# from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings('ignore')
from tifffile import imread
mpl.rcParams['figure.figsize'] = (6,6)
mpl.rcParams['figure.dpi'] = 96
mpl.rcParams['font.size'] = 14
mpl.rcParams['savefig.bbox'] = 'tight'



def PlotPmdCluster(w_p2m,idx_pmd,idx_m1,bound_pmd,bound_m1,link,aff,vmin,vmax,threshold,xlabel,ylabel,cmap,level):
    fig = plt.figure(figsize=(7, 6))
    ax = plt.subplot(1,1,1)
    ax.xaxis.set_ticks_position('none')  # 不显示x轴的刻度
    ax.yaxis.set_ticks_position('none')  # 不显示y轴的刻度
    ax.set_xticks([])  # 移除x轴的刻度标记
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # plt.title(title)
    gs00 = GridSpecFromSubplotSpec(1, 2, subplot_spec=ax.get_subplotspec(), wspace=0.02, hspace=0.02, width_ratios=[3., 0.5])
    ax0 = fig.add_subplot(gs00[0, 0])
    ax1 = fig.add_subplot(gs00[0, 1])
    # ax2 = fig.add_subplot(gs00[0, 0])
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None,linkage=link, affinity=aff)
    model = model.fit(w_p2m)
    am.plot_dendrogram(model, truncate_mode="level", p=level, \
                    no_labels=True, orientation='right', ax=ax1,color_threshold=threshold)
    # ax1.set_xlim([0.5,2])
    ax1.xaxis.set_ticks_position('none')  # 不显示x轴的刻度
    ax1.yaxis.set_ticks_position('none')  # 不显示y轴的刻度
    ax1.set_xticks([])  # 移除x轴的刻度标记
    ax1.set_yticks([])
    ax1.invert_yaxis()
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax_list=list(np.linspace(0,w_p2m.shape[0],24,dtype=int))
    X_sort = w_p2m[idx_pmd][:,idx_m1]
    ax0=sns.heatmap(X_sort, ax=ax0, cmap=cmap, vmin=vmin, vmax=vmax,cbar=False)
    cbar_ax = fig.add_axes([-0.02, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar=fig.colorbar(ax0.collections[0], cax=cbar_ax)
    cbar.ax.yaxis.set_ticks_position('left')  # 将刻度移动到左边
    cbar.ax.yaxis.set_label_position('left')
    ax0.set_xticks(ax_list, ax_list,fontsize=10)
    ax0.set_yticks(ax_list, ax_list,fontsize=10)
    # ax0.invert_yaxis()
    ax0.set_xlabel(xlabel,fontsize=16)
    ax0.set_ylabel(ylabel,fontsize=16)
    boundary = 0
    idx_bound=np.zeros(len(bound_pmd)+1)
    # print(boundaries_y)
    for i in range(len(bound_pmd)-1):
        boundary += bound_pmd[i]
        idx_bound[i+1]=boundary
        # print(boundary)
        ax0.axhline(y=boundary, color='white', linestyle='--')
    idx_bound[-1]=400
    boundary = 0
    for i in range(len(bound_m1)-1):
        boundary += bound_m1[i]  
        # print(boundary)
        ax0.axvline(x=boundary, color='white', linestyle='--')
    return idx_bound.astype(int),fig


def calcium_heatmap(calcium_intensity, neuron_ids, show_id_stride=20, show_vol_stride=20, heatmap_range=(None,0.5),fig_size=None, font_size=90, font_color='black',smooth_kernel=10):
    num_neurons, num_vols = calcium_intensity.shape
    # num_vols = num_vols//3
    DRR0 = calcium_intensity

    if smooth_kernel:
        for i,k in enumerate(DRR0):
            DRR0[i] = cv2.blur(k,(1,smooth_kernel))[:,0]
            # DRR0 = savgol_filter(DRR0, window_length=5, polyorder=2)
    # 绘制热力图
    plt.figure(figsize=fig_size)
    heatmap=sns.heatmap(DRR0, cmap='jet', vmin=heatmap_range[0], vmax=heatmap_range[1], xticklabels=np.arange(num_vols)[::show_vol_stride], yticklabels=neuron_ids[::show_id_stride],cbar=True,)
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=font_size)  # 设置 colorbar 字体大小

    #-------------------------
    # 假设要添加的文本字符串为 'Vertical Text'
    text_str = 'ΔR/R0'

    # 获取颜色条的轴对象
    colorbar_ax = colorbar.ax

    # 设置颜色条轴的标签大小
    font_size = 40
    colorbar_ax.tick_params(labelsize=font_size)

    # 添加垂直文本
    x_pos = 3.1  # 控制文本的水平位置，可以根据需要调整
    y_pos = 0.5  # 控制文本的垂直位置
    rotation = 90  # 文本旋转角度

    colorbar_ax.text(x_pos, y_pos, text_str, rotation=rotation,
                    transform=colorbar_ax.transAxes,
                    verticalalignment='center', horizontalalignment='center', fontsize=font_size)
    #-------------------------
    # x_sticks = np.array([ 244,  668, 1012, 1085, 1139, 1194, 1296, 1364, 1404, 1438]) # 06
    # for x in x_sticks:
    #     plt.axvline(x=x, color='white', linestyle='--', linewidth=4)  # Adjust color and linestyle as needed

    plt.yticks(ticks=np.arange(0, num_neurons, show_id_stride), labels=neuron_ids[::show_id_stride],fontsize=font_size, color=font_color)
    plt.xticks(ticks=np.arange(0, num_vols, show_vol_stride), labels=np.arange(num_vols)[::show_vol_stride],fontsize=font_size, rotation=45, color=font_color)
    # plt.xticks(ticks=np.arange(0, num_vols, show_vol_stride), labels=np.arange(0,1300,100),fontsize=font_size, rotation=45, color=font_color)
    # plt.title('Calcium activity traces (ΔR/R0) Heatmap',fontsize=font_size,)
    plt.xlabel('Volumes',fontsize=font_size, color=font_color)
    plt.ylabel('Neurons',fontsize=font_size,color=font_color)
    # plt.axis('off')
    # plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # 设置y轴的刻度位置，使其从顶部开始，因为神经元通常从顶部向下绘制
    plt.gca().invert_yaxis()
    # plt.show()
    if smooth_kernel:
        plt.savefig(f'{signal_save_path}/cluster_calcium_heatmap(smooth_{smooth_kernel}).png', transparent=False,dpi=100)
    else:
        plt.savefig(f'{signal_save_path}/cluster_calcium_heatmap.png', transparent=False,dpi=100)

def draw_calcium_curve(calcium_intensity, smooth_kernel=None, save=True,fig_size=None, scale=1):

    plt.figure(figsize=fig_size)
    num_neurons, num_timepoints = calcium_intensity.shape
    colors = ['orangered', 'blue', 'limegreen', 'purple', 'gold', 'cyan', 'magenta', 'coral', 'skyblue', 'orange']
    import cv2
    for i in range(calcium_intensity.shape[0]):
        color = colors[i % len(colors)]  # 循环使用颜色列表
        if smooth_kernel:
            smooth_line_data = cv2.blur(calcium_intensity[i], (1, 3))*scale
        else:
            line_data = calcium_intensity[i]*scale
        smooth_line_data = cv2.blur(calcium_intensity[i], (1, 7))*scale
        plt.plot(smooth_line_data + i, color=color, linestyle='-', label='A' if i == 0 else "",linewidth=1)  # 只显示第一个标签
        plt.scatter(y=line_data+i,x=np.arange(line_data.shape[0]),color=color, linestyle='-', label='A' if i == 0 else "",s=1)  # 只显示第一个标签
        if i in bound:
            plt.axhline(y=i-0.2, color='k', linestyle='dashdot', linewidth=4, c='r')  # 在y轴索引位置画一条虚线
        # 在每一行的右侧添加纵轴范围标注
        # plt.text(num_timepoints - 5, i + (y_max - y_min) / 2, f'{y_min:.2f} to {y_max:.2f}', 
        #      color=color, fontsize=10, va='center')
    plt.xlabel("Time point",fontsize=20)  # X轴标签
    plt.ylabel("Intensity",fontsize=20)  # Y轴标签
    # 设置 y 轴范围
    plt.ylim(bottom=0-1, top=num_neurons+1)  # 设置 y 轴的最小值为 0，最大值为 num_neurons

    # 每隔 5 个单位标注一个 y ticks
    y_ticks = np.arange(0, num_neurons + 1, 1)  # 生成从 0 到 num_neurons 的刻度，每隔 5 个单位
    plt.yticks(y_ticks,fontsize=20)  # 设置 y 轴刻度
    x_ticks = np.arange(0, num_timepoints + 1, 200)  # 生成从 0 到 num_neurons 的刻度，每隔 5 个单位
    plt.xticks(x_ticks,fontsize=20,rotation=45)  # 设置 y 轴刻度
    # plt.show()
    plt.tight_layout()
    if save:
        if smooth_kernel:
            plt.savefig(f'{signal_save_path}/cluster_calcium_curve(smooth_{smooth_kernel}).png')
        else:
            plt.savefig(f'{signal_save_path}/cluster_calcium_curve.png')


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "jilab pipeline")
    # --------------------------------- Stack loading and Stage 1: Pre-processing ---------------------------------
    parser.add_argument('--zrange', type = tuple, default = (0, 31))
    parser.add_argument('--prod_thresh', type = int, default = 2000)
    parser.add_argument('--ref_id', type = int, default = 0)
    parser.add_argument('--workspace', type = str, default = '/mnt/c/gaochao/CODE/NeuronTrack/CeNDeR/workspace4', help = '')
    parser.add_argument('--data_path', type = str, default = 'data/Jilabdata/1129/1223-02.nd2', help = '')
    args = parser.parse_args()

    global signal_save_path, idx, bound
    data_name = args.data_path.split('/')[-1].split('.')[0]
    preprocess_root = f'{args.workspace}/{data_name}/stardist'
    signal_save_path = f'{preprocess_root}/extracted_signal(ref{args.ref_id})'

    choose_index = 0
    thresh = 5
    links=['ward','average','average','complete']
    affs=['euclidean','cosine','cityblock','cosine']
    vmin=-0.5
    vmax=1

    lyt = rf'{signal_save_path}/calcium_intensity.npy'
    calcium_intensity= np.load(lyt)
    print(calcium_intensity.shape)

    idx=am.cluster(np.corrcoef(calcium_intensity),link=links[choose_index],aff=affs[choose_index])
    bound=np.cumsum(am.GetBound(np.corrcoef(calcium_intensity),link=links[choose_index],aff=affs[choose_index],threshold=thresh).astype(int))
    # np.save(f'{savname}/idx.npy',idx)
    # np.save(f'{savname}/bound.npy',bound)

    Bound_m1, fig = PlotPmdCluster(np.corrcoef(calcium_intensity), idx, idx, bound, bound, links[choose_index], affs[choose_index], vmin, vmax, thresh, 'C.elegans neuron', 'C.elegans  neuron', 'jet', 35)
    plt.savefig(f'{signal_save_path}/cluster_of_corrcoef_of_neurons.png')

    bound = np.cumsum(bound)
    calcium_intensity = calcium_intensity[idx]

    fig_h = int(10*round(calcium_intensity.shape[0]/10))
    fig_w = int(0.7*round(calcium_intensity.shape[1]/100))
    draw_calcium_curve(calcium_intensity,smooth_kernel=None,fig_size=(fig_w,fig_h),scale=1.5)

    font_size = 30
    fig_w = 40
    fig_h = 20

    neuron_ids = np.arange(calcium_intensity.shape[0])
    calcium_heatmap(calcium_intensity, neuron_ids=neuron_ids, show_id_stride=10, show_vol_stride=500, heatmap_range=(0,0.6),font_size=font_size, smooth_kernel=10,fig_size=(fig_w,fig_h))
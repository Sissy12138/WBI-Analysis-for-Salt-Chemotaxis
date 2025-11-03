# %% [markdown]
# # Import Package

# %%
import sys
sys.path.append(r"D:\data analysis\code\WBI_analysis")  # 例如 r"C:\Users\YourName\Project"
# %%
import os
import pandas as pd
import bisect
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D 
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import grey_opening,grey_closing
from matplotlib.gridspec import GridSpecFromSubplotSpec
import AnalysisMethod as am
from sklearn.cluster import AgglomerativeClustering
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib as mpl
import warnings
import argparse
from scipy.signal import correlate
from scipy.stats import ttest_1samp
# from statsmodels.stats.multitest import multipletests
from scipy.ndimage import gaussian_filter1d,binary_closing, binary_opening
# from matplotlib.gridspec import GridSpec
warnings.filterwarnings('ignore')
from tifffile import imread
from scipy.stats import sem
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import ttest_rel,wilcoxon, shapiro
from scipy.stats import ttest_ind
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib_venn import venn2, venn3
from upsetplot import UpSet, from_contents
mpl.rcParams['figure.figsize'] = (6,6)
mpl.rcParams['figure.dpi'] = 96
mpl.rcParams['font.size'] = 14
mpl.rcParams['savefig.bbox'] = 'tight'

# %% [markdown]
# # PreAnalysis 预处理
# + 行为数据和神经数据分别导入并平滑（平滑使用相同参数）
# + 行为数据
#     + 开闭操作
#     + 计算并验证前进后退的起始和结束
# + 神经数据
#     + 标准化：Z-score
# + 合并行为与神经
#     + 将行为数据中标记的前进后退等

# %% [markdown]
# ## 函数定义

# %% [markdown]
# ### 函数

# %%
def signed_norm(row):
    '''
    求头部速度
    '''
    val = row["head_moving"]
    forward = row["forward"]

    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan

    if isinstance(val, str):
        arr = np.fromstring(val.strip("[]"), sep=" ")
    else:
        arr = np.array(val, dtype=float)

    if arr.size == 0:
        return np.nan

    norm = np.linalg.norm(arr)
    # forward==0 正，forward==1 负
    return norm if forward == 0 else -norm


# %%
def visualize_traces(p_f, calcium_intensity,df_vol_time, smooth=True, sigma=0.3,delta_y=1.2):
    # 平滑和可视化
    n_neurons, _ = calcium_intensity.shape
    print(f'shape of calcium_traces {calcium_intensity.shape}')
    t_max = df_vol_time['Vol_Time'].max()
    fig, ax = plt.subplots(figsize=(t_max/100, n_neurons/4))

    if 'mask' in df_vol_time.columns:
        mask = df_vol_time['mask'].values
        time = df_vol_time['Vol_Time'].values
        in_mask = False
        start = None
        for i in range(len(mask)):
            if mask[i] == 1 and not in_mask:
                in_mask = True
                start = time[i]
            elif (mask[i] == 0 or i == len(mask)-1) and in_mask:
                in_mask = False
                end = time[i] if mask[i] == 0 else time[-1]
                ax.axvspan(start, end, color='pink', alpha=0.2)

    for i in range(n_neurons):
        trace = calcium_intensity[i, :]
        if smooth:
            trace = gaussian_filter1d(trace, sigma=sigma)
        plt.plot(df_vol_time['Vol_Time'], trace + i * delta_y,lw=0.5, label=f"Neuron {i}")

    # 设置坐标轴
    plt.xlabel("Time (frames)")
    plt.ylabel("Neuron index")
    desired_ticks = np.arange(0, df_vol_time['Vol_Time'].max(), 30) # 生成一个数组 [10, 30, 50, 70, 90]
    plt.xticks(desired_ticks, rotation=30)
    y_positions = np.arange(0, n_neurons * delta_y, delta_y)
    plt.yticks(y_positions[:n_neurons], np.arange(n_neurons))
    plt.title(f"Calcium Traces (Gaussian smoothed){sigma}")
    plt.grid(True, linestyle='--', color='grey', linewidth=0.35, alpha=0.5)
    plt.savefig(p_f+f'\\Cal_traces_Gaussion{sigma}.png', bbox_inches='tight')
    plt.close()
    # plt.show()

# %% [markdown]
def draw_calcium_curve(calcium_intensity, smooth_kernel=None, save=True,fig_size=None,signal_save_path=None, scale=1):

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
    plt.tight_layout()
    if save:
        if smooth_kernel:
            plt.savefig(f'{signal_save_path}/cluster_calcium_curve(smooth_{smooth_kernel}).png')
            plt.close()
        else:
            plt.savefig(f'{signal_save_path}/cluster_calcium_curve.png')
            plt.close()

# %%
def PlotPmdCluster(w_p2m,idx_pmd,idx_m1,bound_pmd,bound_m1,link,aff,vmin,vmax,threshold,xlabel,cmap,level):
    '''
    w_p2m: 数据矩阵，用于绘制热图和聚类分析。
    idx_pmd, idx_m1: 用于对行和列排序的索引。
    bound_pmd, bound_m1: 热图中行和列的边界，用于分区的虚线绘制。
    link, aff: 聚类方法的链接方式和距离度量方式。
    vmin, vmax: 热图的颜色范围。
    threshold: 树状图的颜色阈值。
    xlabel, ylabel: 热图的轴标签。
    cmap: 热图的颜色映射。
    level: 树状图的截断层数。
    '''
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(1,1,1)
    ax.xaxis.set_ticks_position('none')  # 不显示x轴的刻度
    ax.yaxis.set_ticks_position('none')  # 不显示y轴的刻度
    ax.set_xticks([])  # 移除x轴的刻度标记
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # plt.title(title)
    gs00 = GridSpecFromSubplotSpec(3, 3, subplot_spec=ax.get_subplotspec(), wspace=0.3, hspace=0.02,
                                   width_ratios=[0.2, 6.5, 1], height_ratios=[1, 6.5, 0.2])
    
    # 热图
    ax0 = fig.add_subplot(gs00[1, 1])
    # 树状图在上方
    ax1 = fig.add_subplot(gs00[0, 1])
    # ax2 = fig.add_subplot(gs00[0, 0])
    
    # 聚类并将聚类的结果画在相关性矩阵旁边
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None,linkage=link, affinity=aff)
    model = model.fit(w_p2m)
    am.plot_dendrogram(model, truncate_mode="level", p=level, \
                    no_labels=True, orientation='bottom', ax=ax1,color_threshold=threshold)
    ax1.set_xlim(ax1.get_xlim())
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
    
    # 重新排序之后画热图
    
#     ax0.set_aspect(1)

    cbar_ax = fig.add_subplot(gs00[1,0])  # [left, bottom, width, height]
    # 设置热图的corlorbar
    ax0=sns.heatmap(X_sort, ax=ax0, cmap=cmap, vmin=vmin, vmax=vmax,cbar=True, cbar_ax = cbar_ax, square=False)
#     cbar=fig.colorbar(ax0.collections[0], cax=cbar_ax)
    cbar_ax.yaxis.set_ticks_position("left")  # 将刻度放到左侧
    cbar_ax.yaxis.set_label_position("left")  # 将标签放到左侧
    ax0.set_xticks(ax_list, ax_list,fontsize=10)
    ax0.set_yticks(ax_list, ax_list,fontsize=10)
    # 逆转y轴以对齐钙信号热图
    ax0.invert_yaxis()
    ax0.set_xlabel(xlabel,fontsize=16)
#     ax0.set_ylabel(ylabel,fontsize=16)
#     ax0.set_xticks([])
#     ax0.set_yticks([])
    print('bound_pwd',bound_pmd)
    
    # 画边界
#     boundary = 0
#     idx_bound=np.zeros(len(bound_pmd)+1)
    for i in bound_pmd:
        ax0.axhline(y=i, color='white', linestyle='--')
    for i in bound_m1:
        ax0.axvline(x=i, color='white', linestyle='--')
        
    # 移除多余的轴（右侧或下方）
    ax_empty = fig.add_subplot(gs00[1, 2])
    ax_empty.axis("off")  # 空白占位，不显示内容
    
    plt.tight_layout()
#     for i in range(len(bound_pmd)-1):
#         boundary += bound_pmd[i]
#         idx_bound[i+1]=boundary
        
#         print('boundary', boundary)
#         print('idx_bound:',idx_bound)
#         # print(boundary)
#         ax0.axhline(y=boundary, color='white', linestyle='--')
#     idx_bound[-1]=400
#     boundary = 0
#     for i in range(len(bound_m1)-1):
#         boundary += bound_m1[i]  
#         # print(boundary)
#         ax0.axvline(x=boundary, color='white', linestyle='--')
#     return idx_bound.astype(int),fig
    return fig

# %%
def calcium_heatmap(calcium_intensity,df, col_draw,  neuron_ids,model,w_p2m,
                     show_id_stride=20, show_vol_stride=10,
                    heatmap_range=(None,0.5),
                    unit_w=0.05, unit_h = 0.2, cal_height_ratio=20, wspace=0.125, hspace = 0.125
                    ,bound_cluster = [],smooth_kernel=15,
                    font_size=90, font_color='black',
                   idx=None,vmin=0,vmax=1,threshold=None,xlabel='',
                   cmap='jet',level=None, signal_save_path=None, filename=''):
    '''
    calcium_intensity: （神经元个数 x 时间点）
    df： PCA和行为对齐
    col_draw: 需要画图的列名
    unit_w: 单位宽度(对应时间)
    unit_h: 单位高度（对应神经元数量）
    cal_height_ratio : 预期钙信号热图相对运动参数等条图的高度比例
    '''
    # 神经元数量及时间长度
    num_neurons, num_vols = calcium_intensity.shape
    
    
    if ('turn_cor' in col_draw)& ('turn_pc' in col_draw):
        # 两者画在一起
        turn_merge = True
        num_row = len(col_draw)
    else:
        turn_merge = False
        num_row = len(col_draw)+1
    
    height_ratios = [1.5 for i in range(num_row-1)]
    height_ratios.append(cal_height_ratio)
    gs = GridSpec(num_row, 4, height_ratios=height_ratios, width_ratios=[0.3,25,50, 0.3], wspace=wspace, hspace=hspace)
    fig_h = (unit_h*num_neurons)/ cal_height_ratio *  sum(height_ratios)
    fig = plt.figure(figsize=(unit_w*num_vols+unit_h*num_neurons*1.1, fig_h))
    # 热图
    ax0 = fig.add_subplot(gs[-1, 1])
    # 树状图在上方
    ax1 = fig.add_subplot(gs[-4:-1, 1])
    
    
    
    # Override the default linewidth.
    mpl.rcParams['lines.linewidth'] = font_size*0.2
    am.plot_dendrogram(model, truncate_mode="level", p=level, \
                    no_labels=True, orientation='bottom', ax=ax1,color_threshold=threshold,
                   above_threshold_color='blue')
    ax1.set_xlim(ax1.get_xlim())
    # ax1.set_xlim([0.5,2])
    ax1.xaxis.set_ticks_position('none')  # 不显示x轴的刻度
    ax1.yaxis.set_ticks_position('none')  # 不显示y轴的刻度
    ax1.set_xticks([])  # 移除x轴的刻度标记
    ax1.set_yticks([])
    ax1.invert_yaxis()
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax_list=list(np.linspace(0,w_p2m.shape[0],24,dtype=int))
    # X_sort = w_p2m[idx][:,idx]
    X_sort = w_p2m.copy()
    # 重新排序之后画热图
    
    # 设置热图的corlorbar
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.0, 0.15, 0.01, 0.4])
    ax0=sns.heatmap(X_sort, ax=ax0, cmap=cmap, vmin=vmin, vmax=vmax,cbar=True, cbar_ax = cbar_ax, square=False)
    cbar_ax.yaxis.set_ticks_position("left")  # 将刻度放到左侧
    cbar_ax.yaxis.set_label_position("left")  # 将标签放到左侧
    cbar_ax.tick_params(labelsize=font_size, pad = font_size*0.75)
    cbar_ax.set_ylabel('Correlation', fontsize = font_size*1.25, labelpad = font_size*0.75)
    
    ax0.set_xticks(ax_list, ax_list,fontsize=font_size)
    ax0.set_yticks(ax_list, ax_list,fontsize=font_size)
    ax0.tick_params(axis = 'x', pad = font_size*0.75)
    ax0.tick_params(axis='y', labelrotation=0,pad = font_size*0.75)  # 将y轴标签设置为水平
    # 逆转y轴以对齐钙信号热图
    ax0.invert_yaxis()
    ax0.set_xlabel(xlabel,fontsize=font_size*1.25, labelpad = font_size*0.75)


    bound_pmd = bound_cluster
    bound_m1 = bound_cluster
    # 画边界
    for i in bound_pmd:
        ax0.axhline(y=i, color='white', linestyle='--')
    for i in bound_m1:
        ax0.axvline(x=i, color='white', linestyle='--')
    
    # 循环画运动参数
    for i, col in enumerate(col_draw):
        # 添加子图
        ax = fig.add_subplot(gs[i, 2])  # 从第一行开始
        vector = df[col].values
        if ('turn' not in col) & ('forward' not in col):
            heatmap_data = vector[np.newaxis, :]  # 变为 1 行 N 列
            im = ax.imshow(heatmap_data, cmap='jet', aspect='auto')  # 绘制热图

            # 调整 colorbar 参数
            cax = fig.add_subplot(gs[i, 3])  # colorbar 放在右侧
            cbar = plt.colorbar(im, cax=cax, fraction=0.5, orientation='vertical', aspect = 3)
            if 'smoothed_' in col:
                col = col.replace('smoothed_', '')
            cbar.ax.set_title(col, fontsize = font_size*0.85, pad = font_size*0.25 )
            cbar.ax.tick_params(labelsize=font_size*0.5, width=5, length=5, pad = font_size*0.25)  # 设置刻度大小和宽度
            # 设置标题和轴
            ax.set_xticks([])  # 隐藏x轴刻度
            ax.set_yticks([])  # 隐藏y轴刻度
        elif ('turn' in col) & (turn_merge==False):
            heatmap_data = vector[np.newaxis, :]  # 变为 1 行 N 列
            # Define two colors
            colors = [ 'grey','#FFC832']
            # Create a ListedColormap
            two_color_cmap = ListedColormap(colors)
            im = ax.imshow(heatmap_data, cmap=two_color_cmap, aspect='auto')  # 选择 colormap
            # 添加颜色条
            cax = fig.add_subplot(gs[i, 3])  # colorbar 放在右侧
            cbar = plt.colorbar(im, cax=cax, fraction=0.5, orientation='vertical', aspect = 3)  # 通过 plt.colorbar 添加颜色条
            cbar.ax.set_title(col, fontsize = font_size*0.85, pad = font_size*0.25 )
            ax.set_xticks([])  # 设置x轴刻度
            ax.set_yticks([])  # 隐藏y轴刻度
        elif 'forward_quies' == col:
            # 不只输出forward,将quies加上
            heatmap_data = vector[np.newaxis, :]  # 变为 1 行 N 列
            # Define two colors
            colors = [ 'red','blue','white']
            # Create a ListedColormap
            cmap = ListedColormap(colors)
            bounds = [-0.5, 0.5, 1.5, 2.5]
            norm = BoundaryNorm(bounds, cmap.N)
            im = ax.imshow(heatmap_data, cmap=cmap, norm=norm, aspect='auto')
            # 添加颜色条
            cax = fig.add_subplot(gs[i, 3])  # colorbar 放在右侧
            cbar = plt.colorbar(im, cax=cax, fraction=0.5, orientation='vertical', aspect = 3)  # 通过 plt.colorbar 添加颜色条
            cbar.ax.set_title(col, fontsize = font_size*0.85, pad = font_size*0.25 )
            ax.set_xticks([])  # 设置x轴刻度
            ax.set_yticks([])  # 隐藏y轴刻度
        elif 'forward' == col:
            heatmap_data = vector[np.newaxis, :]  # 变为 1 行 N 列
            # Define two colors
            colors = [ 'red','blue']
            # Create a ListedColormap
            two_color_cmap = ListedColormap(colors)
            im = ax.imshow(heatmap_data, cmap=two_color_cmap, aspect='auto')  # 选择 colormap
            # 添加颜色条
            cax = fig.add_subplot(gs[i, 3])  # colorbar 放在右侧
            cbar = plt.colorbar(im, cax=cax, fraction=0.5, orientation='vertical', aspect = 3)  # 通过 plt.colorbar 添加颜色条
            cbar.ax.set_title(col, fontsize = font_size*0.85, pad = font_size*0.25 )
            ax.set_xticks([])  # 设置x轴刻度
            ax.set_yticks([])  # 隐藏y轴刻度

    # 倒数第二行
    if turn_merge:
        ax = fig.add_subplot(gs[-2, 2])  # 第一行
        vector_pc = df['turn_pc'].values
        vector_cor = df['turn_cor'].values

        heatmap_data_pc = vector_pc[np.newaxis, :]
        heatmap_data_cor = vector_cor[np.newaxis, :]

        # --------- PC 层：灰色(0) + 黄色(1)
        color_pc = ['grey', '#FFC832']
        cmap_pc = ListedColormap(color_pc)
        ax.imshow(heatmap_data_pc, cmap=cmap_pc, aspect='auto')

        cax = fig.add_subplot(gs[-2, 3])  # colorbar 放在右侧
        cbar = plt.colorbar(im, cax=cax, fraction=0.5, orientation='vertical', aspect = 3)
        cbar.ax.set_title('coiling/turn', fontsize = font_size*0.85, pad = font_size*0.25 )
        cbar.ax.tick_params(labelsize=font_size*0.5, width=5, length=5, pad = font_size*0.25)

        # 去掉坐标
        ax.set_xticks([])
        ax.set_yticks([])

    # 绘制钙信号热力图 
    ax = fig.add_subplot(gs[-1, 2])  # 最后一行
    heatmap=sns.heatmap(calcium_intensity, vmin=heatmap_range[0], vmax=heatmap_range[1],
                        xticklabels=np.arange(num_vols)[::show_vol_stride],
                        yticklabels=neuron_ids[::show_id_stride],cbar=False,cmap='jet',
                         cbar_kws={'orientation':'horizontal'}, ax=ax)
    
    # 调整colorbar位置到heatmap下方
    fig = ax.get_figure()
    # 获取heatmap的位置信息
    pos = ax.get_position()

    # 将colorbar放在heatmap正下方，与其等宽
    cax = fig.add_axes([pos.x0, pos.y0 - 0.08, pos.width, 0.02])  # 在heatmap下方0.08的位置

    text_str = 'ΔR/R0'
    colorbar = plt.colorbar(heatmap.collections[0], cax=cax, orientation='horizontal')
    colorbar.ax.tick_params(labelsize=font_size, pad=font_size*0.75)
    colorbar.set_label(text_str, fontsize=font_size*1.25, labelpad=font_size*0.75)
    
    # # 竖直白线根据轨迹分区
    # if len(start_indices):
    #     x_sticks = start_indices[1:-1]
    #     for x in x_sticks:
    #         ax.axvline(x=x, color='white', linestyle='--', linewidth=font_size*0.12)  # Adjust color and linestyle as needed
    
    # 横向根据聚类结果分块
    if len(bound_cluster):
        for i in bound_cluster:
            ax.axhline(y=i, color='white', linestyle='--', linewidth=font_size*0.12)
    
    ax.set_yticks(ticks=np.arange(0, num_neurons, show_id_stride), labels=neuron_ids[::show_id_stride],fontsize=font_size, 
                  color=font_color)
#     ax.set_xticks(ticks=np.arange(0, num_vols, show_vol_stride), labels=np.arange(num_vols)[::show_vol_stride],fontsize=font_size, rotation=45, color=font_color)
    ax.set_xticks(ticks=np.arange(0, num_vols, show_vol_stride), labels=df.Vol_Time.astype(int).values[::show_vol_stride],
                  fontsize=font_size, rotation=0, color=font_color)
    ax.tick_params(pad = font_size*0.75)
    #     ax.set_xticks(ticks = df.Vol_Time
    # plt.xticks(ticks=np.arange(0, num_vols, show_vol_stride), labels=np.arange(0,1300,100),fontsize=font_size, rotation=45, color=font_color)
    # plt.title('Calcium activity traces (ΔR/R0) Heatmap',fontsize=font_size,)
    ax.set_xlabel('Time(s)',fontsize=font_size*1.25, color=font_color, labelpad = font_size*0.75)
#     ax.set_ylabel('Neuron Index',fontsize=font_size*1.25,color=font_color, labelpad = font_size*0.75)
    # plt.axis('off')
    # plt.gca().spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # 设置y轴的刻度位置，使其从顶部开始，因为神经元通常从顶部向下绘制
    ax.invert_yaxis()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
#     ax.gca().invert_yaxis()
#     plt.subplots_adjust(left=0.125, bottom=0.1, right=0.1, top=0.1, wspace=0.2, hspace=0.35)
    # plt.show()
    if smooth_kernel:
        plt.savefig(f'{signal_save_path}/{filename}clus_cal_heatmap(smooth_{smooth_kernel}).png', transparent=False,dpi=100)
        plt.close()
    else:
        plt.savefig(f'{signal_save_path}/{filename}clus_cal_heatmap.png', transparent=False,dpi=100)
        plt.close()

# %%



def plot_binary_background(ax, df, column, color='blue', alpha=0.3, time_col='Vol_Time'):
    """
    在图中为 df[column] 为 1 的连续段绘制背景色。
    
    参数：
        ax: matplotlib.axes 对象
        df: 包含数据的 DataFrame
        column: 需要绘制背景的二值列名
        color: 背景颜色
        alpha: 透明度
        time_col: 时间列名
    """
    if column not in df.columns:
        return

    values = df[column].values
    in_segment = False
    start = None

    for i in range(len(values)):
        if values[i] == 1 and not in_segment:
            in_segment = True
            start = df[time_col].iloc[i]
        elif (values[i] == 0 or i == len(values)-1) and in_segment:
            in_segment = False
            end = df[time_col].iloc[i] if values[i] == 0 else df[time_col].iloc[-1]
            ax.axvspan(start, end, color=color, alpha=alpha)


# %%
def plot_motion_calcium(p_f,calcium_block, df_vol_time,t_max, col_name,col_color, n_block, block_name, smooth,delta_y, sigma):
    '''
    保存单个block,单组col的图片在指定文件夹
    调用: plot_binary_background
    '''
    fig, ax = plt.subplots(figsize=(t_max/100, n_block/4))
    # 画背景
    plot_binary_background(ax, df_vol_time, 'mask', color='black', alpha=0.5, time_col='Vol_Time')
    for i,c in enumerate(col_name):
        # reversal
        plot_binary_background(ax, df_vol_time,c , color=col_color[i], alpha=0.1, time_col='Vol_Time')
    j = 0
    for k,value in calcium_block.items():
        trace = value
        if smooth:
            trace = gaussian_filter1d(trace, sigma=sigma)
        plt.plot(df_vol_time['Vol_Time'], trace + j * delta_y,lw=0.5, label=f"Neuron {k}")
        j += 1
    # 设置坐标轴
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron index")
    desired_ticks = np.arange(0, df_vol_time['Vol_Time'].max(), 30) # 生成一个数组 [10, 30, 50, 70, 90]
    plt.xticks(desired_ticks, rotation=30, fontsize = 10)
    y_positions = np.arange(0, n_block * delta_y, delta_y)
    plt.yticks(y_positions[:n_block], calcium_block.keys(),fontsize=10)
    plt.title(block_name+f"smthed:{sigma}_{'-'.join(col_name)}")
    plt.grid(True, linestyle='--', color='grey', linewidth=0.35, alpha=0.5)
    plt.savefig(p_f+'\\'+block_name+f"smthed{sigma}_{'-'.join(col_name)}.png", bbox_inches='tight')
    plt.close()
# %%
def block_visualize_traces(p_f,calcium_intensity, calcium_dict,df, bound, smooth=True, sigma=0.3,delta_y=1.2):
    '''
    对于单个文件,可视化函数内指定的几种离散变量在不同block的神经元上，神经元数据装在dict中
    调用,plot_motion_calcium()
    '''
    # 平滑和可视化
    n_neurons, _ = calcium_intensity.shape
    print(f'shape of calcium_traces {calcium_intensity.shape}')
    t_max = df['Vol_Time'].max()
    print('分组是否和神经元数量相同：',bound.max()==n_neurons)
    bound = [0] + bound
    # 每个bound区间生成一张图
    for i in range(len(bound)):
        if i==0:
            block_idx = np.arange(0,bound[i])
            calcium_block = {list(calcium_dict.keys())[i]: list(calcium_dict.values())[i] for i in block_idx}
            block_name = f'0-{bound[i]}'
        else:
            block_idx = np.arange(bound[i-1],bound[i])
            calcium_block = {list(calcium_dict.keys())[i]: list(calcium_dict.values())[i] for i in block_idx}
            block_name = f'{bound[i-1]}-{bound[i]}'
        n_block = len(calcium_block)
        plot_motion_calcium(p_f,calcium_block, df,t_max, ['forward'],["#324DFF"],n_block, block_name, smooth,delta_y, sigma)
        plot_motion_calcium(p_f,calcium_block, df,t_max, ['turn_pc','turn_cor'],["#FFDD32", "#FF8132"],n_block, block_name, smooth,delta_y, sigma)
        plot_motion_calcium(p_f,calcium_block, df,t_max, ['forward','turn_cor'],["#324DFF", "#FF8132"],n_block, block_name, smooth,delta_y, sigma)

# %% [markdown]
# ### 连续变量

# %% [markdown]
# #### 连续运动变量高亮背景：神经元traces可视化by blocks
def plot_continuous_background(ax, df, column, cmap='bwr', alpha=0.3, time_col='Vol_Time'):
    """
    在图中根据连续变量绘制背景色，每一小段用 colormap 映射。

    参数：
        ax: matplotlib.axes 对象
        df: 包含数据的 DataFrame
        column: 需要绘制背景的连续变量列名
        cmap: colormap 名称或对象 (如 'viridis', 'plasma')
        alpha: 透明度
        time_col: 时间列名
    """
    if column not in df.columns:
        return
    
    values = df[column].values
    times = df[time_col].values

    # 创建 colormap
    norm = Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
    cmap = cm.get_cmap(cmap)

    # 遍历相邻区间
    for i in range(len(values)-1):
        if np.isnan(values[i]) or np.isnan(values[i+1]):
            continue  # 跳过 NaN
        start, end = times[i], times[i+1]
        color = cmap(norm(values[i]))
        ax.axvspan(start, end, color=color, alpha=alpha)

# %%
def plot_con_motion_calcium(p_f,calcium_block, df_vol_time,t_max, col_name,col_color, n_block, block_name, smooth,delta_y, sigma):
    '''
    保存单个block,单组col的图片在指定文件夹
    调用: plot_binary_background
    '''
    fig, ax = plt.subplots(figsize=(t_max/100, n_block/4))
    # 画背景
    plot_binary_background(ax, df_vol_time, 'mask', color='black', alpha=0.5, time_col='Vol_Time')
    for i,c in enumerate(col_name):
        # reversal
        plot_continuous_background(ax, df_vol_time,c , cmap=col_color[i], alpha=0.1, time_col='Vol_Time')
    j = 0
    for k,value in calcium_block.items():
        trace = value
        if smooth:
            trace = gaussian_filter1d(trace, sigma=sigma)
        plt.plot(df_vol_time['Vol_Time'], trace + j * delta_y,lw=0.5, label=f"Neuron {k}")
        j += 1
    # 设置坐标轴
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron index")
    desired_ticks = np.arange(0, df_vol_time['Vol_Time'].max(), 30) # 生成一个数组 [10, 30, 50, 70, 90]
    plt.xticks(desired_ticks, rotation=30, fontsize = 10)
    y_positions = np.arange(0, n_block * delta_y, delta_y)
    plt.yticks(y_positions[:n_block], calcium_block.keys(),fontsize=10)
    plt.title(block_name+f"smthed:{sigma}_{'-'.join(col_name)}")
    plt.grid(True, linestyle='--', color='grey', linewidth=0.35, alpha=0.5)
    plt.savefig(p_f+'\\'+block_name+f"smthed{sigma}_{'-'.join(col_name)}.png", bbox_inches='tight')
    plt.close()
# %%
def block_visualize_con_traces(p_f,calcium_intensity, calcium_dict,df, bound, smooth=True, sigma=0.3,delta_y=1.2):
    '''
    对于单个文件,可视化函数内指定的几种离散变量在不同block的神经元上，神经元数据装在dict中
    调用,plot_motion_calcium()
    '''
    # 平滑和可视化
    n_neurons, _ = calcium_intensity.shape
    print(f'shape of calcium_traces {calcium_intensity.shape}')
    t_max = df['Vol_Time'].max()
    print('分组是否和神经元数量相同：',bound.max()==n_neurons)
    bound = [0] + bound
    # 每个bound区间生成一张图
    for i in range(len(bound)):
        if i==0:
            block_idx = np.arange(0,bound[i])
            calcium_block = {list(calcium_dict.keys())[i]: list(calcium_dict.values())[i] for i in block_idx}
            block_name = f'0-{bound[i]}'
        else:
            block_idx = np.arange(bound[i-1],bound[i])
            calcium_block = {list(calcium_dict.keys())[i]: list(calcium_dict.values())[i] for i in block_idx}
            block_name = f'{bound[i-1]}-{bound[i]}'
        n_block = len(calcium_block)
        plot_con_motion_calcium(p_f,calcium_block, df,t_max, ['sm_CTX'],['bwr'],n_block, block_name, smooth,delta_y, sigma)
        plot_con_motion_calcium(p_f,calcium_block, df,t_max, ['sm_velocity'],['bwr'],n_block, block_name, smooth,delta_y, sigma)
        plot_con_motion_calcium(p_f,calcium_block, df,t_max, ['sm_speed'],['bwr'],n_block, block_name, smooth,delta_y, sigma)
        plot_con_motion_calcium(p_f,calcium_block, df,t_max, ['sm_ang'],['bwr'],n_block, block_name, smooth,delta_y, sigma)
        plot_con_motion_calcium(p_f,calcium_block, df,t_max, ['curvature'],['bwr'],n_block, block_name, smooth,delta_y, sigma)

# %%
def plot_lines_neural_activity(df_mot, n_col_ls,hl_cols = ["forward"],hl_color=['lightgrey'], num_bins=5, title="", color_map='inferno',
                               fs=25, x_bin=0.2, time_range_sel=[],axis_invisible = False,
                               color="#1524CA", x_label='', y_label='',sigma = 5,
                               strip_color='#9AC9DB', p_value=None, r_value=None, p_f=None):
    
    '''
    如果需要高亮事件背景
    hl_col = 'forward_sel'
    '''
    # 打印时间范围和裁剪时间
    time_range = [df_mot.Vol_Time.min(), df_mot.Vol_Time.max()]
    print(f'总的时间范围：{time_range}')
    if len(time_range_sel):
        df = df_mot[(df_mot["Vol_Time"]>=time_range_sel[0])&(df_mot["Vol_Time"]<=time_range_sel[1])]
    else:
        df = df_mot.copy()
    n_row = len(n_col_ls)

    fig, ax = plt.subplots(figsize = (10, 1.5*n_row), sharex=True)
    fig.suptitle(title, fontsize=fs)
    fig.set_constrained_layout(False)
    # gs = GridSpec(n_row, 1, height_ratios=[1.5]*n_row, wspace=0.3, hspace=0, figure=fig)
    # 可能画多个折线图
    gs = fig.add_gridspec(n_row, 1, hspace=0.2)

    for i, n_col in enumerate(n_col_ls):
        ax_beh = fig.add_subplot(gs[i])  # 每个神经元一个子图
        beh_trace = df[n_col].values
        ax_beh.plot(df['Vol_Time'], beh_trace, lw = 2,color=color)
        ylabel = n_col.split('sm_')[1] if 'sm_' in n_col else n_col
        ax_beh.set_ylabel(ylabel)
        ax_beh.spines['top'].set_visible(False)
        ax_beh.spines['right'].set_visible(False)

        # 画高亮矩形
        if len(hl_cols):
            for j, hl_col in enumerate(hl_cols):
                mask = df[hl_col].values == 1
                # 将值为1的部分高亮
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    splits = np.where(np.diff(indices) != 1)[0] + 1
                    intervals = np.split(indices, splits)
                    for interval in intervals:
                        start_idx, end_idx = interval[0], interval[-1]
                        start_time, end_time = df['Vol_Time'].iloc[start_idx], df['Vol_Time'].iloc[end_idx]
                        ax_beh.axvspan(start_time, end_time, color=hl_color[j], alpha=0.5)

        if i < n_row-1:
            ax_beh.set_xticklabels([])
        else:
            ax_last = ax_beh  # 假设这是最后一个子图
            print("进入")
    xticks = ax_last.get_xticks()
    xticklabels = [item.get_text() for item in ax_last.get_xticklabels()]
    ax_last.set_xticks(xticks)                  # 设置刻度位置
    ax_last.set_xticklabels(xticklabels)        # 设置刻度标签
    ax_last.set_xlabel('Time(s)')
    ax.axis('off')
    
    plt.show()

# %%
# 计算神经活动的导数
def compute_dF_dt(df_calcium, frame_rate=3.3,sigma_sec = 2.3):
    """
    计算每个神经元钙信号的时间导数，使用高斯导数核估计（sigma=2.3s）
    
    参数:
        df_calcium: DataFrame，每列是一个神经元的钙信号时间序列
        frame_rate: 采样频率（Hz），默认是3
    返回:
        df_derivative: 新的DataFrame，包含每个神经元的'dF/dt'列
    """
      # 秒
    sigma_frames = int(sigma_sec * frame_rate)

    df_derivative = df_calcium.copy()
    
    for col in df_derivative.columns:
        # 找出包含神经活动的列
        if col.startswith('neuron'):
            signal = df_calcium[col].values
            # 忽略NaN的影响，用fillna
            signal_clean = np.nan_to_num(signal, nan=np.nanmean(signal))
            derivative = gaussian_filter1d(signal_clean, sigma=sigma_frames, order=1, mode='nearest')
            df_derivative.loc[:,'neuron_dev' + col[6:]] = derivative

    return df_derivative
# df_dev_motion = compute_dF_dt(df_cal_motion, frame_rate=3.3,sigma_sec = 2.3)

# %% [markdown]
# ### Multiple comparison correction
# 统一校正函数
def multi_compar(df_neu_p,adj_method='fdr_bh'):
    raw_pvals = df_neu_p.p_val.values
    corrected_pvals = np.full_like(raw_pvals, np.nan)
    mask = ~np.isnan(raw_pvals)
    if np.any(mask):
        _, pvals_corr, _, _ = multipletests(raw_pvals[mask], method=adj_method)
    corrected_pvals[mask] = pvals_corr
    df_neu_p['p_cor'] = corrected_pvals
    return df_neu_p
# 根据行为统一校正函数
def multi_compar_by_beh(df, adj_method):
    beh_col_ls = df.Event.unique()
    df_p_cor = []
    for beh_col in beh_col_ls:
        df_p_beh = df[df['Event']==beh_col]
        df_p_beh_cor = multi_compar(df_p_beh,adj_method=adj_method)
        df_p_cor.append(df_p_beh_cor)
    df_p_cor = pd.concat(df_p_cor, axis=0, ignore_index=True)
    return df_p_cor


# %%
def get_event_start_idx(df, beh_col, which='start'):
    s = df[beh_col]
    diff_series = s.diff()
    if which == 'start':
        # 0 -> 1 的转变点，差值为 +1
        rise_indexes = diff_series[diff_series == 1].index
        return rise_indexes
    elif which == 'end':
        # 1 -> 0 的转变点，差值为 -1
        fall_indexes = diff_series[diff_series == -1].index
        print(f"1->0 转变的索引: {list(fall_indexes)}")
        return fall_indexes
# 神经-行为 ETA
def Paired_ttest_neu_discrete(calcium_df,beh_col, event_indices, sel_neuron_col,test_window = 80,event_note=None):
    """
    将所有神经元的事件对齐热图和均值曲线绘制在一张大图中。
    每组图：热图（6列）+ colorbar（1列）+ 均值曲线 + 空白行。
    event_indices_ls: 事件列表，第一维为一个事件，第二维为事件索引
    event_ls: 事件名称列表，应该与event_indices_ls第一个维度对应
    sort_by = 'pre'  # or 'post'
    sort_window = 10  # 排序窗口大小：多少帧
    test_window = 10 # 显著性检验窗口大小
    """
    neuron_data_list = []
    for n,n_col in enumerate(sel_neuron_col):
        df_n = calcium_df.loc[:,n_col]
        df_beh = calcium_df.loc[:,beh_col]
        # 对于单个神经元，生成一个列表，每个元素是一个event的traces
        traces = []
        for evt in event_indices:
            # evt:时间发生的行index
            # start和end找到头和尾的index
            start_idx = evt - test_window
            end_idx = evt + test_window
            if start_idx >= df_n.index[0] and end_idx <= df_n.index[-1]:
                # 判断前后是否没有事件噪声
                pre_tra = df_beh.loc[start_idx:start_idx+test_window-1].dropna().unique()
                post_tra = df_beh.loc[end_idx-test_window+1:end_idx].dropna().unique()
                if (len(pre_tra)==True) & (len(post_tra)==True):
                    trace = df_n.loc[start_idx:end_idx].values
                    traces.append(trace)
                else:
                    pass
                    # print(f'前后有噪声:pre{pre_tra}, post{post_tra}')
            else:
                print(f'神经元{n}在开始索引为{evt}事件前后window超出范围')
        if len(traces)>=3:
            # 只有事件数量超过3次才保留
            arr_1 = np.array(traces)
            if n == 0:
                print(f'(事次*timestamp){arr_1.shape}')
        else:
            print('足够计算的事件数量不足')
            return None

        # 对每个神经元进行显著性检验
        # 配对t检验或wilcoxon
        # 提取事件前后的平均值（每行是一个 trial 的平均）
        pre_vals = np.nanmean(arr_1[:, 0 : test_window], axis=1)
        post_vals = np.nanmean(arr_1[:, test_window : -1], axis=1)

        # 计算差值
        diff = post_vals - pre_vals
        diff = diff[~np.isnan(diff)]  # 去掉 NaN

        # 正态性检验
        if len(diff) < 3:
            print("数据太少，无法进行统计检验")
            t_stat, p_val, test_used = np.nan, np.nan, None
        else:
            stat, p_normal = shapiro(diff)
            if p_normal > 0.05:
                # 差值符合正态分布 → 使用配对 t 检验
                t_stat, p_val = ttest_rel(post_vals, pre_vals, nan_policy='omit')
                test_used = "paired t"
            else:
                # 差值不符合正态分布 → 使用 Wilcoxon 符号秩检验
                t_stat, p_val = wilcoxon(post_vals, pre_vals)
                test_used = "Wilcoxon"

        # print(f"{n_col}: {test_used}, p={p_val:.3f}")
        if event_note:
            neuron_data_list.append({
                    'Neuron': n_col,
                    'Event': beh_col,
                    'EventNote': event_note,
                    'test_used':test_used,
                    'p_val':p_val,
                })
        else:
            neuron_data_list.append({
                    'Neuron': n_col,
                    'Event': beh_col,
                    'test_used':test_used,
                    'p_val':p_val,
                })
        # print(f'{n_col}事件{beh_col}p_value{p_value}')
    df_neu_p = pd.DataFrame(neuron_data_list)
    return df_neu_p
# ETA可视化离散事件显著神经元
def get_p_marker(p_val):
    # 显著性标记（可自定义阈值）
    if p_val < 0.001:
        sig_label = '***'
    elif p_val < 0.01:
        sig_label = '**'
    elif p_val < 0.05:
        sig_label = '*'
    else:
        sig_label = 'n.s.'
    return sig_label
def plot_event_aligned_traces_combined(calcium_df,beh_col,df_p_cor, sel_neuron_col, event_indices,pre_window = 10,post_window=30, 
                                       sort_by='pre',sort_window=None,  frame_interval=0.3,
                                       heatmap_range = (0,0.6),save_title_str='',x_label_str='',
                                       cmap='jet', unify_yaxis=True, save_path=None):
    """
    df_p_cor: plotting p_val
    event_indices: event indexes
    beh_col: behavioral column
    sort_by = 'pre' or 'post'
    sort_window = 10  # 排序窗口大小：多少帧
    """


    # 时间轴定义
    time_axis = np.round(np.arange(-pre_window, post_window) * frame_interval, 2)
    # # 时间轴上选择7个等间距点，确定刻度位置
    # xticks = np.linspace(0, len(time_axis) - 1, 7).astype(int)
    # # 统一 y 轴范围
    # ymin = ymax = None

    for n,n_col in enumerate(sel_neuron_col):
        df_n = calcium_df.loc[:,n_col]
        df_beh = calcium_df.loc[:,beh_col]
        # 对于单个神经元，生成一个列表，每个元素是一个event的traces
        traces = []
        for evt in event_indices:
            # evt:时间发生的行index
            # start和end找到头和尾的index
            start_idx = evt - pre_window
            end_idx = evt + post_window-1
            if start_idx >= df_n.index[0] and end_idx <= df_n.index[-1]:
                # 判断前后是否没有事件噪声
                pre_tra = df_beh.loc[start_idx:evt-1].dropna().unique()
                post_tra = df_beh.loc[evt+1:end_idx].dropna().unique()
                if (len(pre_tra)==True) & (len(post_tra)==True):
                    trace = df_n.loc[start_idx:end_idx].values
                    traces.append(trace)
                else:
                    # print(f'前后有噪声:pre{pre_tra}, post{post_tra}')
                    pass
            else:
                print(f'神经元{n_col}在开始索引为{evt}事件前后window超出范围')
        if len(traces)>=3:
            # 只有事件数量超过3次才保留
            arr_1 = np.array(traces)
            if n==0:
                print(f'(事次*timestamp){arr_1.shape}')
        else:
            print('足够显示的事件数量不足')
            
        # 对每个神经元画图
        # 排序
        if sort_by == 'pre':
            if not sort_window:
                sort_window = pre_window
            # print(f'根据时间发生前{pre_window}帧的活动从高到低排序')
            sort_vals = np.nanmean(arr_1[:, pre_window - sort_window: pre_window], axis=1)
        elif sort_by == 'post':
            if not sort_window:
                sort_window = post_window
            # print(f'根据时间发生后{post_window}帧的活动从高到低排序')
            sort_vals = np.nanmean(arr_1[:, pre_window: pre_window + sort_window], axis=1)
            sorted_indices = np.argsort(-sort_vals)  # 高到低
            arr_1 = arr_1[sorted_indices]  # 重新排序
        elif sort_by==None:
            pass
        fig = plt.figure(figsize=(4, 6))  # 每个 neuron 高度小一点
        gs = GridSpec(nrows=8, ncols=7,  # 每个神经元占3行（热图、线图、空白）
           height_ratios=[3,3,3,3,3,1,1,1],
           width_ratios=[1, 1, 1, 1, 1, 1, 0.1],  # 第7列窄一些用于 colorbar
           hspace=0.1)
        row_base = 0
        ax_h = fig.add_subplot(gs[row_base:row_base+3, 0:6])     # 热图：3行高
        ax_cb = fig.add_subplot(gs[row_base:row_base+3, 6])      # colorbar：3行高，0.1列
        ax_cb_m = fig.add_subplot(gs[row_base+3:row_base+5, 6])
        ax_cb_m.axis("off")   # 不显示
        ax_m = fig.add_subplot(gs[row_base+3:row_base+5, 0:6])   # 均值线图：2行高
        ax_blank = fig.add_subplot(gs[row_base+5:row_base+8, 0:6])  # 空白：3行
        ax_blank.axis('off')

        # 热图
        # sns.heatmap(arr_1, cmap=cmap, ax=ax_h,xticklabels=False, yticklabels=False,
        #             cbar=True, cbar_ax=ax_cb,vmin=heatmap_range[0],vmax=heatmap_range[1])
        cbar = sns.heatmap(
            arr_1,
            cmap=cmap,
            ax=ax_h,
            xticklabels=False,
            yticklabels=False,
            cbar=True,
            cbar_ax=ax_cb,
            vmin=heatmap_range[0],
            vmax=heatmap_range[1]
        ).collections[0].colorbar

        # 设置 colorbar 标签
        cbar.set_label("ΔR/R", fontsize=12)
        ax_h.axvline(pre_window, color='red', ls='--',lw=2)
        ax_h.set_ylabel('Epoches')
        ax_h.set_xlabel('')
        ax_h.set_title(f'{n_col}', fontsize=15)
        ax_h.set_xticks([])       # 删除刻度
        ax_h.set_xticklabels([])  # 删除标签
        # 均值线图，先用索引画，再映射时间标签
        df_long = pd.DataFrame(arr_1)  # 直接用索引作为列
        df_long['Trial'] = np.arange(len(df_long))
        df_long = df_long.melt(id_vars='Trial', var_name='Frame', value_name='Activity')
        df_long['Frame'] = df_long['Frame'].astype(int)
        print(len(df_long))
        sns.lineplot(data=df_long, x='Frame', y='Activity', ax=ax_m,
                    ci='sd', color='grey', alpha=0.3, lw=2)
        ax_m.set_xlim(ax_h.get_xlim())
        ax_m.set_facecolor("white")
        ax_m.axvline(pre_window, color='red', ls='--', lw=2)  # 注意这里是索引 pre_window

        # ---- begin: 用于修正 xticks / 0 点对齐的代码 ----
        n_frames = arr_1.shape[1]                # 列数 = 时间点数 = len(time_axis)

        # 构造刻度索引，保证包含 pre_window（即 time=0）
        xticks_idx = np.linspace(0, n_frames - 1, 7).round().astype(int)  # 初步等距索引
        if pre_window not in xticks_idx:
            # 把离 pre_window 最近的一个刻度替换为 pre_window，确保 0 出现在刻度上
            replace_pos = np.argmin(np.abs(xticks_idx - pre_window))
            xticks_idx[replace_pos] = pre_window
        xticks_idx = np.unique(xticks_idx)  # 去重并排序（np.unique 会排序）

        # 对应的刻度标签（时间，保留小数）
        xtick_labels = [f"{time_axis[i]:.1f}" for i in xticks_idx]
        ax_m.set_xticks(xticks_idx)
        ax_m.set_xticklabels(xtick_labels)
        
        ax_m.set_xlabel(f'Time (s){x_label_str} at t=0')
        ax_m.set_ylabel('Mean activity')
        # ax_m.legend(loc='upper right', fontsize=8)
        # 将显著性结果写在均值图中间上方
        # 坐标轴右上角（与 legend 靠近）
        x_max = df_long['Frame'].max()
        y_mean = np.nanmean(df_long['Activity'])
        p_val = df_p_cor.loc[(df_p_cor['Neuron']==n_col)&(df_p_cor['Event']==beh_col),'p_cor'].values[0]
        ax_m.text(x_max*0.95, y_mean*1.1, f'p = {p_val:.3g} ({get_p_marker(p_val)})',
                    ha='right', va='bottom', fontsize=12, color='black')
        
        plt.tight_layout()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            out_path = os.path.join(save_path, f'{n_col}Alg{beh_col}{save_title_str}.png')
            plt.savefig(out_path, dpi=150)
            plt.close()
        else:
            plt.show()
    

# %%
def shuffled_cor_coef(n_timepoints,s_z,uns_z, n_shuffles):
    '''
    n_timepoints: 数据总长度
    s_z: 需要被shuffle的序列
    uns_z:不shuffle的序列, 用于计算pearson coeffient
    '''
    shuffle_rs = np.zeros(n_shuffles)
    for j in range(n_shuffles):
        # 一个随机的shift
        shift = np.random.randint(n_timepoints)
        x_shuf = np.roll(s_z[::-1], shift)  # time-reverse and circular shift
        valid_shuf = ~np.isnan(x_shuf) & ~np.isnan(uns_z)
        r_shuf, _ = pearsonr(x_shuf[valid_shuf], uns_z[valid_shuf])
        shuffle_rs[j] = r_shuf
    return shuffle_rs

def pearson_cor_shuffle_test(calcium_df, sel_neuron_col, beh_col, n_shuffles=5000,adj_method=None):
    '''
    calcium_df       # 神经数据df
    sel_neuron_col   # 指定神经(导数)列名
    beh_col          # 一个行为序列名:连续变量
    n_shuffles=5000
    '''

    # 行为序列均值化
    behavior_trace = calcium_df[beh_col]
    # 3. 可选：标准化（效果等价，但有时利于后续统一处理）
    beh_z = (behavior_trace - np.nanmean(behavior_trace)) / (np.nanstd(behavior_trace) + 1e-8)  # zero-mean for correlation
    n_timepoints = len(calcium_df)
    neuron_data_list = []
    for n_col in sel_neuron_col:
        # 对于每个神经元，提取神经信号作均值处理
        x = calcium_df.loc[:,n_col].values
        x_z = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)
        # 计算r
        valid = ~np.isnan(x_z) & ~np.isnan(beh_z)
        r_true, _ = pearsonr(x_z[valid], beh_z[valid])

        # 对于每个神经元和每个事件，随机shuffle k次后计算相关系数
        shuffle_rs = shuffled_cor_coef(n_timepoints,x_z,beh_z, n_shuffles)

        # 对于每个神经元，计算每个事件下的真实相关系数大于shuffle相关系数的比例作为p值
        # 如果真实相关系数为nan
        if np.isnan(r_true):
                p_value = np.nan
        else:
            # 根据真实相关系数的正负来计算单边尾部比例
            if r_true >= 0:
                # 真实值为正或零，计算右尾比例
                p_one_tailed = np.nanmean(shuffle_rs >= r_true)
            else: # r_true < 0
                # 真实值为负，计算左尾比例
                p_one_tailed = np.nanmean(shuffle_rs <= r_true)

            # 为了获得双边P值 (two-tailed p-value), 通常将单边P值乘以2。
            # 文章说 "greater than r (or less than, depending on the sign)"
            # 这意味着我们取了单边，然后为了表示双边，需要乘以2。
            # 但是，如果 p_one_tailed * 2 > 1，则通常截断为 1。
            p_value = min(1.0, 2 * p_one_tailed)
        neuron_data_list.append({
                    'Neuron': n_col,
                    'Event': beh_col,
                    'p_val':p_value,
                    'true_r':r_true,
                })
        # print(f'{n_col}事件{beh_col}p_value{p_value}')
    df_neu_p = pd.DataFrame(neuron_data_list)
    # 如果直接校正
    if adj_method:
        raw_pvals = df_neu_p.p_val.values
        corrected_pvals = np.full_like(raw_pvals, np.nan)
        mask = ~np.isnan(raw_pvals)
        if np.any(mask):
            _, pvals_corr, _, _ = multipletests(raw_pvals[mask], method=adj_method)
        corrected_pvals[mask] = pvals_corr
        df_neu_p['p_cor'] = corrected_pvals

    return df_neu_p
    '''
    calcium_df       # 神经数据df
    sel_neuron_col   # 指定神经(导数)列名
    beh_col          # 一个行为序列名:离散变量
    n_shuffles=5000
    '''

    # 行为序列均值化
    behavior_trace = calcium_df[beh_col]
    # 3. 可选：标准化（效果等价，但有时利于后续统一处理）
    beh_z = (behavior_trace - np.nanmean(behavior_trace)) / (np.nanstd(behavior_trace) + 1e-8)  # zero-mean for correlation
    n_timepoints = len(calcium_df)
    neuron_data_list = []
    for n_col in sel_neuron_col:
        # 对于每个神经元，提取神经信号作均值处理
        x = calcium_df.loc[:,n_col].values
        x_z = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)
        # 计算r
        valid = ~np.isnan(x_z) & ~np.isnan(beh_z)
        r_true, _ = pearsonr(x_z[valid], beh_z[valid])

        # 对于每个神经元和每个事件，随机shuffle k次后计算相关系数
        shuffle_rs = shuffled_cor_coef(n_timepoints,x_z,beh_z, n_shuffles)

        # 对于每个神经元，计算每个事件下的真实相关系数大于shuffle相关系数的比例作为p值
        # 如果真实相关系数为nan
        if np.isnan(r_true):
                p_value = np.nan
        else:
            # 根据真实相关系数的正负来计算单边尾部比例
            if r_true >= 0:
                # 真实值为正或零，计算右尾比例
                p_one_tailed = np.nanmean(shuffle_rs >= r_true)
            else: # r_true < 0
                # 真实值为负，计算左尾比例
                p_one_tailed = np.nanmean(shuffle_rs <= r_true)

            # 为了获得双边P值 (two-tailed p-value), 通常将单边P值乘以2。
            # 文章说 "greater than r (or less than, depending on the sign)"
            # 这意味着我们取了单边，然后为了表示双边，需要乘以2。
            # 但是，如果 p_one_tailed * 2 > 1，则通常截断为 1。
            p_value = min(1.0, 2 * p_one_tailed)
        neuron_data_list.append({
                    'Neuron': n_col,
                    'Event': beh_col,
                    'p_val':p_value,
                    'true_r':r_true,
                })
        # print(f'{n_col}事件{beh_col}p_value{p_value}')
    df_neu_p = pd.DataFrame(neuron_data_list)
    # 如果直接校正
    if adj_method:
        raw_pvals = df_neu_p.p_val.values
        corrected_pvals = np.full_like(raw_pvals, np.nan)
        mask = ~np.isnan(raw_pvals)
        if np.any(mask):
            _, pvals_corr, _, _ = multipletests(raw_pvals[mask], method=adj_method)
        corrected_pvals[mask] = pvals_corr
        df_neu_p['p_cor'] = corrected_pvals

    return df_neu_p
# %%
def corr_neu_con_var(df,behavior_col_ls=['sm_velocity','sm_speed','sm_ang','curvature','sm_CTX'],
                     neuron_col_str='neuron',adj_method = 'fdr_bh' ):
    '''
    输入: 包含神经数据和beh数据列的df
    neuron_col_str:神经数据列关键字
    behavior_col_ls:连续变量行为列
    输出: 所有神经元和行为变量的相关性df,包括p值和r
    '''
    # behavior_col_ls = ['sm_velocity','sm_speed','sm_ang','curvature','sm_CTX']
    # neuron_col_str = 'neuron'
    neu_col_ls = [col for col in df.columns if neuron_col_str in col]
    ls_neu_con_p = []
    for beh_col in behavior_col_ls:
        # adj_method = 'bonferroni'# 校正方法可选'fdr_by', 'fdr_bh', 'bonferroni'
        df_neu_p = pearson_cor_shuffle_test(df, neu_col_ls, beh_col, n_shuffles=5000, adj_method=adj_method)
        ls_neu_con_p.append(df_neu_p)
    df_neu_con_p = pd.concat(ls_neu_con_p, axis=0, ignore_index=True)
    return df_neu_con_p
# #### 相关分析
# 可视化所有行为变量的热图
def plot_correlation_heatmap_nomask_from_df(df, 
                                     row_col = 'Behavior',
                                     col_col = 'Neuron',
                                     r_col = 'true_r',
                                     p_col = 'p_cor',
                                     colormap = "coolwarm",
                                     alpha_original=0.05,
                                     title="Neuron-Behavior_Correlation_Heatmap",
                                     x_ticklabels = [],
                                     r_threshold = 0.30,
                                     mask_color = '#f0f0f0',
                                     save_path=None,
                                     save_title=None,
                                    fs = 16):
    """
    根据 cross_correlation_shuffle_analysis 函数的DataFrame输出，绘制相关性热图。

    参数:
    analysis_results_df (pd.DataFrame): cross_correlation_shuffle_analysis 函数返回的DataFrame。
    alpha_original (float): 原始的显著性水平，用于与校正后的P值比较。默认为0.05。
    title (str): 图表的标题。

    """
    analysis_results_df = df.copy()
    # 替换Behavior值标签
    analysis_results_df[row_col] = analysis_results_df[row_col].astype(str)
    mask = analysis_results_df[row_col].str.contains("smoothed_")
    analysis_results_df.loc[mask, row_col] = (
        analysis_results_df.loc[mask, row_col]
        .str.replace("smoothed_", "", regex=False)
    )
    analysis_results_df.loc[:,row_col] = analysis_results_df[row_col].str.replace('_',' ')
    # 保证顺序一致
    original_neuron_order = analysis_results_df[col_col].unique()
    original_behavior_order = analysis_results_df[row_col].unique()
    
    # 2. 将 'Neuron' 和 'Behavior' 列转换为有序的分类类型
    analysis_results_df[col_col] = pd.Categorical(
        analysis_results_df[col_col], categories=original_neuron_order, ordered=True
    )
    analysis_results_df[row_col] = pd.Categorical(
        analysis_results_df[row_col], categories=original_behavior_order, ordered=True
    )
    # 将长格式的DataFrame转换为宽格式的矩阵
    true_correlations = analysis_results_df.pivot_table(
        index=row_col, columns=col_col, values=r_col
    )
    p_values_corrected = analysis_results_df.pivot_table(
        index=row_col, columns=col_col, values=p_col
    )
    width = true_correlations.shape[1] * 1
    height = true_correlations.shape[0] * 1
    fig = plt.figure(figsize=(width,height), constrained_layout=True) # Create figure and axes explicitly
    # Create a divider for the existing axes. This makes it easy to append new axes next to it.
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.2])  # 主图在左边
    # 添加 colorbar axes
    # 添加主图 axes（位置: 左, 下, 宽, 高）
    cbar_ax = fig.add_axes([0.65, 0.1, 0.002, 0.2])  # colorbar 精确位置（右侧）
    sns.heatmap(
        true_correlations,
        ax =ax,
        annot=False,
        fmt=".2f",
        cmap=colormap,
        cbar=True,
        cbar_ax=cbar_ax,
        linewidths=1,
        linecolor='white',
        square=True,
        vmin = -0.7,
        vmax = 0.7,
#         mask=sig_mask
    )
    ax.set_yticks(np.arange(len(true_correlations.index)) + 0.5)  # 每一行的中心
    ax.set_yticklabels(true_correlations.index, rotation=0, ha='center')

#     ax.set_facecolor(mask_color)  # <- 设置mask区域为白色
#     在显著的格子上添加星号 (*) 标记
    significance_levels = {
        0.001: "***",  # p < 0.001
        0.01: "**",   # p < 0.01
        0.05: "*",    # p < 0.05
    }
    sorted_levels = sorted(significance_levels.keys())
    for i in range(true_correlations.shape[0]):   # 行（行为变量）
        for j in range(true_correlations.shape[1]):  # 列（神经元）
            p_val = p_values_corrected.iloc[i, j]
            true_r = true_correlations.iloc[i, j]
            if not pd.isna(p_val) and not pd.isna(true_r):
                asterisks = ""
                # 按照显著性水平选择星号
                for thresh in sorted_levels:
                    if (p_val < thresh) & (abs(true_r) > r_threshold):
                        asterisks = significance_levels[thresh]
                if asterisks:
                    ax.text(
                        j + 0.5, i + 0.5,  # cell 的中心坐标
                        asterisks,
                        ha='center', va='center',
                        color='black', fontsize=fs*0.8, fontweight='bold'
                    )

    cbar_ax.tick_params(labelsize=fs*0.75) # Example: adjust colorbar tick font size
    cbar_ax.set_ylabel('Correlation Value', rotation=270, labelpad=fs*0.75, fontsize=fs*0.75)
    ax.set_title(title, fontsize=fs,y = 1.05)
    ax.set_xlabel(col_col+" Index", fontsize=fs, labelpad = fs)
    ax.set_ylabel('', fontsize=fs, labelpad = fs*1.1)
    if len(x_ticklabels):
        ax.set_xticklabels(x_ticklabels)
#     ax.set_xticklabels([])
    ax.tick_params(axis='x', rotation=45, labelsize=fs)
    ax.tick_params(axis='y', rotation=0, labelsize=fs*0.9, pad = fs*3)
    # 获取当前的刻度标签
    labels = ax.get_yticklabels()
    # 设置对齐方式（以居中为例）
    ax.set_yticklabels(labels, ha='center')
    if save_path:
        if save_title:
            plt.savefig(save_path+'\\'+save_title, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            plt.savefig(save_path+'\\'+title, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        print(f"图片已保存到: {save_path}")


# %% [markdown]
# #### 根据变量进行单神经元可视化
# + correlation coefficient 大于 0.2

# %%
# 选择显著的神经元进行分组箱线图可视化
def plot_binned_neural_activity(calcium_df, neuron_col,
                                beh_col, num_bins=5, title="",
                                fs=25,x_bin = 0.2,
                               color='#2878B5',x_label = '',y_label = 'ΔR/R',
                               strip_color ='#9AC9DB',p_value=None,r_value=None, p_f = None):
    """
    根据行为变量的分箱，绘制神经活动的箱线图。
    calcium_df: 包含行为变量曲线的df
    neuron_idx: 神经元索引
    beh_col:变量列名
    """
    if beh_col not in calcium_df.columns:
        print(f"错误: 行为变量 '{beh_col}' 不存在于数据中。")
        return
#     neuron_name = 'neuron'+str(neuron_idx)
    neuron_name = neuron_col
    # 提取神经信号
    temp_df = calcium_df[[neuron_name, beh_col]].copy()
    temp_df.dropna(inplace=True)

    if temp_df.empty:
        print(f"警告: 神经元 '{neuron_name}' 和行为 '{beh_col}' 没有有效数据点可用于绘图。")
        return
    
    # 求出行为变量的上下界
    min_beh_val = temp_df[beh_col].min()
    max_beh_val = temp_df[beh_col].max()
    
    # 将行为变量分箱
    temp_df['beh_bin'], bins = pd.cut(
        temp_df[beh_col],
        bins=num_bins,
        labels=False,
        include_lowest=True,
        retbins=True,
        duplicates='drop'
    )

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    temp_df['bin_center'] = temp_df['beh_bin'].map(lambda x: bin_centers[int(x)] if pd.notna(x) else np.nan)
    sorted_temp_df = temp_df.sort_values(by='bin_center', ascending=True)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    sns.boxplot(
        x='bin_center',
        y=neuron_name,
        data=sorted_temp_df,
        showcaps=True,
        color = color,
        boxprops=dict(facecolor='none', edgecolor='grey', linewidth=2),
        whiskerprops=dict(color='grey', linewidth=2),
        capprops=dict(color='grey', linewidth=2),
        medianprops=dict(color='black', linewidth=5),
        showfliers=False
    )
     # 叠加散点图
    sns.stripplot(
        x='bin_center',
        y=neuron_name,
        data=sorted_temp_df,
        color=strip_color,
        size=10,
        alpha=0.1,
        jitter=True
    )
    if title:
        plt.title(f"#{neuron_name} {title}",
                  fontsize=fs, pad=fs)
    else:
        plt.title(f"#{neuron_name} Activity over {beh_col}",
                  fontsize=fs, pad=fs)
    if x_label:
        plt.xlabel(f"{x_label}", fontsize=fs*0.75, labelpad=fs*0.75)
    else:
        plt.xlabel(f"{beh_col}", fontsize=fs*0.75, labelpad=fs*0.75)
    plt.axhline(y=0, ls='dashed', color='black', linewidth=2)
    plt.ylabel(y_label, fontsize=fs*0.75)
    
    # 手动设置标签
    ax.set_xticklabels([]) # Clear existing labels
    
    bin_center = list(sorted_temp_df['bin_center'].unique())
    custom_labels_sparse =  [f"{x:.3f}" for x in bin_center]

#     ax.set_xticks(custom_positions)
    ax.set_xticklabels(custom_labels_sparse, rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=fs*0.75)
    ax.tick_params(axis='y', labelsize=fs*0.75)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加p值和相关性系数信息
        
    stat_text = ""
    if r_value is not None:
        stat_text += f"$r$ = {r_value:.2f}  "
    if p_value is not None:
        if p_value < 0.001:
            stat_text += f"\n$p$ < 0.001"
        else:
            stat_text += f"\n$p$ = {p_value:.3f}"
    if stat_text:
        plt.text(0.5, 0.9, stat_text, fontsize=fs*0.7, ha='center', va='center',color='red', transform=ax.transAxes)
    if p_f:
        os.makedirs(p_f, exist_ok=True)
        file_path = os.path.join(p_f, f'bin{beh_col}_{neuron_name}{x_label}.png')
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

# %%
# 对于每种连续变量，打印对应神经元的折线图
# 对于每种连续变量，打印对应神经元的折线图
def plot_lines_neural_activity(calcium_df, n_col_ls,df_neu_p_cor,
                               beh_col,hl_col = ["forward"],hl_color=['lightgrey'], num_bins=5, title="", color_map='inferno',
                               fs=25, x_bin=0.2, time_range_sel=[],axis_invisible = False,
                               color='#2878B5', x_label='Time(s)', y_label='',sigma = 5,
                               strip_color='#9AC9DB', label_pr = True, p_f=None):
    
    '''
    这个函数打印行为和神经变量平滑后的trace,并用指定离散变量高亮。
    ***不适合可视化离散的行为变量！！***
    df_neu_p_cor:提供p值和r值
    如果需要高亮事件背景,提供一个列表
    hl_col = ['forward_sel']
    时间范围可选
    '''
    if len(n_col_ls) == 0:
        print(f"⚠️ 没有显著相关的神经元用于 {beh_col}，跳过绘图")
        return None
    if beh_col not in calcium_df.columns:
        print(f"错误: 行为变量 '{beh_col}' 不存在于数据中。")

    # 提取神经,行为和时间traces
    col_ls  = list(n_col_ls).copy()
    col_ls.append(beh_col)
    col_ls.append('Vol_Time')
    if len(hl_col):
        col_ls.extend(hl_col)
    temp_df = calcium_df[col_ls].copy()
    temp_df.dropna(inplace=True)

    # 打印时间范围和裁剪时间
    time_range = [calcium_df.Vol_Time.min(), calcium_df.Vol_Time.max()]
    print(f'总的时间范围：{time_range}')
    if len(time_range_sel):
        df = temp_df[(temp_df["Vol_Time"]>=time_range_sel[0])&(temp_df["Vol_Time"]<=time_range_sel[1])]
    else:
        df = temp_df.copy()
    n_row = len(n_col_ls)+1
#     fig,ax = plt.subplots(n_row, 1, figsize = (10, 2.5*n_row))
    fig, ax = plt.subplots(figsize = (10, 1.5*n_row))
    fig.set_constrained_layout(False)
    ratios = [1.5]*(n_row-1) + [1.5]
    gs = GridSpec(n_row, 1, height_ratios=ratios, wspace=0.3, hspace=0, figure=fig)
    # 可能画多个折线图
    ax_neu = fig.add_subplot(gs[0:n_row-1], sharex=ax)
    ax_beh = fig.add_subplot(gs[-1], sharex=ax)
    # 神经元trace颜色
    cmap = plt.cm.get_cmap(color_map)
    color_positions = np.linspace(0, 0.8, len(n_col_ls))
    colors = [cmap(pos) for pos in color_positions]
    neuron_to_color = dict(zip(n_col_ls, colors))
    # 行为平滑曲线
    df[beh_col+'smhed'] = gaussian_filter1d(df[beh_col], sigma=sigma)
    beh_trace = df[beh_col].values
    beh_n_z = (beh_trace-np.nanmean(beh_trace))/np.nanstd(beh_trace)
    beh_smthed = gaussian_filter1d(beh_n_z, sigma=sigma)
    # 画神经元Z-score线图
    for i, n_col in enumerate(n_col_ls):
        cal_n = df[n_col].values
        mean = np.nanmean(cal_n)
        std = np.nanstd(cal_n)
        cal_n_z = (cal_n-mean)/std

        # 先打印灰色的线
        
        ax_neu.plot(df['Vol_Time'], beh_smthed+(i*3), lw = 3, color='grey',alpha=0.5)
        # 高斯平滑
        cal_n_z_smoothed = gaussian_filter1d(cal_n_z, sigma=sigma)
        trace_y = cal_n_z_smoothed + (i * 3)
        ax_neu.plot(df['Vol_Time'], trace_y, lw = 2,color=neuron_to_color[n_col])
        # ✅ 在左侧标注神经元编号（代替 legend）
        x_min = df['Vol_Time'].iloc[0]
        ax_neu.text(x_min-10, i * 3, n_col.replace('neuron','#'), ha='right', va='center',
                    fontsize=fs, color=neuron_to_color[n_col])
        
        if label_pr:
            # 打印p值和r值
            p_n = df_neu_p_cor[(df_neu_p_cor['Neuron']==n_col)&(df_neu_p_cor['Event']==beh_col)]['p_cor'].values[0]
            r_n = df_neu_p_cor[(df_neu_p_cor['Neuron']==n_col)&(df_neu_p_cor['Event']==beh_col)]['true_r'].values[0]
            # 取最后一个点坐标，在右侧标注文字
            x_pos = df['Vol_Time'].iloc[-1]
            y_pos = trace_y[-1]
            ax_neu.text(
                x_pos+5,   # 向右偏移一些
                y_pos, 
                f"r={r_n:.2f}\np={p_n:.3f}",
                ha='left', va='center',
                fontsize=fs*0.75, color=neuron_to_color[n_col]
            )
    
    # 画高亮矩形
    if len(hl_col):
        for h, hl_col in enumerate(hl_col):
            mask = df[hl_col].values == 1
            # 将值为1的部分高亮
            indices = np.where(mask)[0]
            if len(indices) > 0:
                splits = np.where(np.diff(indices) != 1)[0] + 1
                intervals = np.split(indices, splits)
                for interval in intervals:
                    start_idx, end_idx = interval[0], interval[-1]
                    start_time, end_time = df['Vol_Time'].iloc[start_idx], df['Vol_Time'].iloc[end_idx]
                    ax_neu.axvspan(start_time, end_time, color=hl_color[h], alpha=0.5)
                    ax_beh.axvspan(start_time, end_time, color=hl_color[h], alpha=0.5)
    
    
    df[beh_col+'smhed'] = gaussian_filter1d(df[beh_col], sigma=sigma)
    ax_beh.plot(df['Vol_Time'], df[beh_col+'smhed'], lw = 2, color='black', label = n_col)
    ax_beh.set_ylabel(beh_col.replace('sm_',''), fontsize=fs)
    ax_neu.set_title(title, fontsize=fs, pad = fs)
    ax.axis('off')
    if axis_invisible:
        # 去掉神经元trace子图边框
        for spine in ax_neu.spines.values():
            spine.set_visible(False)
        ax_neu.tick_params(left=False, right=False, labelleft=False, bottom=False, top=False, labelbottom=False)
        ax_beh.spines['top'].set_visible(False)
        ax_beh.spines['right'].set_visible(False)
    if p_f:
        os.makedirs(p_f, exist_ok=True)
        file_path = os.path.join(p_f, f'Line{title}.png')
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

def plot_lines_neural_activity_disc(calcium_df, n_col_ls,df_neu_p_cor,beh_col,
                               hl_col = ["forward"],hl_color=['lightgrey'], num_bins=5, title="", color_map='inferno',
                               fs=25, x_bin=0.2, time_range_sel=[],axis_invisible = False,
                               color='#2878B5', x_label='Time(s)', y_label='',sigma = 5,
                               strip_color='#9AC9DB', label_p = True,label_r = True, p_f=None, Event_col='Event'):
    
    '''
    这个函数适合可视化离散的行为变量与神经trace
    df_neu_p_cor:提供p值和r值
    如果需要高亮事件背景,提供一个列表
    hl_col = ['forward_sel']
    时间范围可选
    '''

    # 提取神经,行为和时间traces
    col_ls  = list(n_col_ls).copy()
    col_ls.append('Vol_Time')
    if len(hl_col):
        col_ls.extend(hl_col)
    temp_df = calcium_df[col_ls].copy()
    temp_df.dropna(inplace=True)

    # 打印时间范围和裁剪时间
    time_range = [calcium_df.Vol_Time.min(), calcium_df.Vol_Time.max()]
    delta_time = time_range[1]-time_range[0]
    print(delta_time)
    print(f'总的时间范围：{time_range}')
    if len(time_range_sel):
        df = temp_df[(temp_df["Vol_Time"]>=time_range_sel[0])&(temp_df["Vol_Time"]<=time_range_sel[1])]
    else:
        df = temp_df.copy()
    n_row = len(n_col_ls)
#     fig,ax = plt.subplots(n_row, 1, figsize = (10, 2.5*n_row))
    fig, ax = plt.subplots(figsize = (delta_time/80, 1.5*n_row))
    fig.set_constrained_layout(False)
    ratios = [1.5]*(n_row)
    gs = GridSpec(n_row, 1, height_ratios=ratios, wspace=0.3, hspace=0, figure=fig)
    # 可能画多个折线图
    ax_neu = fig.add_subplot(gs[0:n_row], sharex=ax)
    # 神经元trace颜色
    cmap = plt.cm.get_cmap(color_map)
    color_positions = np.linspace(0, 0.8, len(n_col_ls))
    colors = [cmap(pos) for pos in color_positions]
    neuron_to_color = dict(zip(n_col_ls, colors))
    # 画神经元Z-score线图
    for i, n_col in enumerate(n_col_ls):
        cal_n = df[n_col].values
        mean = np.nanmean(cal_n)
        std = np.nanstd(cal_n)
        cal_n_z = (cal_n-mean)/std

        # 先打印灰色的线
        # 高斯平滑
        cal_n_z_smoothed = gaussian_filter1d(cal_n_z, sigma=sigma)
        trace_y = cal_n_z_smoothed + (i * 3)
        ax_neu.plot(df['Vol_Time'], trace_y, lw = 2,color=neuron_to_color[n_col])
        # ✅ 在左侧标注神经元编号（代替 legend）
        x_min = df['Vol_Time'].iloc[0]
        ax_neu.text(x_min-10, i * 3, n_col.replace('neuron','#'), ha='right', va='center',
                    fontsize=fs, color=neuron_to_color[n_col])
        
        if label_p:
            # 打印p值和r值
            p_n = df_neu_p_cor[(df_neu_p_cor['Neuron']==n_col)&(df_neu_p_cor[Event_col]==beh_col)]['p_cor'].values[0]
            
            # 取最后一个点坐标，在右侧标注文字
            x_pos = df['Vol_Time'].iloc[-1]
            y_pos = trace_y[-1]
            ax_neu.text(
                x_pos+5,   # 向右偏移一些
                y_pos, 
                f"p={p_n:.3f}",
                ha='left', va='center',
                fontsize=fs*0.75, color=neuron_to_color[n_col]
            )
        if label_r:
            r_n = df_neu_p_cor[(df_neu_p_cor['Neuron']==n_col)&(df_neu_p_cor[Event_col]==beh_col)]['true_r'].values[0]
            # 取最后一个点坐标，在右侧标注文字
            x_pos = df['Vol_Time'].iloc[-1]
            y_pos = trace_y[-1]
            ax_neu.text(
                x_pos+5,   # 向右偏移一些
                y_pos, 
                f"\nr={r_n:.2f}",
                ha='left', va='center',
                fontsize=fs*0.75, color=neuron_to_color[n_col]
            )
    # 画高亮矩形
    if len(hl_col):
        for h, hl_col in enumerate(hl_col):
            mask = df[hl_col].values == 1
            # 将值为1的部分高亮
            indices = np.where(mask)[0]
            if len(indices) > 0:
                splits = np.where(np.diff(indices) != 1)[0] + 1
                intervals = np.split(indices, splits)
                for interval in intervals:
                    start_idx, end_idx = interval[0], interval[-1]
                    start_time, end_time = df['Vol_Time'].iloc[start_idx], df['Vol_Time'].iloc[end_idx]
                    ax_neu.axvspan(start_time, end_time, color=hl_color[h], alpha=0.5)
    ax_neu.set_title(title, fontsize=fs, pad = fs)
    ax.axis('off')
    if axis_invisible:
        # 去掉神经元trace子图边框
        for spine in ax_neu.spines.values():
            spine.set_visible(False)
        ax_neu.tick_params(left=False, right=False, labelleft=False, bottom=True, top=False, labelbottom=True)
    ax_neu.set_xlabel('Time(s)', fontsize=fs)
    ax_neu.tick_params(axis='x', labelsize=fs*0.8)
    if p_f:
        os.makedirs(p_f, exist_ok=True)
        file_path = os.path.join(p_f, f'Line{title}.png')
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
    plt.show()
# %%
def visualize_cont_beh_by_neu(df_neu,calcium_df, beh_col, p_uplim=0.05, r_threshold=0.2, p_f_con = None):
    '''
    save all pics into one 'beh_col' folder
    '''
    # 生成一个文件夹
    pf_con_beh = p_f_con+f'\\{beh_col}'
    os.makedirs(pf_con_beh, exist_ok=True)
    # 筛选
    df_neu['sign'] = 0
    df_neu.loc[(df_neu['p_cor']<=p_uplim)&(df_neu['true_r'].abs()>r_threshold),'sign'] = 1

    # 显著的神经元
    sign_neu_ls = df_neu[(df_neu['sign']==1)&((df_neu['Event']==beh_col))]['Neuron'].unique()
    # 首先作折线图
    plot_lines_neural_activity(calcium_df, sign_neu_ls,df_neu,
                               beh_col,hl_col = ["forward"],hl_color=["#9AB3CF"], num_bins=5,
                                 title=f"NeuCorrWith{beh_col.replace('sm_','')}HighlightedByReverse", color_map='inferno',
                               fs=25, x_bin=0.2, time_range_sel=[],axis_invisible = True,
                               color='#2878B5', x_label='', y_label='',sigma = 5,
                               strip_color='#9AC9DB', label_pr=True, p_f=pf_con_beh)
    plot_lines_neural_activity(calcium_df, sign_neu_ls,df_neu,
                               beh_col,hl_col = ["turn_pc"],hl_color=["#EEE536"], num_bins=5,
                                 title=f"NeuCorrWith{beh_col.replace('sm_','')}HighlightedByCoilingTurn", color_map='inferno',
                               fs=25, x_bin=0.2, time_range_sel=[],axis_invisible = True,
                               color='#2878B5', x_label='', y_label='',sigma = 5,
                               strip_color='#9AC9DB', label_pr=True, p_f=pf_con_beh)
    plot_lines_neural_activity(calcium_df, sign_neu_ls,df_neu,
                               beh_col,hl_col = ['forward',"turn_pc"],hl_color=["#9AB3CF","#EEE536"], num_bins=5,
                                 title=f"NeuCorrWith{beh_col.replace('sm_','')}HighlightedByRevTurn", color_map='inferno',
                               fs=25, x_bin=0.2, time_range_sel=[],axis_invisible = True,
                               color='#2878B5', x_label='', y_label='',sigma = 5,
                               strip_color='#9AC9DB', label_pr=True, p_f=pf_con_beh)
    # 对每个神经元作箱线tuning图
    for n_col in sign_neu_ls:
        p_val = df_neu[(df_neu['Event']==beh_col)&(df_neu['Neuron']==n_col)]['p_cor'].values[0]
        r_val = df_neu[(df_neu['Event']==beh_col)&(df_neu['Neuron']==n_col)]['true_r'].values[0]
        plot_binned_neural_activity(calcium_df, n_col,
                                beh_col, num_bins=5, title="",
                                fs=25,x_bin = 0.2,
                               color='#2878B5',x_label = '',y_label = 'ΔR/R',
                               strip_color ='#9AC9DB',p_value=p_val,r_value=r_val, p_f = pf_con_beh)
    
# %%

import os
import cv2 as cv
import numpy as np
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from scipy.ndimage import distance_transform_edt
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 


# 导入自定义工具函数库，核心的骨架处理逻辑在process_frame中
from worm_analysis_utils import (
    calc_distance, calc_angle, filter_subset,
    find_circular_paths_by_node_ids, find_branching_paths, process_frame
)

def data_analysis(file_folder, segment_file, save_file,
                   video_name = 'c1.mp4', video_version='new',body_len = 50, window_k = 3):
    '''
    对某个文件夹下的视频和对应的掩码进行骨架分析'''
    # 加载数据
    mask_path = file_folder + segment_file   # 二值化视频路线
    video_path = file_folder + video_name    # 原视频路线
    mask_v = cv.VideoCapture(mask_path) # 分割视频，掩码二值
    video = cv.VideoCapture(video_path) # 原视频
    if not video.isOpened() or not mask_v.isOpened():
        raise RuntimeError(f"Failed to open video files:\n  Video: {video_path}\n  Mask: {mask_path}")

    output_path = file_folder + save_file
    # 对于原始的时间戳，取5帧采样
    f_data = np.loadtxt(file_folder + 'c1.txt', delimiter=',')
    if video_version == 'old':
        indices = np.concatenate(([0], np.arange(5, len(f_data), 5)))
    elif video_version == 'new':
        # 找到最后一列等于 1 的行索引
        indices = np.where(f_data[:, -1] == 1)[0]
    t_c = f_data[indices, 1].astype(int)
    # 读取载物台数据
    stage_data = np.loadtxt(file_folder + 'stage_data.txt', delimiter=',')
    t_stage = stage_data[:, 0]
    x_stage = -stage_data[:, 3]
    y_stage = -stage_data[:, 4]
    # 平滑x和y
    window_size = 40
    x_m = np.convolve(x_stage, np.ones(window_size) / window_size, mode='same')
    y_m = np.convolve(y_stage, np.ones(window_size) / window_size, mode='same')
    # 加载咽喉位置数据

    try:
        p_data = np.loadtxt(os.path.join(file_folder, 'tracking_data.txt'), delimiter=',')
    except FileNotFoundError:
        print("tracking_data.txt not found, trying data1.txt...")
        p_data = np.loadtxt(os.path.join(file_folder, 'data1.txt'), delimiter=',')

    p_t = p_data[:, 0]
    p_x = p_data[:, 1]   # 咽喉的像素位置
    p_y = p_data[:, 2]
    p_index = p_data[:, 3]   # 如果太小说明tracking失败了
    # 完成时间对齐
    idx_p = []
    idx_stages = []
    for t in t_c:
        idx_p.append(np.searchsorted(p_t, t))
        idx_stages.append(np.searchsorted(t_stage, t))
    # 并行计算
    batch_size = 1024 # 并行大小设置
    frame_count = 0
    analysis_data = {}
    futures = []
    iou = []
    prev_mask = None
    total_frames = len(t_c)
    print()
    with ThreadPoolExecutor(max_workers=24) as executor, tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            # 读取追踪视频和分割视频数据
            ret_mask, frame = mask_v.read()
            ret_video, image = video.read()
            if frame_count >= len(t_c):
                break
            if not ret_mask or not ret_video:
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            binary_mask = (gray / 255) > 0.5
            # 计算逐帧中间的iou大小
            if prev_mask is not None:
                intersection = np.logical_and(binary_mask, prev_mask).sum()
                iou1 = intersection / np.sum(prev_mask) if np.sum(prev_mask) > 0 else 0
                iou2 = intersection / np.sum(binary_mask) if np.sum(binary_mask) > 0 else 0
                iou.append((iou1, iou2))
            prev_mask = binary_mask
            # 将逐帧处理加入线程
            futures.append(executor.submit(process_frame, 
                                           frame_count, frame, image, p_x, p_y,
                                             p_index, idx_stages, idx_p, x_m, y_m,
                                             body_l=body_len, k = window_k
            ))
            frame_count += 1
            pbar.update(1)  # 进度条更新
            # 超过batch-size后提取结论
            if len(futures) >= batch_size:
                for future in futures:
                    actual_frame_count, result = future.result()
                    if result is not None:
                        analysis_data[actual_frame_count] = result
                futures = []

    # 处理剩余frames
    for future in futures:
        actual_frame_count, result = future.result()
        if result is not None:
            analysis_data[actual_frame_count] = result
    mask_v.release()
    video.release()
    # 存储分析数据
    np.savez(output_path, analysis_data)
    np.savez(file_folder + 'iou_results.npz', iou=iou)
    print(f'Saved longest skeleton paths to {output_path}')
    print(f'Saved IoU results to {file_folder}iou_results.npz')

if __name__ == '__main__':
    # 寻找root 文件夹下所有符合要求的文件进行骨架提取及分析处理
    root_path = r'Y:\\SZX\\2025_wbi_analysis\\good_WBI\\to_do\\redo'
    segment_file = r'c1_onnx.mp4'
    # segment_file = r'c1_new_onnx.mp4'
    save_file = r'output-0516.npz'
    video_name = 'c1.mp4'
    video_version = 'new'   # 从2025年初的只要是c1.txt包含保存帧列信息的都为new
    folders = []
    for root, dirs, files in os.walk(root_path):
        if segment_file in files and save_file not in files:
            folders.append(root)
    print('\n'.join(folders))
    for folder in folders:
        print(f"\nProcessing folder: {folder}")
        try:
            data_analysis(folder + '\\', segment_file, save_file,
                           video_name=video_name,
                           video_version=video_version,body_len = 200, window_k = 40)
        except Exception as e:
            print(f"Error processing {folder}: {e}")
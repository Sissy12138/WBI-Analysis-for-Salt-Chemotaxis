import numpy as np
import cv2 as cv
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from scipy.ndimage import distance_transform_edt
import networkx as nx

def calc_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# def calc_angle(vector1, vector2):
#     '''
#     计算不带符号的角度
#     '''
#     dot_product = np.dot(vector1, vector2)
#     magnitude1 = np.linalg.norm(vector1)
#     magnitude2 = np.linalg.norm(vector2)
#     cos_theta = dot_product / (magnitude1 * magnitude2)
#     angle = np.arccos(cos_theta)
#     return np.degrees(angle)
def calc_angle(vector1, vector2):
    """
        计算带符号的角度（-180° 到 180°）
        正角度表示从 vector1 到 vector2 是逆时针旋转
        负角度表示顺时针旋转
        """
    dot_product = np.dot(vector1, vector2)
    cross_product = np.cross(vector1, vector2)  # 在 2D 中，这给出标量值
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cos_theta = dot_product / (magnitude1 * magnitude2)
    sin_theta = cross_product / (magnitude1 * magnitude2)
    # 使用 arctan2 获取带符号的角度
    angle_rad = np.arctan2(sin_theta, cos_theta)
    return np.degrees(angle_rad)

def filter_subset(arr, threshold=500):
    '''
    过滤数组中较小的元素，返回大于threshold的子集'''
    arr_sorted = sorted(arr)
    n = len(arr_sorted)
    if n < 2:
        return []
    for i in range(n - 1):
        if arr_sorted[i] + arr_sorted[i+1] > threshold:
            if n - i >= 2:
                return arr_sorted[i:]
            else:
                return []
    return []

def find_circular_paths_by_node_ids(path_summary, circle_threshold=500):
    circular_paths = []
    # 寻找拥有相同节点的path:如果两个路径节点相同，分到一组
    node_groups = {}
    for idx in path_summary.index:
        src = path_summary.loc[idx, 'node-id-src']
        dst = path_summary.loc[idx, 'node-id-dst']
        node_pair = tuple(sorted([src, dst]))
        if node_pair not in node_groups:
            node_groups[node_pair] = []
        node_groups[node_pair].append(idx)
    # 排除长度短于threshold 的圆环
    for node_pair, path_indices in node_groups.items():
        if len(path_indices) > 1:
            l_arr = [path_summary.loc[i, 'branch-distance'] for i in path_indices]
            f_indices = filter_subset(l_arr, threshold=circle_threshold)
            if len(f_indices) > 0:
                circular_paths.append(path_indices)
    # 使用节点编号构建graph
    G = nx.Graph()
    for idx in path_summary.index:
        src = path_summary.loc[idx, 'node-id-src']
        dst = path_summary.loc[idx, 'node-id-dst']
        G.add_edge(src, dst, path_idx=idx)
    # 寻找graph 中的环形
    for cycle in nx.cycle_basis(G):
        path_indices = set()
        for i in range(len(cycle)):
            src = cycle[i]
            dst = cycle[(i + 1) % len(cycle)]
            if G.has_edge(src, dst):
                path_indices.add(G[src][dst]['path_idx'])
        if path_indices:  # Only add non-empty sets
            path_list = list(path_indices)
            total_length = sum(path_summary.loc[idx, 'branch-distance']
                             for idx in path_list)
            # 排除短于threshold 的圆形
            if total_length > circle_threshold and path_list not in circular_paths:
                circular_paths.append(path_list)
    return circular_paths

def find_branching_paths(path_summary):
    # 构建graph
    G = nx.Graph()
    for idx in path_summary.index:
        src = path_summary.loc[idx, 'node-id-src']
        dst = path_summary.loc[idx, 'node-id-dst']
        branch_type = path_summary.loc[idx, 'branch-type']
        G.add_edge(src, dst, path_idx=idx, branch_type=branch_type)
    
    matching_paths = []
    for idx in path_summary.index:
        if path_summary.loc[idx, 'branch-distance'] < 200:
            src = path_summary.loc[idx, 'node-id-src']
            dst = path_summary.loc[idx, 'node-id-dst']
            # Count type-1 paths connected to source node
            src_connections = [
                G[src][neighbor]['path_idx']
                for neighbor in G[src]
                if (G[src][neighbor]['path_idx'] != idx and  # not the current path
                    path_summary.loc[G[src][neighbor]['path_idx'], 'branch-type'] == 1 and
                    path_summary.loc[G[src][neighbor]['path_idx'], 'branch-distance'] > 50
                    )
            ]
            # Count type-1 paths connected to destination node
            dst_connections = [
                G[dst][neighbor]['path_idx']
                for neighbor in G[dst]
                if (G[dst][neighbor]['path_idx'] != idx and  # not the current path
                    path_summary.loc[G[dst][neighbor]['path_idx'], 'branch-type'] == 1 and
                    path_summary.loc[G[dst][neighbor]['path_idx'], 'branch-distance'] > 50)
            ]
            # Check if both endpoints have exactly two type-1 connections
            if len(src_connections) == 2 and len(dst_connections) == 2:
                matching_paths.append({
                    'main_path': idx,
                    'src_connections': src_connections,
                    'dst_connections': dst_connections
                })
    return matching_paths

def process_frame(actual_frame_count, frame, image,
                   p_x, p_y, p_index, idx_stages,
                     idx_p, x_m, y_m, body_l=50, k = 3):
    '''
    receive one video frame and return analysis result
    body_l: 计算头部方向的单边像素长度
    k: 计算运动方向的时间窗'''
    
    # 到达文件末尾时退出
    if idx_p[actual_frame_count] >= len(p_x):
        return actual_frame_count, None
    # 筛选咽喉识别率大于0.9的点
    if p_index[actual_frame_count] < 0.9:
        return actual_frame_count, None
    # 读取咽喉位置
    middle_pos = [p_x[idx_p[actual_frame_count]] * 2, p_y[idx_p[actual_frame_count]] * 2]
    # 读取处理图片
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gaussian = cv.GaussianBlur(gray, (7, 7), 2)
    _, binary_image = cv.threshold(gaussian, 127, 255, cv.THRESH_BINARY)
    binary_bool = binary_image > 0
    # 寻找尾部位置
    dist = distance_transform_edt(binary_bool)
    max_loc = np.unravel_index(np.argmax(dist), dist.shape)
    tail = [max_loc[1], max_loc[0]]
    # 计算骨架
    skeleton = skeletonize(binary_image // 255)

    # 如果骨架不存在，返回none
    if np.sum(skeleton) == 0:
        return actual_frame_count, None
    # 提取骨架信息
    try:
        ske = Skeleton(skeleton)
    except ValueError:
        return actual_frame_count, None
    # 总结骨架路径
    summary = summarize(ske, separator='-')
    if len(summary) > 6:
        return actual_frame_count, None
    # 保留有效骨架：筛选 type大于2 或 长度大于50 的path
    valid_paths = summary[
            (summary['branch-type'] > 1) |
            (summary['branch-distance'] >= 50)
        ]
    if len(valid_paths) == 0:
        return actual_frame_count, None
    # 获取所有骨架坐标
    all_paths = []
    for path_idx in summary.index:
        path_coords = ske.path_coordinates(path_idx)
        all_paths.append(path_coords)
    # 寻找距离咽喉最近的骨架
    min_distance = float('inf')
    closest_path_idx = None
    paths = all_paths
    for path_idx in valid_paths.index:
        path_coords = paths[path_idx]
        distances = np.sqrt(
            (path_coords[:, 1] * 2 - middle_pos[0] * 2)**2 + 
            (path_coords[:, 0] * 2 - middle_pos[1] * 2)**2
        )
        min_path_distance = np.min(distances)
        if min_path_distance < min_distance:
            min_distance = min_path_distance
            closest_path_idx = path_idx
            min_index = np.where(distances == min_path_distance)[0][0]
    
    if closest_path_idx is None:
        return actual_frame_count, None
    # 存距离咽喉坐标最近的骨架的坐标
    closest_path = ske.path_coordinates(closest_path_idx)

    # 存储发生卷曲行为的frames
    # 存储满足靠近咽喉，端到端path的frames
    circle_threshold = 300
    circular_paths = find_circular_paths_by_node_ids(summary, circle_threshold)
    branching_paths = find_branching_paths(summary)
    circular = 0
    branching = 0
    choose_frame = 0
    if circular_paths:
        circular = 1
    if branching_paths:
        branching = 1
    if summary.loc[closest_path_idx, 'branch-type'] == 0:
        choose_frame = 1

    # 根据尾部位置确定骨架方向
    head_pos = [int(closest_path[0, 1]), int(closest_path[0, 0])]
    tail_pos = [int(closest_path[-1, 1]), int(closest_path[-1, 0])]
    if calc_distance(tail_pos, tail) > calc_distance(head_pos, tail):
        closest_path = np.flip(closest_path, axis=0)
        head_pos = [int(closest_path[0, 1]), int(closest_path[0, 0])]
        tail_pos = [int(closest_path[-1, 1]), int(closest_path[-1, 0])]
        min_index = len(closest_path) - 1 - min_index
    # 确定身体方向和载物台移动方向,计算前进后退方向
    idx_stage = idx_stages[actual_frame_count]
    # body_l = 50 # 取咽喉部位前后50像素位置作为身体方向
    start_index = max(min_index - body_l, 0)
    stop_index = min(len(closest_path) - 1, min_index + body_l)
    vector1 = np.array([int(closest_path[start_index, 1]), int(closest_path[start_index, 0])]) \
        - np.array([int(closest_path[stop_index, 1]), int(closest_path[stop_index, 0])])
    
    # 差分计算angle_m
    # k = 3 # 差分时间窗
    if idx_stage - k < 0 or idx_stage + k >= len(x_m):
        angle_m = None
        angle_md = None
        vector_m = None
        # vector_m = [x_m[idx_stage + 1] - x_m[idx_stage], y_m[idx_stage + 1] - y_m[idx_stage]]
    elif (idx_stage - k > 0) & (idx_stage + k < len(x_m)):
        dx = x_m[idx_stage + k] - x_m[idx_stage - k]
        dy = y_m[idx_stage + k] - y_m[idx_stage - k]
        vector_m = (dx, dy)
        angle_md = calc_angle(vector1, vector_m)
        angle_m = np.abs(calc_angle(vector1, vector_m))
        # 检查 angle_m
        if angle_md is None or np.isnan(angle_md):
            angle_md = 0.0
        if angle_m is None or np.isnan(angle_m):
            angle_m = 0.0
    else:
        return actual_frame_count, None
        
    # 原始代码
    # if idx_stage + 1 < len(x_m) - 1:
    #     vector_m = [x_m[idx_stage + 1] - x_m[idx_stage], y_m[idx_stage + 1] - y_m[idx_stage]]
    # elif idx_stage < len(x_m) - 1:
    #     vector_m = [x_m[idx_stage] - x_m[idx_stage - 1], y_m[idx_stage] - y_m[idx_stage - 1]]
    # else:
    #     return actual_frame_count, None
    # angle_m = np.abs(calc_angle(vector1, vector_m))
    # 检查 angle_m
    # if angle_m is None or np.isnan(angle_m):
    #     angle_m = 0.0

    body_angle = np.arctan2(vector1[1], vector1[0])
    if vector_m:
        # 归一化头部方向和运动方向向量
        norm_m = np.linalg.norm(vector_m)
        if norm_m == 0:
            pass
        else:
            # 向量归一化模长为1
            
            vector_m = vector_m/norm_m
    norm1 = np.linalg.norm(vector1)
    vector1 = vector1/norm1
    
    # 计算食物浓度
    head_mask = np.zeros_like(binary_bool, dtype=np.uint8)
    mask_radius = 100 #以头部周围半径100像素为范围
    cv.circle(head_mask, (head_pos[0], head_pos[1]), mask_radius, 1, -1)
    search_mask = np.logical_and(head_mask, ~binary_bool)
    search_mask = search_mask.astype(np.uint8)
    search_mask = cv.resize(search_mask, (image.shape[1], image.shape[0]))
    laplacian = cv.Laplacian(image, cv.CV_64F, ksize=5) #求Laplace对比度
    laplacian_masked = np.multiply(laplacian, search_mask)
    laplacian_masked = cv.convertScaleAbs(laplacian_masked)
    _, stddev = cv.meanStdDev(laplacian, mask=search_mask) #以对比度的标准差作为判断依据
    
    
    
    # 存储在npz中的数据
    return actual_frame_count, {
        'closest path': closest_path_idx,
        'all_paths': all_paths,
        'tail': tail, 
        'angle_m': angle_m,     # 头部方向与运动方向夹角的绝对值
        'angle_md': angle_md,     # 头部方向与运动方向夹角
        'vector_h':vector1,     # 头部方向
        'vector_m':vector_m,    # 运动方向
        'pos_phr':middle_pos,
        'body_angle': body_angle,
        'stddev': stddev[0][0], 
        'path_summary': summary,
        'circular': circular,
        'branching': branching,
        'choose_frame': choose_frame
    }
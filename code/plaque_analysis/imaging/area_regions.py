"""Regions helpers for the legacy red/white area workflow."""

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt
from plaque_analysis.settings.area import AreaConfig


def merge_small_regions_to_nearest_large(white_mask, small_region_threshold=25,
                                         merge_distance_threshold=10):
    """
    基于欧几里得距离将小白色区域合并到最近的大白色区域
    """
    labeled_original, num_original = ndimage.label(white_mask)

    if num_original == 0:
        return labeled_original, {'original_count': 0, 'merged_count': 0, 'merge_map': {},
                                  'large_regions': [], 'small_regions': [],
                                  'small_region_threshold': small_region_threshold,
                                  'merge_distance_threshold': merge_distance_threshold}

    region_info = {}
    for i in range(1, num_original + 1):
        region_mask = (labeled_original == i)
        size = np.sum(region_mask)
        if size > 0:
            y_coords, x_coords = np.where(region_mask)
            center_y = np.mean(y_coords)
            center_x = np.mean(x_coords)
            region_info[i] = {
                'size': size,
                'center': (center_x, center_y),
                'coords': (x_coords, y_coords),
                'is_large': size >= small_region_threshold
            }

    large_regions = {k: v for k, v in region_info.items() if v['is_large']}
    small_regions = {k: v for k, v in region_info.items() if not v['is_large']}

    merge_map = {}

    for small_id, small_info in small_regions.items():
        if not large_regions:
            continue

        min_distance = float('inf')
        nearest_large_id = None

        for large_id, large_info in large_regions.items():
            small_coords = np.column_stack((small_info['coords'][0], small_info['coords'][1]))
            large_coords = np.column_stack((large_info['coords'][0], large_info['coords'][1]))

            if len(small_coords) > 100:
                indices = np.random.choice(len(small_coords), 100, replace=False)
                small_coords = small_coords[indices]
            if len(large_coords) > 100:
                indices = np.random.choice(len(large_coords), 100, replace=False)
                large_coords = large_coords[indices]

            distances = cdist(small_coords, large_coords, 'euclidean')
            min_dist = np.min(distances)

            if min_dist < min_distance:
                min_distance = min_dist
                nearest_large_id = large_id

        if nearest_large_id is not None and min_distance <= merge_distance_threshold:
            merge_map[small_id] = nearest_large_id

    merged_labels = np.zeros_like(labeled_original)
    new_label_map = {}
    new_label_counter = 1

    for large_id in large_regions.keys():
        new_label_map[large_id] = new_label_counter
        merged_labels[labeled_original == large_id] = new_label_counter
        new_label_counter += 1

    for small_id in small_regions.keys():
        if small_id in merge_map:
            target_label = new_label_map[merge_map[small_id]]
            merged_labels[labeled_original == small_id] = target_label
        else:
            new_label_map[small_id] = new_label_counter
            merged_labels[labeled_original == small_id] = new_label_counter
            new_label_counter += 1

    merge_info = {
        'original_count': num_original,
        'merged_count': new_label_counter - 1,
        'merge_map': merge_map,
        'large_regions': list(large_regions.keys()),
        'small_regions': list(small_regions.keys()),
        'small_region_threshold': small_region_threshold,
        'merge_distance_threshold': merge_distance_threshold,
    }

    return merged_labels, merge_info


def calculate_red_areas_for_all_regions(merged_labels, mask_red, custom_params=None, *, config: AreaConfig = AreaConfig()):
    """
    计算所有白色区域周围的红色区域，避免重复计算
    如果提供了custom_params，使用自定义的搜索圈参数
    """
    # 统计所有区域的面积
    region_sizes = {}
    for i in range(1, np.max(merged_labels) + 1):
        size = np.sum(merged_labels == i)
        if size > 0:
            region_sizes[i] = size

    # 按面积排序
    sorted_regions = sorted(region_sizes.items(), key=lambda x: x[1], reverse=True)

    # 红色区域计算结果
    all_red_area_results = []

    # 存储每个红色像素的候选分配信息
    red_pixel_candidates = {}

    # 为所有区域计算搜索圈和候选红色像素
    for region_id, size in sorted_regions:
        region_mask = (merged_labels == region_id)

        # 如果提供了自定义参数，使用自定义参数；否则使用默认参数
        if custom_params and region_id in custom_params:
            params = custom_params[region_id]
            dilation_size = params['dilation_size']
            inner_erosion = params['erosion_size']
            range_multiplier = params['range_multiplier']
        else:
            dilation_size = config.dilation_size
            inner_erosion = config.inner_erosion
            range_multiplier = config.search_range_multiplier

        # 根据白色区域大小动态计算搜索圈大小
        area_scale = min(2.0, max(0.5, np.sqrt(size) / 50))
        dynamic_search_size = int(dilation_size * range_multiplier * area_scale)
        inner_erosion_size = max(2, int(inner_erosion * area_scale))

        # 创建搜索区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (dynamic_search_size * 2 + 1, dynamic_search_size * 2 + 1))
        dilated = cv2.dilate(region_mask.astype(np.uint8), kernel)

        # 内部腐蚀
        kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (inner_erosion_size * 2 + 1, inner_erosion_size * 2 + 1))
        eroded = cv2.erode(region_mask.astype(np.uint8), kernel_erosion)

        ring_mask = (dilated > 0) & (eroded == 0)

        # 计算该搜索圈内的红色区域面积
        red_in_ring = (mask_red > 0) & ring_mask

        # 为了解决相邻区域重叠问题，我们采用距离加权分配策略
        # 对于重叠区域，根据距离白色区域边界的远近进行分配
        if np.sum(red_in_ring) > 0:
            # 计算白色区域的欧几里得距离变换
            white_dist = distance_transform_edt(~region_mask)

            # 对于每个红色像素，记录它到当前白色区域的距离
            y_coords, x_coords = np.where(red_in_ring)
            for y, x in zip(y_coords, x_coords):
                pixel_key = (y, x)
                dist = white_dist[y, x]

                if pixel_key not in red_pixel_candidates:
                    red_pixel_candidates[pixel_key] = []
                red_pixel_candidates[pixel_key].append((region_id, dist, size))

        # 存储区域基本信息
        all_red_area_results.append({
            'region_id': region_id,
            'white_area': size,
            'white_area_um2': size * config.pixel_size_x * config.pixel_size_y,
            'search_ring_area': np.sum(ring_mask),
            'search_ring_area_um2': np.sum(ring_mask) * config.pixel_size_x * config.pixel_size_y,
            'region_mask': region_mask,
            'ring_mask': ring_mask
        })

    # 统一的红色像素分配
    # 初始化一个字典来记录每个区域最终分配的红色像素
    final_red_assignments = {result['region_id']: np.zeros_like(mask_red, dtype=bool)
                             for result in all_red_area_results}

    # 处理每个红色像素，分配给"吸引力"最强的白色区域
    for pixel_key, candidates in red_pixel_candidates.items():
        y, x = pixel_key

        if len(candidates) == 1:
            # 只有一个候选区域，直接分配
            region_id, _, _ = candidates[0]
            final_red_assignments[region_id][y, x] = True
        else:
            # 多个候选区域，使用加权评分选择
            # 评分基于：1) 区域大小（大区域优先） 2) 距离（近的区域优先）
            best_score = -1
            best_region_id = None

            for region_id, dist, region_size in candidates:
                # 计算综合评分：区域大小权重 + 距离权重
                # 距离越近分数越高，区域越大分数越高
                size_score = np.log(region_size + 1)  # 使用对数避免过大区域主导
                dist_score = 1.0 / (dist + 1)  # 距离越近分数越高

                # 综合评分（可根据需要调整权重）
                total_score = size_score * 0.3 + dist_score * 0.7

                if total_score > best_score:
                    best_score = total_score
                    best_region_id = region_id

            if best_region_id is not None:
                final_red_assignments[best_region_id][y, x] = True

    # 更新每个区域的红色区域计算结果
    for result in all_red_area_results:
        region_id = result['region_id']
        result['red_in_ring'] = final_red_assignments[region_id]
        result['red_area'] = np.sum(final_red_assignments[region_id])
        result['red_area_um2'] = result['red_area'] * config.pixel_size_x * config.pixel_size_y

    return all_red_area_results, sorted_regions

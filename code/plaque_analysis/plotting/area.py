"""Plotting helpers for the legacy red/white area workflow."""

import cv2
import numpy as np
from plaque_analysis.settings.area import AreaConfig
from plaque_analysis.imaging.area_boundaries import get_red_white_boundary


def select_and_annotate_top_regions(bgr, merged_labels, top_n=10, output_path=None, *, config: AreaConfig = AreaConfig()):
    """
    筛选面积最大的前N个区域，并在图像上添加编号标注
    """
    # 统计每个区域的面积
    region_sizes = {}
    for i in range(1, np.max(merged_labels) + 1):
        size = np.sum(merged_labels == i)
        if size > 0:
            region_sizes[i] = size

    # 按面积排序，选择前N大区域
    sorted_regions = sorted(region_sizes.items(), key=lambda x: x[1], reverse=True)
    top_regions = sorted_regions[:top_n]

    print(f"前{top_n}大白色区域面积:")
    for i, (region_id, size) in enumerate(top_regions, 1):
        area_um2 = size * config.pixel_size_x * config.pixel_size_y
        print(f"  区域{i}: ID={region_id}, 面积={size} 像素, {area_um2:.2f} um²")

    # 创建标注图像
    annotation_img = bgr.copy()

    # 为每个区域计算质心并添加编号
    for i, (region_id, size) in enumerate(top_regions, 1):
        region_mask = (merged_labels == region_id)
        if np.sum(region_mask) == 0:
            continue

        # 计算区域质心
        y_coords, x_coords = np.where(region_mask)
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))

        # 添加编号（带透明度）
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(2.0, size / 5000))  # 根据区域大小自适应字体大小
        thickness = max(1, int(font_scale * 2))  # 根据字体大小自适应粗细
        text_color = (255, 0, 0)  # BGR格式，蓝色
        background_color = (255, 255, 255)  # BGR格式，白色
        background_alpha = 0.6  # 背景透明度 (0=完全透明, 1=完全不透明)

        text = str(i)
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # 创建临时图像用于alpha混合
        overlay = annotation_img.copy()

        # 添加半透明白色背景
        cv2.rectangle(overlay,
                      (center_x - text_width // 2 - 5, center_y - text_height // 2 - 5),
                      (center_x + text_width // 2 + 5, center_y + text_height // 2 + 5),
                      background_color, -1)

        # 应用透明度混合背景
        cv2.addWeighted(overlay, background_alpha, annotation_img, 1 - background_alpha, 0, annotation_img)

        # 添加文字（使用纯色，不透明）
        cv2.putText(annotation_img, text,
                    (center_x - text_width // 2, center_y + text_height // 2),
                    font, font_scale, text_color, thickness)

        # 注释掉轮廓绘制，只显示编号
        # contours, _ = cv2.findContours(region_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(annotation_img, contours, -1, (0, 255, 0), 2)

    # 保存标注图像
    if output_path:
        cv2.imwrite(output_path, annotation_img)

    return annotation_img, top_regions


def create_three_split_images(bgr, merged_labels, all_red_area_results, top_n=10):
    """
    生成三幅拆分图：
    1. 白色区域前十的图
    2. 含有搜索圈的图
    3. 不含搜索圈但显示白色区域各自搜索到的红色区域的边界
    """
    # 获取前N大区域的数据
    top_red_results = all_red_area_results[:top_n]

    # 存储区域数据
    region_data = []

    # 预定义30种不同的颜色（BGR格式），避免白色和红色
    predefined_colors = [
        (141, 211, 199),
        (255, 255, 179),
        (190, 186, 218),
        (128, 177, 211),
        (253, 180, 98),
        (179, 222, 105),
        (252, 205, 229),
        (217, 217, 217),
        (188, 128, 189),
        (204, 235, 197),
        (255, 237, 111),
        (84, 179, 69),
        (5, 185, 226),
        (98, 0, 170),
        (73, 101, 176),
        (158, 22, 157)
    ]

    # 过滤掉红色(0,0,255)和白色(255,255,255)
    filtered_colors = [color for color in predefined_colors if color != (0, 0, 255) and color != (255, 255, 255)]

    # 处理前N个区域
    for i, result in enumerate(top_red_results):
        # 从预定义颜色中选择
        if i < len(filtered_colors):
            color = filtered_colors[i]
        else:
            # 如果颜色不够，循环使用
            color = filtered_colors[i % len(filtered_colors)]

        # 存储区域数据
        region_data.append({
            'region_id': result['region_id'],
            'region_mask': result['region_mask'],
            'color': color,
            'ring_mask': result['ring_mask'],
            'red_in_ring': result['red_in_ring'],
            'size': result['white_area'],
            'red_area': result['red_area']
        })

        print(f"区域{i + 1}: 白色面积={result['white_area']}像素, 红色面积={result['red_area']}像素, "
              f"搜索圈面积={result['search_ring_area']}像素")

    # ===== 图1: 白色区域前十的图 =====
    white_regions_img = bgr.copy()
    colored_white_img = np.zeros_like(bgr)

    for data in region_data:
        colored_white_img[data['region_mask']] = data['color']

    # 添加原始图像作为背景（半透明）
    background_alpha = 0.3
    white_regions_result = cv2.addWeighted(white_regions_img, background_alpha, colored_white_img, 1 - background_alpha,
                                           0)

    # ===== 图2: 含有搜索圈的图 =====
    search_rings_img = bgr.copy()

    for data in region_data:
        # 绘制搜索圈轮廓
        contours, _ = cv2.findContours(data['ring_mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(search_rings_img, contours, -1, data['color'], 2)

    # ===== 图3: 红色区域边界图 =====
    red_boundaries_img = bgr.copy()

    for data in region_data:
        # 绘制红色区域边界
        if np.sum(data['red_in_ring']) > 0:
            red_contours, _ = cv2.findContours(data['red_in_ring'].astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(red_boundaries_img, red_contours, -1, data['color'], 2)

    # ===== 为每幅图添加图例 =====
    def add_legend_to_image(img, title, region_data, include_red_info=False):
        """为图像添加图例"""
        legend_height = 40 * len(region_data) + 60
        legend_width = 400
        legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

        # 添加标题
        cv2.putText(legend, title, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        for i, data in enumerate(region_data):
            color = data['color']

            cv2.rectangle(legend, (10, 40 + i * 40), (30, 60 + i * 40), color, -1)

            if include_red_info:
                # 包含红色区域信息
                label1 = f"Region {i + 1}: {data['size']} px"
                label2 = f"Red: {data['red_area']} px"
                cv2.putText(legend, label1, (40, 50 + i * 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(legend, label2, (40, 65 + i * 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            else:
                # 仅白色区域信息
                label = f"Region {i + 1}: {data['size']} px"
                cv2.putText(legend, label, (40, 55 + i * 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 将图例添加到图像右侧
        if img.shape[0] > legend_height:
            new_width = img.shape[1] + legend_width + 10
            new_height = max(img.shape[0], legend_height + 10)
            combined_img = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255
            combined_img[0:img.shape[0], 0:img.shape[1]] = img
            combined_img[10:10 + legend_height, img.shape[1] + 10:img.shape[1] + 10 + legend_width] = legend
            return combined_img
        else:
            new_width = max(img.shape[1], legend_width)
            new_height = img.shape[0] + legend_height + 10
            combined_img = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255
            combined_img[0:img.shape[0], 0:img.shape[1]] = img
            combined_img[img.shape[0] + 10:img.shape[0] + 10 + legend_height, 0:legend_width] = legend
            return combined_img

    # 为三幅图分别添加图例
    white_regions_with_legend = add_legend_to_image(
        white_regions_result, "Top White Regions", region_data, include_red_info=False
    )

    search_rings_with_legend = add_legend_to_image(
        search_rings_img, "Search Rings", region_data, include_red_info=False
    )

    red_boundaries_with_legend = add_legend_to_image(
        red_boundaries_img, "Red Areas in Search Rings", region_data, include_red_info=True
    )

    return (white_regions_with_legend, search_rings_with_legend, red_boundaries_with_legend,
            region_data)


def visualize_region_relationship(bgr, mask_red, mask_white, output_path=None):
    """
    可视化红白区域关系
    """
    # 创建彩色可视化
    vis = bgr.copy()

    # 用半透明红色标记红色区域
    red_color = np.zeros_like(vis)
    red_color[mask_red > 0] = [0, 0, 255]  # BGR: 红色
    cv2.addWeighted(red_color, 0.3, vis, 0.7, 0, vis)

    # 用半透明白色标记白色区域
    white_color = np.zeros_like(vis)
    white_color[mask_white > 0] = [255, 255, 255]  # BGR: 白色
    cv2.addWeighted(white_color, 0.3, vis, 0.7, 0, vis)

    # 标记边界
    boundary = get_red_white_boundary(mask_red, mask_white)
    vis[boundary > 0] = [0, 255, 0]  # 绿色边界

    # 添加文本说明
    cv2.putText(vis, "Red Region (Outside)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis, "White Region (Inside)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, "Boundary", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, vis)
    return vis


def visualize_segmentation_comparison(bgr, mask_red, mask_white, mask_red_ref, mask_white_ref, output_path=None):
    """
    可视化分割结果与参考图的对比
    """
    H, W = bgr.shape[:2]
    comparison = np.zeros((H, W * 2, 3), dtype=np.uint8)

    # 左侧：分割结果
    left_vis = bgr.copy()
    red_color = np.zeros_like(left_vis)
    red_color[mask_red > 0] = [0, 0, 255]
    cv2.addWeighted(red_color, 0.5, left_vis, 0.5, 0, left_vis)
    white_color = np.zeros_like(left_vis)
    white_color[mask_white > 0] = [255, 255, 255]
    cv2.addWeighted(white_color, 0.3, left_vis, 0.7, 0, left_vis)
    cv2.putText(left_vis, "Segmentation Result", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 右侧：参考图（如果存在）
    if mask_red_ref is not None and mask_white_ref is not None:
        right_vis = bgr.copy()
        red_color_ref = np.zeros_like(right_vis)
        red_color_ref[mask_red_ref > 0] = [0, 0, 255]
        cv2.addWeighted(red_color_ref, 0.5, right_vis, 0.5, 0, right_vis)
        white_color_ref = np.zeros_like(right_vis)
        white_color_ref[mask_white_ref > 0] = [255, 255, 255]
        cv2.addWeighted(white_color_ref, 0.3, right_vis, 0.7, 0, right_vis)
        cv2.putText(right_vis, "Reference", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        right_vis = np.zeros_like(left_vis)
        cv2.putText(right_vis, "No Reference Available", (W // 4, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    comparison[:, :W] = left_vis
    comparison[:, W:] = right_vis

    # 添加分割线
    cv2.line(comparison, (W, 0), (W, H), (255, 255, 255), 2)

    if output_path:
        cv2.imwrite(output_path, comparison)
    return comparison

"""Segmentation helpers for the legacy red/white area workflow."""

import cv2
import numpy as np
from scipy import ndimage
from plaque_analysis.settings.area import AreaConfig
from plaque_analysis.analysis.area_reference import evaluate_segmentation_quality


def enhanced_color_clustering(bgr):
    """
    增强的颜色聚类方法，专门针对红白分割
    """
    # 转换到多个颜色空间
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    H, W = bgr.shape[:2]

    # 创建增强的特征向量
    # 使用多个颜色空间的通道组合
    features = np.zeros((H * W, 5), dtype=np.float32)

    # 特征1: LAB的a通道（红-绿轴）
    features[:, 0] = lab[:, :, 1].reshape(-1)

    # 特征2: BGR的红色通道
    features[:, 1] = bgr[:, :, 2].reshape(-1)

    # 特征3: HSV的色调（Hue）
    features[:, 2] = hsv[:, :, 0].reshape(-1)

    # 特征4: HSV的饱和度
    features[:, 3] = hsv[:, :, 1].reshape(-1)

    # 特征5: LAB的亮度
    features[:, 4] = lab[:, :, 0].reshape(-1)

    # 归一化每个特征
    for i in range(features.shape[1]):
        features[:, i] = (features[:, i] - np.mean(features[:, i])) / (np.std(features[:, i]) + 1e-6)

    # K-means聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _compactness, labels, centers = cv2.kmeans(
        data=features, K=2, bestLabels=None, criteria=criteria,
        attempts=10, flags=cv2.KMEANS_PP_CENTERS
    )

    labels = labels.reshape(H, W)

    # 根据特征确定红色类
    # 红色应该在LAB的a通道和BGR的R通道有较高值
    red_scores = []
    for i in range(2):
        # 计算每个类的"红色程度"得分
        red_score = centers[i, 0] + centers[i, 1]  # a通道 + R通道
        red_scores.append(red_score)

    red_idx = int(np.argmax(red_scores))

    mask_red = (labels == red_idx).astype(np.uint8) * 255
    mask_white = (labels != red_idx).astype(np.uint8) * 255

    return mask_red, mask_white, labels, centers


def threshold_based_red_white_segmentation(bgr):
    """
    基于颜色阈值的红白分割方法
    """
    # 转换到HSV颜色空间，对红色检测更好
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 定义红色范围（考虑红色在HSV中的两个区间）
    # 红色范围1: 低色调值
    lower_red1 = np.array([0, 20, 20])  # 大幅降低饱和度和亮度阈值
    upper_red1 = np.array([25, 255, 255])  # 大幅扩大色调范围

    # 红色范围2: 高色调值 - 大幅放宽
    lower_red2 = np.array([155, 20, 20])  # 大幅降低饱和度和亮度阈值
    upper_red2 = np.array([180, 255, 255])  # 保持色调范围

    # 创建红色掩膜
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # 定义白色范围（低饱和度、高亮度）
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # 清理噪声（使用较小的内核，避免去除太多小区域）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)

    return mask_red, mask_white


def hybrid_red_white_segmentation(bgr, mask_red_ref=None, mask_white_ref=None, *, config: AreaConfig = AreaConfig()):
    """
    混合方法：先尝试K-means，失败则使用阈值方法
    """
    print("尝试增强的K-means聚类...")
    mask_red_kmeans, mask_white_kmeans, _, _ = enhanced_color_clustering(bgr)

    # 检查K-means结果是否合理
    red_ratio = np.sum(mask_red_kmeans > 0) / (bgr.shape[0] * bgr.shape[1])
    white_ratio = np.sum(mask_white_kmeans > 0) / (bgr.shape[0] * bgr.shape[1])

    print(f"K-means结果 - 红色占比: {red_ratio:.2%}, 白色占比: {white_ratio:.2%}")

    # 如果有参考图，评估分割质量
    if mask_red_ref is not None and mask_white_ref is not None:
        metrics_kmeans = evaluate_segmentation_quality(mask_red_kmeans, mask_white_kmeans, mask_red_ref, mask_white_ref)
        print(
            f"K-means质量 - 红色IoU: {metrics_kmeans.get('red_iou', 0):.3f}, 白色IoU: {metrics_kmeans.get('white_iou', 0):.3f}")

    # 如果某一类占比过高或过低，说明K-means可能失败了
    if red_ratio < 0.05 or red_ratio > 0.95 or white_ratio < 0.05 or white_ratio > 0.95:
        print("K-means结果不合理，使用HSV阈值方法")
        mask_red, mask_white = threshold_based_red_white_segmentation(bgr)

        # 检查阈值方法的结果
        red_ratio_thresh = np.sum(mask_red > 0) / (bgr.shape[0] * bgr.shape[1])
        white_ratio_thresh = np.sum(mask_white > 0) / (bgr.shape[0] * bgr.shape[1])
        print(f"阈值方法结果 - 红色占比: {red_ratio_thresh:.2%}, 白色占比: {white_ratio_thresh:.2%}")

        # 如果有参考图，评估阈值方法的质量
        if mask_red_ref is not None and mask_white_ref is not None:
            metrics_thresh = evaluate_segmentation_quality(mask_red, mask_white, mask_red_ref, mask_white_ref)
            print(
                f"阈值方法质量 - 红色IoU: {metrics_thresh.get('red_iou', 0):.3f}, 白色IoU: {metrics_thresh.get('white_iou', 0):.3f}")

            # 如果K-means质量更好，使用K-means
            if (metrics_kmeans.get('red_iou', 0) + metrics_kmeans.get('white_iou', 0) >
                    metrics_thresh.get('red_iou', 0) + metrics_thresh.get('white_iou', 0)):
                print("K-means质量更好，使用K-means结果")
                mask_red, mask_white = mask_red_kmeans, mask_white_kmeans

        # 如果阈值方法也失败，尝试简单的方法
        if red_ratio_thresh < 0.01 and white_ratio_thresh < 0.01:
            print("阈值方法也失败，使用备选简单方法")
            # 简单方法：基于亮度阈值
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            _, mask_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            mask_red = cv2.bitwise_not(mask_white)
    else:
        mask_red, mask_white = mask_red_kmeans, mask_white_kmeans

    # 应用最小区域尺寸过滤
    if config.min_white_region_size > 0:
        mask_white = filter_small_regions(mask_white, config.min_white_region_size)
        print(f"已过滤掉小于{config.min_white_region_size}像素的白色区域")

    return mask_red, mask_white


def filter_small_regions(mask, min_size):
    """
    过滤掉小于指定尺寸的区域
    """
    # 标记连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 创建新掩膜
    new_mask = np.zeros_like(mask)

    # 保留大于最小尺寸的区域
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            new_mask[labels == i] = 255

    return new_mask


def verify_red_outside_white_inside(mask_red, mask_white):
    """
    验证红色区域是否在白色区域外面
    """
    # 查找白色区域的轮廓
    white_contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not white_contours:
        print("警告：未找到白色区域轮廓")
        return mask_red, mask_white

    # 找到最大的白色区域
    largest_white = max(white_contours, key=cv2.contourArea)

    # 计算白色区域的质心
    M = cv2.moments(largest_white)
    if M["m00"] == 0:
        print("警告：无法计算白色区域质心")
        return mask_red, mask_white

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # 检查红色区域是否包围白色区域
    red_contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_outside = False
    for red_contour in red_contours:
        # 如果白色质心在红色轮廓内部，说明红色在外面
        if cv2.pointPolygonTest(red_contour, (cx, cy), False) >= 0:
            red_outside = True
            break

    if not red_outside:
        print("检测到位置关系：红色在里面，白色在外面，正在交换掩膜...")
        return mask_white, mask_red
    else:
        print("位置关系正确：红色在外面，白色在里面")
        return mask_red, mask_white

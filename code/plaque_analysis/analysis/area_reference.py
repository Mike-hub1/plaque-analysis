"""Reference helpers for the legacy red/white area workflow."""

import os
import cv2
import numpy as np


def load_reference_masks(red_ref_path, white_ref_path, target_shape):
    """
    加载红色和白色参考图，并调整到目标尺寸
    保持颜色信息，根据实际颜色生成掩膜
    """
    mask_red_ref = None
    mask_white_ref = None

    if red_ref_path is not None and os.path.exists(red_ref_path):
        img_red_ref = cv2.imread(red_ref_path, cv2.IMREAD_COLOR)
        if img_red_ref is not None:
            if img_red_ref.shape[:2] != target_shape:
                img_red_ref = cv2.resize(img_red_ref, (target_shape[1], target_shape[0]))

            # 根据颜色提取红色区域
            # 方法1: 如果参考图是纯红色标记，直接提取红色通道
            # 方法2: 使用HSV颜色空间检测红色
            hsv_ref = cv2.cvtColor(img_red_ref, cv2.COLOR_BGR2HSV)

            # 检测红色（HSV空间中的两个红色区间）
            lower_red1 = np.array([0, 30, 30])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 30, 30])
            upper_red2 = np.array([180, 255, 255])

            mask_red1 = cv2.inRange(hsv_ref, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv_ref, lower_red2, upper_red2)
            mask_red_ref = cv2.bitwise_or(mask_red1, mask_red2)

            # 如果检测到的红色区域太少，尝试使用亮度阈值
            red_pixels = np.sum(mask_red_ref > 0)
            total_pixels = mask_red_ref.size
            if red_pixels < total_pixels * 0.01:  # 红色区域少于1%
                # 使用灰度阈值作为备选方案
                gray_ref = cv2.cvtColor(img_red_ref, cv2.COLOR_BGR2GRAY)
                _, mask_red_ref = cv2.threshold(gray_ref, 50, 255, cv2.THRESH_BINARY)
                print(f"警告: 红色参考图中红色像素太少，使用灰度阈值")

            print(
                f"已加载红色参考图: {mask_red_ref.shape}, 红色像素={np.sum(mask_red_ref > 0)} ({np.sum(mask_red_ref > 0) / total_pixels:.2%})")
        else:
            print("警告: 无法加载红色参考图")

    if white_ref_path is not None and os.path.exists(white_ref_path):
        img_white_ref = cv2.imread(white_ref_path, cv2.IMREAD_COLOR)
        if img_white_ref is not None:
            if img_white_ref.shape[:2] != target_shape:
                img_white_ref = cv2.resize(img_white_ref, (target_shape[1], target_shape[0]))

            # 根据颜色提取白色区域
            # 白色区域特征: 低饱和度、高亮度
            hsv_ref = cv2.cvtColor(img_white_ref, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 50, 255])
            mask_white_ref = cv2.inRange(hsv_ref, lower_white, upper_white)

            # 如果检测到的白色区域太少，尝试使用亮度阈值
            white_pixels = np.sum(mask_white_ref > 0)
            total_pixels = mask_white_ref.size
            if white_pixels < total_pixels * 0.01:  # 白色区域少于1%
                # 使用灰度阈值作为备选方案
                gray_ref = cv2.cvtColor(img_white_ref, cv2.COLOR_BGR2GRAY)
                _, mask_white_ref = cv2.threshold(gray_ref, 200, 255, cv2.THRESH_BINARY)
                print(f"警告: 白色参考图中白色像素太少，使用灰度阈值")

            print(
                f"已加载白色参考图: {mask_white_ref.shape}, 白色像素={np.sum(mask_white_ref > 0)} ({np.sum(mask_white_ref > 0) / total_pixels:.2%})")
        else:
            print("警告: 无法加载白色参考图")

    return mask_red_ref, mask_white_ref


def evaluate_segmentation_quality(mask_red, mask_white, mask_red_ref, mask_white_ref):
    """
    评估分割质量，计算与参考图的相似度
    """
    metrics = {}

    if mask_red_ref is not None:
        # 调试信息
        seg_red_pixels = np.sum(mask_red > 0)
        ref_red_pixels = np.sum(mask_red_ref > 0)
        intersection_red = np.sum((mask_red > 0) & (mask_red_ref > 0))
        union_red = np.sum((mask_red > 0) | (mask_red_ref > 0))

        print(f"红色分割调试信息:")
        print(f"  分割结果红色像素数: {seg_red_pixels}")
        print(f"  参考图红色像素数: {ref_red_pixels}")
        print(f"  交集像素数: {intersection_red}")
        print(f"  并集像素数: {union_red}")

        # 计算红色分割的准确度
        iou_red = intersection_red / union_red if union_red > 0 else 0

        # 计算Dice系数
        dice_red = 2 * intersection_red / (seg_red_pixels + ref_red_pixels) if (
                                                                                       seg_red_pixels + ref_red_pixels) > 0 else 0

        metrics['red_iou'] = iou_red
        metrics['red_dice'] = dice_red
        metrics['red_accuracy'] = np.sum(mask_red == mask_red_ref) / mask_red.size

    if mask_white_ref is not None:
        # 调试信息
        seg_white_pixels = np.sum(mask_white > 0)
        ref_white_pixels = np.sum(mask_white_ref > 0)
        intersection_white = np.sum((mask_white > 0) & (mask_white_ref > 0))
        union_white = np.sum((mask_white > 0) | (mask_white_ref > 0))

        print(f"白色分割调试信息:")
        print(f"  分割结果白色像素数: {seg_white_pixels}")
        print(f"  参考图白色像素数: {ref_white_pixels}")
        print(f"  交集像素数: {intersection_white}")
        print(f"  并集像素数: {union_white}")

        # 计算白色分割的准确度
        iou_white = intersection_white / union_white if union_white > 0 else 0

        # 计算Dice系数
        dice_white = 2 * intersection_white / (seg_white_pixels + ref_white_pixels) if (
                                                                                               seg_white_pixels + ref_white_pixels) > 0 else 0

        metrics['white_iou'] = iou_white
        metrics['white_dice'] = dice_white
        metrics['white_accuracy'] = np.sum(mask_white == mask_white_ref) / mask_white.size

    return metrics


def adjust_segmentation_with_reference(bgr, mask_red, mask_white, mask_red_ref, mask_white_ref):
    """
    基于参考图微调分割结果
    """
    H, W = bgr.shape[:2]

    # 如果参考图质量足够高，直接使用参考图
    if mask_red_ref is not None and mask_white_ref is not None:
        red_ref_quality = np.sum(mask_red_ref > 0) / (H * W)
        white_ref_quality = np.sum(mask_white_ref > 0) / (H * W)

        # 如果参考图覆盖了合理的区域比例，直接使用
        if 0.05 < red_ref_quality < 0.95 and 0.05 < white_ref_quality < 0.95:
            print("使用参考图作为分割结果")
            return mask_red_ref, mask_white_ref

    # 否则，基于参考图微调现有分割
    if mask_red_ref is not None:
        # 使用参考图指导红色分割的形态学操作
        kernel_size = max(1, min(5, int(W / 100)))  # 根据图像大小自适应
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # 计算当前分割与参考图的差异
        red_diff = cv2.bitwise_xor(mask_red, mask_red_ref)

        # 如果差异较大，尝试调整
        if np.sum(red_diff > 0) > 0.1 * np.sum(mask_red_ref > 0):
            # 尝试开运算去除噪声
            mask_red_cleaned = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

            # 如果清理后与参考图更接近，则使用清理后的版本
            new_red_diff = cv2.bitwise_xor(mask_red_cleaned, mask_red_ref)
            if np.sum(new_red_diff > 0) < np.sum(red_diff > 0):
                mask_red = mask_red_cleaned
                print("已根据参考图清理红色分割")

    if mask_white_ref is not None:
        # 类似地处理白色分割
        kernel_size = max(1, min(5, int(W / 100)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        white_diff = cv2.bitwise_xor(mask_white, mask_white_ref)

        if np.sum(white_diff > 0) > 0.1 * np.sum(mask_white_ref > 0):
            mask_white_cleaned = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)

            new_white_diff = cv2.bitwise_xor(mask_white_cleaned, mask_white_ref)
            if np.sum(new_white_diff > 0) < np.sum(white_diff > 0):
                mask_white = mask_white_cleaned
                print("已根据参考图清理白色分割")

    return mask_red, mask_white

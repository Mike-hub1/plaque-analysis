"""Interactive helpers for the legacy red/white area workflow."""

import cv2
import numpy as np
from plaque_analysis.settings.area import AreaConfig


def render_region_preview(bgr, merged_labels, mask_red, region_id, region_size, current_region_idx, region_count, dilation_size, erosion_size, range_multiplier):
    # 获取区域掩膜
    region_mask = (merged_labels == region_id).astype(np.uint8)

    # 计算搜索圈
    area_scale = min(2.0, max(0.5, np.sqrt(region_size) / 50))
    dynamic_dilation_size = int(dilation_size * range_multiplier * area_scale)
    dynamic_erosion_size = int(erosion_size * area_scale)

    # 创建搜索圈
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (dynamic_dilation_size * 2 + 1, dynamic_dilation_size * 2 + 1))
    dilated = cv2.dilate(region_mask, kernel_dilate)

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (dynamic_erosion_size * 2 + 1, dynamic_erosion_size * 2 + 1))
    eroded = cv2.erode(region_mask, kernel_erode)

    ring_mask = (dilated > 0) & (eroded == 0)

    # 计算搜索圈内的红色区域
    red_in_ring = (mask_red > 0) & ring_mask
    red_area = np.sum(red_in_ring)

    # 创建可视化图像
    vis_img = bgr.copy()

    # 绘制白色区域（蓝色半透明）
    overlay = vis_img.copy()
    overlay[region_mask > 0] = [255, 200, 100]  # BGR: 浅蓝色
    vis_img = cv2.addWeighted(vis_img, 0.5, overlay, 0.5, 0)

    # 绘制搜索圈（绿色半透明）
    overlay2 = vis_img.copy()
    overlay2[ring_mask] = [0, 255, 0]  # BGR: 绿色
    vis_img = cv2.addWeighted(vis_img, 0.7, overlay2, 0.3, 0)

    # 绘制红色区域（红色）
    vis_img[red_in_ring] = [0, 0, 255]

    # 绘制区域轮廓
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_img, contours, -1, (255, 255, 0), 2)

    # 绘制搜索圈轮廓
    ring_contours_outer, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_img, ring_contours_outer, -1, (0, 255, 0), 2)

    # 添加信息文本
    info_y = 30
    cv2.putText(vis_img, f"Region {current_region_idx + 1}/{region_count} (ID: {region_id})",
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    info_y += 40
    cv2.putText(vis_img, f"White Area: {region_size} pixels",
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    info_y += 35
    cv2.putText(vis_img, f"Search Ring: {np.sum(ring_mask)} pixels",
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    info_y += 35
    cv2.putText(vis_img, f"Red Area: {red_area} pixels",
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    info_y += 35
    cv2.putText(vis_img, f"Dilation: {dynamic_dilation_size}, Erosion: {dynamic_erosion_size}",
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    info_y += 50
    cv2.putText(vis_img, "n=Next | p=Prev | s=Save | q=Finish",
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return vis_img


def render_control_panel(current_region_idx, region_count, dilation_size, erosion_size, range_multiplier):
    # 创建控制面板显示
    control_img = np.ones((300, 500, 3), dtype=np.uint8) * 50
    y_pos = 40
    cv2.putText(control_img, f"Region {current_region_idx + 1}/{region_count}",
                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 60
    cv2.putText(control_img, f"Dilation Size: {dilation_size}",
                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    y_pos += 40
    cv2.putText(control_img, f"Erosion Size: {erosion_size}",
                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    y_pos += 40
    cv2.putText(control_img, f"Range Multiplier: {range_multiplier:.1f}",
                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    y_pos += 60
    cv2.putText(control_img, "Keyboard Controls:",
                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    y_pos += 30
    cv2.putText(control_img, "N - Next  P - Previous",
                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_pos += 25
    cv2.putText(control_img, "S - Save  Q - Finish All",
                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return control_img


def interactive_search_ring_verification(bgr, merged_labels, mask_red, sorted_regions, top_n=10, *, config: AreaConfig = AreaConfig()):
    """
    交互式验证和调整每个区域的搜索圈大小
    返回每个区域的自定义搜索圈参数
    """
    print("\n" + "=" * 60)
    print("Interactive Search Ring Verification")
    print("=" * 60)
    print("Instructions:")
    print("- Use sliders to adjust search ring size")
    print("- 'Dilation' controls outer expansion range")
    print("- 'Erosion' controls inner shrinkage range")
    print("- Press 'n' to go to next region")
    print("- Press 'p' to go to previous region")
    print("- Press 's' to save current settings")
    print("- Press 'q' to finish all adjustments")
    print("=" * 60)

    # 只处理前N大区域
    regions_to_verify = sorted_regions[:top_n]

    # 存储每个区域的自定义参数
    custom_params = {}

    current_region_idx = 0

    # 创建窗口
    window_name = "Search Ring Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)

    # 创建控制面板窗口
    control_window = "Control Panel"
    cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(control_window, 500, 300)

    # 滑块回调函数（不需要做任何事，只是更新值）
    def nothing(x):
        pass

    # 初始化滑块
    cv2.createTrackbar("Dilation", control_window, config.dilation_size, 100, nothing)
    cv2.createTrackbar("Erosion", control_window, config.inner_erosion, 50, nothing)
    cv2.createTrackbar("Range x10", control_window, int(config.search_range_multiplier * 10), 50, nothing)

    while True:
        if current_region_idx >= len(regions_to_verify):
            print("所有区域已检查完毕")
            break

        region_id, region_size = regions_to_verify[current_region_idx]

        # 获取当前滑块值
        dilation_size = cv2.getTrackbarPos("Dilation", control_window)
        erosion_size = cv2.getTrackbarPos("Erosion", control_window)
        range_multiplier = cv2.getTrackbarPos("Range x10", control_window) / 10.0

        # 确保最小值
        dilation_size = max(1, dilation_size)
        erosion_size = max(1, erosion_size)
        range_multiplier = max(0.1, range_multiplier)

        vis_img = render_region_preview(
            bgr, merged_labels, mask_red, region_id, region_size,
            current_region_idx, len(regions_to_verify), dilation_size, erosion_size, range_multiplier)

        # 显示图像
        cv2.imshow(window_name, vis_img)

        control_img = render_control_panel(
            current_region_idx, len(regions_to_verify), dilation_size, erosion_size, range_multiplier)

        cv2.imshow(control_window, control_img)

        # 等待按键
        key = cv2.waitKey(50) & 0xFF

        if key == ord('q') or key == 27:  # q 或 ESC
            print("Finished all adjustments")
            # 保存当前参数
            custom_params[region_id] = {
                'dilation_size': dilation_size,
                'erosion_size': erosion_size,
                'range_multiplier': range_multiplier
            }
            break
        elif key == ord('n'):  # 下一个
            # 保存当前参数
            custom_params[region_id] = {
                'dilation_size': dilation_size,
                'erosion_size': erosion_size,
                'range_multiplier': range_multiplier
            }
            print(f"Saved parameters for region {current_region_idx + 1}")
            current_region_idx += 1
        elif key == ord('p'):  # 上一个
            if current_region_idx > 0:
                # 保存当前参数
                custom_params[region_id] = {
                    'dilation_size': dilation_size,
                    'erosion_size': erosion_size,
                    'range_multiplier': range_multiplier
                }
                current_region_idx -= 1
                # 恢复上一个区域的参数
                prev_region_id = regions_to_verify[current_region_idx][0]
                if prev_region_id in custom_params:
                    params = custom_params[prev_region_id]
                    cv2.setTrackbarPos("Dilation", control_window, params['dilation_size'])
                    cv2.setTrackbarPos("Erosion", control_window, params['erosion_size'])
                    cv2.setTrackbarPos("Range x10", control_window, int(params['range_multiplier'] * 10))
        elif key == ord('s'):  # 保存
            custom_params[region_id] = {
                'dilation_size': dilation_size,
                'erosion_size': erosion_size,
                'range_multiplier': range_multiplier
            }
            print(
                f"Saved region {current_region_idx + 1} params: Dilation={dilation_size}, Erosion={erosion_size}, Multiplier={range_multiplier:.1f}")

    cv2.destroyAllWindows()

    # 为未设置的区域使用默认参数
    for region_id, _ in regions_to_verify:
        if region_id not in custom_params:
            custom_params[region_id] = {
                'dilation_size': config.dilation_size,
                'erosion_size': config.inner_erosion,
                'range_multiplier': config.search_range_multiplier
            }

    print(f"\nVerification complete! Adjusted parameters for {len(custom_params)} regions")
    return custom_params

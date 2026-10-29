"""Boundaries helpers for the legacy red/white area workflow."""

import cv2
import math
import numpy as np


def get_red_white_boundary(mask_red, mask_white):
    """
    专门提取红白边界
    """
    # 尝试不同的结构元素大小和迭代次数
    kernel_sizes = [(3, 3), (5, 5), (7, 7)]
    iterations_list = [1, 2, 3]

    best_boundary = None
    max_pixels = 0

    for kernel_size in kernel_sizes:
        for iterations in iterations_list:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

            # 对红色区域进行膨胀，对白色区域进行腐蚀
            red_dilated = cv2.dilate(mask_red, kernel, iterations=iterations)
            white_eroded = cv2.erode(mask_white, kernel, iterations=iterations)

            # 红白边界 = 膨胀的红色 AND 腐蚀的白色
            boundary = cv2.bitwise_and(red_dilated, white_eroded)

            # 清理小的噪声
            boundary = cv2.morphologyEx(boundary, cv2.MORPH_OPEN, kernel)

            pixel_count = np.sum(boundary > 0)
            print(f"  边界提取 - kernel={kernel_size}, iter={iterations}, 边界像素数={pixel_count}")

            if pixel_count > max_pixels:
                max_pixels = pixel_count
                best_boundary = boundary

    if best_boundary is None or max_pixels == 0:
        print("警告：无法提取有效边界，使用备选方案")
        # 备选方案：使用简单的形态学梯度
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        best_boundary = cv2.morphologyEx(mask_red, cv2.MORPH_GRADIENT, kernel)

    return best_boundary


def auto_canny(gray, sigma=0.33):
    """
    用中位数自适应阈值：lower=(1-sigma)*median，upper=(1+sigma)*median
    """
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


def subpixel_polylines(edge_binary, grad_mag, gx, gy):
    """
    输入：
      edge_binary: Canny 的二值边缘(0/255)，建议先与红白边界相交以弱化无关边
      grad_mag:    梯度幅值 float32
      gx, gy:      Sobel 梯度分量 float32
    输出：
      list[np.ndarray(N,2)]：每条折线的亚像素点(x,y)，浮点坐标
    """
    H, W = edge_binary.shape
    contours, _ = cv2.findContours(edge_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    def bilinear(img, xx, yy):
        # 双线性采样（边界 clamp）
        if xx < 0: xx = 0
        if yy < 0: yy = 0
        if xx > W - 1.0001: xx = W - 1.0001
        if yy > H - 1.0001: yy = H - 1.0001
        x0, y0 = int(math.floor(xx)), int(math.floor(yy))
        dx, dy = xx - x0, yy - y0
        v00 = img[y0, x0]
        v01 = img[y0, x0 + 1] if x0 + 1 < W else img[y0, x0]
        v10 = img[y0 + 1, x0] if y0 + 1 < H else img[y0, x0]
        v11 = img[y0 + 1, x0 + 1] if (x0 + 1 < W and y0 + 1 < H) else img[y0, x0]
        return (v00 * (1 - dx) * (1 - dy) +
                v01 * dx * (1 - dy) +
                v10 * (1 - dx) * dy +
                v11 * dx * dy)

    polylines = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2)  # (x,y) int
        poly = []
        for x_i, y_i in pts:
            x, y = float(x_i), float(y_i)

            # 梯度方向（法线）
            gx0 = float(gx[int(y), int(x)])
            gy0 = float(gy[int(y), int(x)])
            norm = math.hypot(gx0, gy0)
            if norm < 1e-6:
                poly.append([x, y])
                continue
            nx, ny = gx0 / norm, gy0 / norm

            # 在 t = -1, 0, +1 像素处对 |∇I| 取样（沿法线方向），做二次拟合
            f_m1 = float(bilinear(grad_mag, x + nx * (-1.0), y + ny * (-1.0)))
            f_0 = float(bilinear(grad_mag, x, y))
            f_p1 = float(bilinear(grad_mag, x + nx * (+1.0), y + ny * (+1.0)))

            # 拟合 f(t)=a t^2 + b t + c，三点解：a=0.5*(f1+fm1-2f0)，b=0.5*(f1-fm1)
            a = 0.5 * (f_p1 + f_m1 - 2.0 * f_0)
            b = 0.5 * (f_p1 - f_m1)

            # 极值 t* = -b / (2a)，限制在 [-1,1]（避免外推）
            if abs(a) > 1e-6:
                t_star = -b / (2.0 * a)
                t_star = max(-1.0, min(1.0, t_star))
            else:
                t_star = 0.0

            subx = x + nx * t_star
            suby = y + ny * t_star
            poly.append([subx, suby])

        if len(poly) > 0:
            polylines.append(np.asarray(poly, dtype=np.float32))
    return polylines


def draw_overlay(bgr, polylines, color=(0, 255, 0)):
    vis = bgr.copy()
    for pl in polylines:
        for (x, y) in pl.astype(int):
            cv2.circle(vis, (x, y), 1, color, -1)
    return vis


def save_svg(polylines, svg_path, stroke_color="lime", stroke_width=1.0, *, image_shape):
    """Write boundary geometry with explicit image size and no global image."""
    import xml.etree.ElementTree as ET
    height, width = image_shape[:2]
    root = ET.Element("svg", {"xmlns": "http://www.w3.org/2000/svg", "version": "1.1",
                              "width": str(width), "height": str(height),
                              "viewBox": f"0 0 {width} {height}"})
    for line in polylines:
        style = {"stroke": stroke_color, "fill": "none"}
        if len(line) >= 2:
            ET.SubElement(root, "polyline", {**style, "stroke-width": str(stroke_width),
                          "points": " ".join(f"{float(x)},{float(y)}" for x, y in line)})
        elif len(line) == 1:
            ET.SubElement(root, "circle", {**style, "cx": str(float(line[0][0])),
                          "cy": str(float(line[0][1])), "r": "0.5"})
    ET.ElementTree(root).write(svg_path, encoding="utf-8", xml_declaration=True)

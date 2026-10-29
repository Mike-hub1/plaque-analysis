"""Stable CSV schema retained from the original area measurement script."""

import csv


def export_measurements(rows, path, config):
    scale = config.pixel_size_x * config.pixel_size_y
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["编号", "AD斑块面积(像素)", "AD斑块面积(um²)", "营养不良面积(像素)", "营养不良面积(um²)"])
        for index, row in enumerate(rows, 1):
            writer.writerow([index, row["size"], f'{row["size"] * scale:.2f}',
                             row["red_area"], f'{row["red_area"] * scale:.2f}'])
        white = sum(row["size"] for row in rows)
        red = sum(row["red_area"] for row in rows)
        writer.writerow([])
        writer.writerow(["总计", white, f"{white * scale:.2f}", red, f"{red * scale:.2f}"])

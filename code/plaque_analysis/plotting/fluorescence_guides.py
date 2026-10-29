"""Figure 3: plot guides."""
from __future__ import annotations

from plaque_analysis.settings.fluorescence import FIGURE_FONT
from plaque_analysis.plotting.fluorescence_layout import FIGURE_BUNDLES, figure_bundle_base
from plaque_analysis.plotting.fluorescence_style import RAW_CHANNEL_FIGURE_FOLDER
import numpy as np


PANEL_EXPLANATIONS_ZH = {
    "boundary_phenotyping": {
        "a": "显示 Aβ 原始探测器强度，线性灰度映射，不作插值平滑。色条单位为探测器相对强度，比例尺来自物理像素标定。金色轮廓标记达到有效位深上限的像素；这些像素的真实强度差异已经无法从当前文件恢复。",
        "b": "显示 LAMP1 原始探测器信号，读取方式、坐标和比例尺与 a 相同。两个通道的探测器强度不是同一蛋白浓度标尺，因此不能由两张图的亮暗直接比较 Aβ 与 LAMP1 的绝对含量。",
        "c": "蓝色为 Aβ 阳性掩膜、陶红色为 LAMP1 阳性掩膜、紫色为二者重叠；虚线为仅以 Aβ 几何边界定义的固定分析窗口。图中可显示窗口以外的 LAMP1，但后续局部读出的统计域限定在窗口内。阈值下阴性只表示当前分割规则未检出，不等于生物学绝对缺失。",
        "d": "将几何形态与距离场合在同一物理坐标系：像素中心到 Aβ 单元外露边界的有符号距离为底图，负值在内、正值在外；黑色为真实单元边界，金色为栅格掩膜凸包，黑色虚线为主轴，白点为几何质心。颜色不是荧光强度，几何主轴不是分子取向，不因绘图放大产生亚像素结构证据。",
        "e": "上部为每个环带的 LAMP1 阳性面积密度，即阳性像素数除以已观测环带像素数，标注的 n 是分母像素数；彩色点表示完整环带，灰色虚线为已观测部分。下部保留窗口内全部像素，陶红色为 LAMP1 阳性、灰色为其余像素；黑点为中位数，黑线为第 25–75 百分位区间，紫色空心方块为均值。蓝色浅底表示 Aβ 内部。四分位距描述像素离散程度，不是动物间变异或置信区间。",
        "f": "按边界外环带和角扇区交叉划分区域，热图数值为各区域的 LAMP1 阳性面积密度。角度以向右为 0°、沿图像顺时针增加。灰色叉号表示该单元不完整或观测像素少于 3 个，不能将其解释为密度为零。该图用于辨别径向平均可能掩盖的方向性分布。",
        "g": "显示窗口内 LAMP1 阳性掩膜的八连通信号斑片，色调编码斑片面积的十进制对数。图像保留全部斑片，仅对面积最大的至多四个标注编号；蓝色轮廓为 Aβ，虚线为窗口。这里的斑片是图像连通域，不代表已经分割确认的单个溶酶体或轴突球状体，且可能被固定窗口裁开。",
        "h": "窗口内每个像素以 Aβ 校正强度为横轴、LAMP1 校正强度为纵轴；蓝色、陶红色、紫色、灰色分别表示仅 Aβ 阳性、仅 LAMP1 阳性、双阳性和双阴性。金色虚线为可表示成单一校正值的有效强度上限。该图可观察强度同步程度与饱和堆积，但受饱和影响时只能用于诊断展示。",
        "i": "逐类汇总当前可输出数值的字段数与总字段数。深色为可计算的表型字段，灰色为窗口质控或已观测量，金色为未满足测量条件的字段。条旁分数表示可计算数/总数，不表示分类准确率；金色长度也不是该特征的生物学效应为零。",
    },
    "abeta_organization": {
        "a": "将几何形态和边界距离重绘于同一坐标系：蓝白陶红底图为有符号距离，黑色为原生像素单元外露边界，金色为栅格掩膜凸包，黑色虚线为像素单元二阶矩的主轴，白点为几何质心。不叠加椭圆以避免遮蔽距离场；这些几何主轴不是 Aβ 分子取向。",
        "b": "在 Aβ 掩膜内部显示到边界的内向距离。金色轮廓指出采用前景八连通、背景四连通约定识别的掩膜孔隙。这些孔隙受分割和像素采样影响，不能命名为分子孔道；深度只反映二维掩膜的几何内部。",
        "c": "经验累积分布逐一纳入 Aβ 掩膜像素的背景校正强度。纵轴为不超过当前强度的像素比例，不是概率模型拟合。上限处的集中跃升提示截顶像素；仅在背景校正后上限为同一常数时画出上限线。受截顶影响的强度分布仍作为诊断图显示，但不恢复成正式可用特征。",
        "d": "内部梯度只使用中心及上下左右四邻居均属于 Aβ 掩膜的中心差分模板，单位为相对强度/微米。蓝色为掩膜边界，灰色为不满足模板条件的位置。此处理避免将掩膜边缘与外部背景的跳变错误当作内部纹理，但不能修复核心强度截顶。",
        "e": "展示灰度共生矩阵的对比度、同质性和熵；符号及蓝色深浅区分四个位移方向。每个方向使用 1、2、3 个原生像素的位移，横轴使用实际物理距离，因此对角方向不能与同像素步数的水平/垂直方向视为同一尺度。该面板三个小坐标系统合起来构成子图 e，单独导出时一并保留。",
        "f": "比较凸实度、包围框填充率、圆度、几何内区保留比例和掩膜孔隙面积比例等无量纲几何量。所有指标分别采用自身定义；数值接近并不代表生物学意义相同。孔隙比例分母为填孔后的掩膜面积；圆度使用像素单元外露边界周长，不进行人为截断。",
    },
    "lamp1_spatial_architecture": {
        "a": "在固定窗口内按八连通规则识别 LAMP1 信号斑片。颜色为斑片面积的十进制对数，蓝线为 Aβ 轮廓，虚线为窗口。只标注最大的至多四个斑片编号，但所有斑片均进入图和面积分布；窗口截断和像素连接方式都会影响斑片统计。",
        "b": "将全部内外环带与角向扇区交叉，展示各单元的阳性面积密度。角度为从右向顺时针的图像坐标角。灰色叉号表示不完整或像素少于 3 个的单元，不能补零或插值。与主图 f 仅画外区不同，此处同时展示 Aβ 内部。",
        "c": "每个环带以半小提琴、全部像素点及中位数/四分位距共同展示 LAMP1 强度。蓝色为 Aβ 内部，陶红色为外部；核密度采用 Scott 带宽，取值恒定或支持不足时不画密度曲线。散点抖动只用于减少遮挡，不改变强度数值。密度曲线不是额外样本，像素点也不是动物重复。",
        "d": "极坐标半径为边界外各角扇区的 LAMP1 阳性面积密度。0° 指向右侧，角度沿图像顺时针增加；空心点用于标识不完整扇区。比较的是方向性分布而不是斑块在组织中的绝对解剖朝向，跨图比较前需要一致的图像方向定义。",
        "e": "逐个纳入窗口内八连通斑片面积，绘制无平滑的经验累积分布。横轴为平方微米，纵轴为累计斑片比例；图中 n 表示图像斑片数。斑片很少时曲线呈阶梯状是正确呈现，不应为美观拟造连续分布或扩大样本量。",
        "f": "每条横条对应一个环带的已观测面积，陶红色部分为 LAMP1 阳性面积，灰色为其余面积。条长同时体现分母差异，便于避免把同一阳性比例误认为同一病理总量。若窗口截断，条形仅描述已观测部分，不替代完整固定窗口的主结局。",
    },
    "channel_relationships": {
        "a": "二乘二列联表包含 Aβ 与 LAMP1 掩膜的四种阳性/阴性组合，每格给出像素数和物理面积。横向区分 Aβ 掩膜状态，纵向区分 LAMP1 掩膜状态；颜色与主图一致。该表描述阈值化的空间共现，与强度相关性是不同问题。",
        "b": "散点纳入固定窗口的所有像素，横轴和纵轴分别为两个通道的带符号背景校正强度。边缘直方图采用 24 个分箱，分别显示每个通道的强度频数。灰色、蓝色、陶红色、紫色对应双阴性、仅 Aβ 阳性、仅 LAMP1 阳性和双阳性；金线提示上限。未进行回归拟合，不能据图直接宣称独立预测效能。",
        "c": "比较完整窗口、前景并集、双阳性子集三个不同统计域的 Pearson 相关系数。并集保留至少一个通道阳性的像素，双阳性只保留交集；条件化选择会改变相关系数，三者不能混称同一共定位指标。n 为像素数；线段连接零点与系数值，不是置信区间。空心符号表示被质控保留为诊断而未进入正式数值列。",
        "d": "Dice 为两倍交集除以两个掩膜面积之和，Jaccard 为交集除以并集；两个方向性面积覆盖比例分别以 Aβ 或 LAMP1 阳性面积为分母。Manders 系数按相应通道的非负前景信号加权，因此不是单纯的面积比例。实心点表示可计算结果，空心点表示仅诊断值；各指标必须结合分母、阈值和强度质控解释。",
        "e": "在同一个已观测固定窗口中，分别统计 Aβ 掩膜内外全部像素的 LAMP1 校正强度，包含阈值下和负背景噪声值。经验累积分布的横向位移表示分布差异，不是组间显著性检验，也没有把像素作为独立生物学重复。",
        "f": "对各 Aβ 强度等级的外八邻域，统计 LAMP1 低于阈值状态及各阳性强度等级的组成。等级 1 对应最亮；气泡面积为比例，颜色为该比例减去固定窗口基线比例，旁边 n 为邻域像素分母。不同 Aβ 等级的邻域可能共享像素，因此各行不构成互斥总体；该图不是转移概率、作用力或时间演变模型。强度等级受饱和影响时，该联合关系也只能作为诊断描述。",
    },
    "manuscript_figure": {
        "a": "同一 CZI 平面的原始 Aβ（蓝色）和 LAMP1（红色）信号加色合并。显示范围为零至有效位深上限，不扣背景、不做百分位拉伸或伽马增强；原生浮点通道按记录范围显示。两通道原始值没有因组版改变，比例尺由当前物理标定计算，以图内标注为准。",
        "b": "当前 Aβ 对象掩膜内的背景校正非负信号，按既有 K 均值结果分为五个有序强度等级。等级 1 最亮；横向色键标注各级实际范围。上限像素比例和诊断状态在图中直接标出，不能把截顶部分解释为均一的分子核心。",
        "c": "固定窗口内 LAMP1 阳性掩膜的排序强度分层，与 b 使用相同视野、方向和物理比例尺。两个通道分别分层，不能将相同等级或颜色亮度解释为相同蛋白浓度。",
        "d": "同一 Aβ 掩膜的几何—边界距离合图。距离场内部为负、外部为正；黑色为像素单元外露边界，金色为栅格凸包，黑色虚线和白点分别表示几何主轴与质心。轮廓不插值、不平滑；距离色阶不是荧光强度或病理分级。",
        "e": "完整保留上部环带 LAMP1 阳性面积比例和下部全部像素的背景校正强度分布。n 为各环带已观测像素数，陶红点为阳性像素、灰点为阈值下像素；黑点及棒为中位数和四分位距，紫色空心方块为均值。浅蓝背景为 Aβ 内部，零点虚线为边界。没有用另一张云雨图重复相同径向结果，原云雨图仍留在空间表型主题中。",
        "f": "凸实度、包围框填充率、圆度、几何内区保留比例和孔隙面积比例的正式数值。各自定义和分母不同，不能把五者当成同一量表或按数值大小排列生物学重要性。",
        "g": "距离环带与角扇区交叉单元的 LAMP1 阳性面积比例。此图保留径向汇总中看不到的角向差异，不是 e 的重复；灰色叉号为不完整或少于三个观测像素的单元。角度从图像右侧沿顺时针增加，不代表经统一配准的解剖朝向。",
        "h": "所有窗口像素的双通道强度散点，附 24 分箱边缘直方图。灰、蓝、陶红、紫分别为双阴性、仅 Aβ 阳性、仅 LAMP1 阳性和双阳性。金色虚线为可用单一校正值表示的 Aβ 上限。上限堆积保留，不抽样、不移除极端点，不进行回归或像素层面显著性检验。",
        "i": "Dice、Jaccard 和两种方向性面积覆盖为实心点；Manders 在强度质控不通过时仅以空心诊断点显示。Coverage Aβ→L 的分母为 Aβ 阳性面积，Coverage L→Aβ 的分母为 LAMP1 阳性面积，L 代表 LAMP1。各指标定义不同，重叠不能单独证明共定位、物理接触或严格核心—晕环机制。",
    },
}


def write_chinese_figure_guides(data, raw, saturation, spacing, config, plaque_dir, dpi, *, only_manuscript=False):
    """Generate detailed Chinese explanations; feature codes and standard symbols stay traceable."""
    master = data["master"]
    lookup = master.set_index("feature")
    def val(key, diagnostic=False, digits=4):
        value = lookup.loc[key, "diagnostic_value" if diagnostic else "value"]
        return f"{float(value):.{digits}g}" if np.isfinite(value) else "缺失"
    source_base = f"../../../plot_source_data/{plaque_dir.name}"
    common = [
        "## 研究定位与共同前提", "",
        "本组图用于把单个斑块的荧光图像拆解为几何、强度、边界分布和通道关系等可追溯描述，为检验“相近面积负载下的局部 LAMP1 病理差异”提供候选信息。它本身不证明构型具有面积之外的独立预测增益。",
        f"当前展示 1 个二维斑块对象；原始矩阵为 {raw[0].shape[0]}×{raw[0].shape[1]} 像素，纵横像素间距分别为 {spacing[0]:.6f}、{spacing[1]:.6f} 微米。",
        f"Aβ 边界外窗口半径为 {config.window_radius_um:g} 微米，名义环宽为 {config.ring_width_um:.6g} 微米，角扇区数为 {config.angular_sectors}。实际每个环带上下界以源数据表为准。",
        f"当前窗口有 {len(data['pixel'])} 个已观测像素，其中 {int(data['pixel'].lamp1_positive.sum())} 个为 LAMP1 阳性；这些 n 均是图像采样点数，不是动物数。",
        f"Aβ 掩膜内达到有效强度上限的像素比例为 {saturation[0]['saturated_fraction_mask']:.2%}；LAMP1 为 {saturation[1]['saturated_fraction_mask']:.2%}。超过当前允许上限的强度及纹理主特征保持缺失，不通过显示变换修复。",
        "背景校正后的负噪声保留在强度分布和均值中；分割所用的非负矩阵与定量所用的带符号矩阵分开。阳性掩膜只代表所用阈值与分割规则的结果。",
        "当前没有把像素、连通斑片、重叠邻域或参数扰动当作独立生物学重复；不报告像素层面的显著性星号或动物层面置信区间。", "",
    ]
    anchors = {
        "manuscript_figure": [("Aβ 面积（平方微米）", "abeta_area_um2"), ("LAMP1 面积（平方微米）", "lamp1_area_um2"),
            ("凸实度", "solidity"), ("Dice", "dice"), ("Aβ 被 LAMP1 覆盖比例", "abeta_area_covered_by_lamp1_fraction")],
        "boundary_phenotyping": [("Aβ 面积（平方微米）", "abeta_area_um2"), ("窗口内 LAMP1 面积（平方微米）", "lamp1_area_um2"),
            ("Aβ 内部 LAMP1 面积密度", "lamp1_inside_density"), ("Aβ 外部 LAMP1 面积密度", "lamp1_outside_density"), ("已观测窗口比例", "observable_window_fraction")],
        "abeta_organization": [("二阶矩长轴（微米）", "major_axis_length_um"), ("二阶矩短轴（微米）", "minor_axis_length_um"),
            ("凸实度", "solidity"), ("掩膜孔隙数", "mask_hole_count_4"), ("最大内向深度（微米）", "max_inward_distance_um")],
        "lamp1_spatial_architecture": [("八连通信号斑片数", "lamp1_window_patch_count_8"), ("最大斑片占阳性面积比例", "lamp1_largest_patch_area_fraction"),
            ("外区阳性像素距离中位数（微米）", "lamp1_outside_d50_um"), ("归一化角向熵", "lamp1_angular_entropy_norm"), ("近边界减远边界面积密度", "lamp1_near_minus_far_density")],
        "channel_relationships": [("Dice 系数", "dice"), ("Jaccard 系数", "jaccard"), ("Aβ 面积被 LAMP1 覆盖的比例", "abeta_area_covered_by_lamp1_fraction"),
            ("LAMP1 面积位于 Aβ 内的比例", "lamp1_area_inside_abeta_fraction"), ("LAMP1 质心偏移（微米）", "lamp1_centroid_offset_um")],
    }
    role_text = {
        "X_A": "本角色仅使用 Aβ 通道，属于后续模型的候选输入。可计算不等于稳定、非冗余或具有独立预测效能；仍须在训练动物内完成标准化、共线性处理和选择。",
        "Y_L": "本角色主要为 LAMP1 局部病理读出，同时包含明确分开的窗口质控/已观测量。不能把同一结局或其等价变换再作为预测该结局的输入。",
        "Z_AL": "本角色同时使用 Aβ 和 LAMP1，属于联合描述。不能混入声称只依赖 Aβ 构型来预测 LAMP1 病理的主输入矩阵，否则存在信息泄漏。",
    }
    for key, spec in FIGURE_BUNDLES.items():
        if only_manuscript and key != "manuscript_figure":
            continue
        base = figure_bundle_base(plaque_dir, key)
        lines = [f"# {spec['title']}：详细中文说明", "", f"[查看{'组合图' if spec['panels'] else '单图'} PNG]({base.name}.png) · [打开可编辑 SVG]({base.name}.svg)", ""]
        lines += common
        if spec["panels"]:
            lines += ["## 子图逐项解读", ""]
            for letter, name in spec["panels"].items():
                lines += [f"### {letter}｜{name}", "", PANEL_EXPLANATIONS_ZH[key][letter], "",
                    f"[独立 PNG](子图/{letter}_{name}.png) · [独立 SVG](子图/{letter}_{name}.svg)", ""]
            lines += ["## 当前对象的数值对照", "", "以下来自正式数值列，不以诊断值替代缺失。数值仅用于核对本图，不代表多动物总体结论。", "",
                      "| 指标 | 当前数值 |", "| --- | --- |"]
            lines += [f"| {label} | {val(feature)} |" for label, feature in anchors[key]]
            if key == "channel_relationships":
                lines += ["", f"三个 Pearson 系数的诊断值依次为：完整窗口 {val('pearson_window', True)}、前景并集 {val('pearson_foreground_union', True)}、双阳性子集 {val('pearson_double_positive_conditional', True)}。这些诊断值不会在正式数值缺失时被偷偷代入模型。"]
        elif key.startswith("feature_atlas_"):
            role = key.removeprefix("feature_atlas_")
            subset = master[master.role == role]
            lines += ["## 读图顺序与用途", "", role_text[role], "",
                f"本图完整列出该角色的 {len(subset)} 个字段，其中当前有数值 {int(subset.value.notna().sum())} 项、正式数值缺失 {int(subset.value.isna().sum())} 项。此图是一个整体的特征—扰动矩阵，没有独立的字母编号子图，因此不重复拆分。", "",
                "从左至右依次为：字段代码、当前基线值、原单位和各单因素扰动下的变化。原代码保留便于与 CSV 逐列匹配，下表提供中文名称。不同单位的基线值不能跨行比较大小，颜色也不是特征重要性。", "",
                "彩色实心点表示可计算表型，灰色实心点表示窗口质控或已观测量，金色空心点表示基线未满足测量条件。图中 NA 表示缺失，而不是零。", "",
                "敏感性单元计算为 100×（扰动值−基线值）/|基线值|。色标在 ±30% 处饱和，但单元格文字显示实际变化；基线为零且未变时标“=”，基线为零而发生变化时显示绝对变化量“Δ”，不伪造百分比。基线或扰动值失效时仍保留该行和缺失状态。", "",
                "各列分别对应 Aβ 阈值、LAMP1 阈值、窗口半径、环宽及 LAMP1 水平方向配准偏移的单因素扰动；具体系数或像素位移标在列头。它们是同一图像的算法敏感性，不是重复成像，也不能据此报告组内相关系数或动物间稳定性。", "",
                "## 完整字段中文对照", "", "诊断列仅保留用于排查测量失效的试算值；状态为不可用时，不应将诊断值投入主模型。", "",
                "| 中文名称 | 对应代码 | 正式数值 | 诊断值 | 单位 | 状态 |", "| --- | --- | --- | --- | --- | --- |"]
            unit_names = {"1": "无量纲", "count": "计数", "boolean": "布尔值", "detector a.u.": "探测器相对强度",
                "detector a.u.·pixel": "相对强度×像素", "detector a.u.·μm²": "相对强度×平方微米", "count/(1000 μm²)": "每千平方微米计数", "bit": "比特", "level²": "量化等级平方"}
            for row in subset.itertuples():
                state = "正式值缺失" if not np.isfinite(row.value) else "已观测量或窗口质控" if row.measurement_kind != "phenotype" else "可计算，待独立验证"
                lines.append(f"| {row.feature_name_zh} | {row.feature} | {val(row.feature)} | {val(row.feature, True)} | {unit_names.get(str(row.units), row.units)} | {state} |")
        else:
            ci = 0 if key.startswith("blue") else 1
            channel = "Aβ" if ci == 0 else "LAMP1"
            lines += ["## 分层图应如何理解", "",
                f"本图仅在{'当前 Aβ 对象掩膜' if ci == 0 else '固定窗口内的 LAMP1 阳性掩膜'}内对背景校正后的非负强度进行 K 均值聚类。聚类编号按强度重新排序，等级 1 为最亮，色条列出每一级实际覆盖的强度范围。", "",
                "深色背景为未分配区域，不是额外的最低强度簇。显示采用原生像素最近邻方式，比例尺依照物理标定；不进行视觉平滑来掩盖像素尺度。此图包含一幅强度图及对应色条，是单一面板，不将色条单独拆成子图。", "",
                "强度等级是该对象的相对分层，不是经过生物学验证的病理亚型，也不是跨通道的同一浓度水平。等级 1 的面积不能直接等同于某种分子核心，核心—周缘结构仍需边界距离与空间分布等独立证据。", "",
                f"当前 {channel} 掩膜内上限像素比例为 {saturation[ci]['saturated_fraction_mask']:.2%}。强度截顶时，聚类不能恢复原本的亮度差异；本图仍可用于定位问题，但不能据此把受质控剔除的强度特征恢复成正式值。", "",
                "## 当前强度分层", "", "| 等级 | 最低校正强度 | 最高校正强度 |", "| --- | --- | --- |"]
            for row in data["strata"][ci]:
                lines.append(f"| {row['rank']} | {row['min']:.6g} | {row['max']:.6g} |")
        lines += ["", "## 统计与解释边界", "",
            "1. 四分位距、经验累积分布及核密度均描述当前图像内的分布，不是生物学重复带来的不确定性；本图不包含显著性检验。",
            "2. 当前二维裁剪及像素间距不能支持三维体积、亚像素接触或 Aβ 分子折叠构象的结论；固定窗口完整也不自动代表完整覆盖了全部病理晕环。",
            "3. 空间共现、强度同步、面积覆盖和静态邻域组成回答不同问题，不能用单一指标证明共定位机制、相互排斥或严格核心—晕环机制。",
            "4. 后续验证构型信息的增量价值时，应以动物等独立层级划分训练与验证集，在训练集内拟合标准化、残差化、特征选择及正则化；仍需控制面积、年龄、脑区和批次等背景。", "",
            "## 文件、数据来源与导出规则", "",
            f"图像文件只保留 SVG 和 PNG；另有用户要求的 Illustrator 可编辑组版文件。SVG 的文字可编辑；PNG 按 {dpi} DPI 导出。图内英文与数字使用 {FIGURE_FONT}；正文整合图中的希腊字母及专用符号采用 Symbol，其余核查图保持原字体。解释文档统一中文，保留必要的指标名称与代码。",
            "独立子图从绘图对象重新导出，不是从组合 PNG 截屏裁剪；保留关联坐标、色条、局部图例和数值。v7.4 将几何与边界距离整合为同一画面，并新增正文组版；改变的是图形呈现，不是原始矩阵或测量数值。", "",
            "完整分析数据仍在结果目录，绘图明细移至与 figures 并列的 plot_source_data，以免与成图混放。", "",
            "- [完整特征长表](../../../all_features_complete.csv)：正式值、诊断值、定义、单位与失效原因。",
            "- [分割和测量质控](../../../feature_qc.csv)：逐字段核查不能计算的原因。",
            "- [参数敏感性原表](../../../sensitivity_analysis.csv)：各场景的对象匹配、分裂及数值。",
            f"- [全部绘图像素]({source_base}/pixels.csv)：带符号强度、掩膜类别、距离和角度。",
            f"- [径向汇总]({source_base}/radial_summary.csv) · [径向角向单元]({source_base}/radial_angular.csv) · [信号斑片]({source_base}/lamp1_patches.csv)。",
            "- [方向纹理原表](../../../glcm_directions.csv) · [邻域组成原表](../../../neighborhood_composition.csv)。", ""]
        if key.startswith("feature_atlas_"):
            lines += [f"- [本图敏感性单元的实际显示值]({source_base}/{key}_display_values.csv)。", ""]
        if key == "manuscript_figure":
            lines += ["## 正文版式与 A4 校样", "",
                "正文 SVG/PNG 画布为 183×170 毫米，采用图像板与定量图结合的非等宽三行版式。v7.4.1 将 f 的数据绘图区由 15×33.5 毫米扩大至 32×38 毫米，并重新分配 d、e、f 的水平间距。e 仍完整保留面积比例及全部像素强度读出，没有为版式删减数据。",
                "保留 a–i 的引用编号；删除 Shape ratios 等描述性小标题，具体含义由坐标、通道名及本图图注交代。b、c 图内的 Aβ 和 LAMP1 是必要的通道身份标记，予以保留。Nature 并未统一禁止描述性标题，本次删除是为了减少冗余。",
                "正文编号为 8 pt 粗体，其他文字为 5–7 pt。除显微像素和强度分层矩阵外，散点、曲线、掩膜轮廓、比例尺和文字均保留矢量。没有利用高导出 DPI 伪造更高的显微分辨率。", "",
                "英文和数字为 Arial；β、μ、箭头、角度符号等使用 Symbol。两种字体均保留文字，不转换为轮廓。h 新增金色上限虚线的图例，空心诊断值仍明确区别于正式结果。",
                "[A4 校样 PNG](A4校样.png) · [A4 校样 SVG](A4校样.svg)：210×297 毫米，图形本身仍保持 183×170 毫米，下面附英文校样图注；中文完整解释以本文档为准。",
                "导出格式为 SVG 与 PNG。绘图使用输入观测值，不抽样、不补造像素；缺失值按测量质量控制结果保留。",
                "官方参照：[Nature 尺寸与组版](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/)；[Nature 字体与导出](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)。依据本次查询的指南执行排版，但当前原始图采样粗且存在截顶，不能仅凭版式合规宣称图像已经满足所有正式投稿条件。", ""]
        (base.parent/"说明.md").write_text("\n".join(lines), encoding="utf-8")
    if only_manuscript:
        return
    summary = ["# 当前斑块：图形阅读总览", "", "建议先看边界空间表型，再看 Aβ 自身构型、LAMP1 空间表型及双通道关系，最后用三个完整图谱检查可测量性和参数敏感性。", "",
               f"图内英文保留期刊常用表述及 {FIGURE_FONT} 字体；本目录中的解释文字全部采用中文。", "",
               "| 主题 | 组合图或单图 | 详细解释 | 独立子图数 |", "| --- | --- | --- | --- |"]
    for key, spec in FIGURE_BUNDLES.items():
        base = figure_bundle_base(plaque_dir, key)
        relative = base.relative_to(plaque_dir).as_posix()
        summary.append(f"| {spec['title']} | [PNG]({relative}.png) / [SVG]({relative}.svg) | [中文说明]({spec['folder']}/说明.md) | {len(spec['panels'])} |")
    summary += ["", "共十类图：原有九类分析主题，加一张正文整合图。五张组合图合计三十六张独立子图；其余五张本身为单图。正文组版用于论文叙事，原有主题图用于完整方法及结果核查。", "",
                "几何与面积、强度与纹理、联合关系必须结合各自质控解释；不能将诊断图的可见数值直接当作正式模型输入。", ""]
    (plaque_dir/"阅读总览.md").write_text("\n".join(summary), encoding="utf-8")


def write_figures_index(directory):
    lines = ["# 图形结果索引", "", "所有图像仅保留 SVG（可编辑矢量）与 PNG（默认 600 DPI）两种格式。按斑块编号进入后，每个图形主题都有独立文件夹、中文说明和必要的子图。", ""]
    if (directory/RAW_CHANNEL_FIGURE_FOLDER/"说明.md").is_file():
        lines += [f"- [CZI 原始图：三个单通道、双通道及三通道合并]({RAW_CHANNEL_FIGURE_FOLDER}/说明.md)", ""]
    for plaque in sorted(directory.iterdir()):
        if plaque.is_dir() and (plaque/"阅读总览.md").is_file():
            lines.append(f"- [{plaque.name}：阅读总览]({plaque.name}/阅读总览.md)")
    lines += ["", "## 收纳规则", "", "- 01–04 主题：组合图、说明.md、子图文件夹。", "- 05–09 主题：单图与说明.md。",
              "- 10_正文整合图：183×170 毫米正文图、A4 校样、独立子图及完整中文图注；Illustrator 文件在界面中核查后保存。",
              "- 00_CZI原始图：所选原始平面的五幅通道图与中文说明；不按斑块重复导出完整视野。",
              "- 图形计算用的像素、环带、斑片和敏感性明细单独保留在结果目录的 plot_source_data。",
              "- 原始数据、分析 CSV 和矩阵文件保持独立，不因图形格式精简而删除。", ""]
    (directory/"阅读索引.md").write_text("\n".join(lines), encoding="utf-8")


"""
生成模拟的UI，用于训练模型
"""
import sys
import os
import random
import json
import math
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QRectF, QTimer
from PyQt5.QtGui import QPainter, QColor, QFont, QPainterPath, QPen, QFontMetrics



# ---------- 自动保存的相关配置 ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(PROJECT_ROOT, "datasets", "synthetic")
COUNTER_FILE = os.path.join(PROJECT_ROOT, "counter.txt")
CLASSES = ["RD", "RS", "GD", "GS"]
DATASET_RANDOM_SEED = 2026
AUTO_SAVE_DELAY_MS = int(os.environ.get("MOCK_UI_SAVE_DELAY_MS", "500"))

SPLIT_RANGES = {
    "synthetic_train": range(0, 400),
    "synthetic_val": range(400, 450),
    "synthetic_test": range(450, 500),
}


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_next_counter():
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, "r", encoding="utf-8") as f:
            count = int(f.read().strip())
    else:
        count = 0
    # 写入下一个计数
    with open(COUNTER_FILE, "w", encoding="utf-8") as f:
        f.write(str(count + 1))
    return count


def get_split(counter):
    for split, split_range in SPLIT_RANGES.items():
        if counter in split_range:
            return split
    raise ValueError(f"图片编号超出预定范围：{counter}")


def save_annotation(split, image_index, entities):
    """将实体标签写入与图片同名的 JSON。"""
    annotation_dir = os.path.join(SAVE_DIR, "annotations", split)
    ensure_dir(annotation_dir)
    annotation_path = os.path.join(annotation_dir, f"{image_index}.json")
    data = {"entities": entities}

    temp_path = f"{annotation_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(temp_path, annotation_path)
    return annotation_path

# -------------------------------------
class MockSCADAScreen(QWidget):
    def __init__(self, image_index, auto_save_enabled=True):
        super().__init__()
        self.image_index = image_index
        self.setWindowTitle("工业仿真界面 - 数据集生成器")
        self.resize(1280, 720)
        
        self.COLOR_RED = QColor(220, 50, 50)
        self.COLOR_GREEN = QColor(60, 180, 60)
        self.COLOR_TEXT = QColor(0, 0, 0)
        self.COLOR_BG = QColor(255, 255, 255)
        self.COLOR_TITLE_BG = QColor(0, 80, 140)
        self.label_font = QFont("Arial", 10)  # 编号绘制、避让、标注共用同一字体

        # 若当前随机场景无法按指定角度分布合法放置全部编号，则整体重新生成。
        for _ in range(100):
            self.targets = []
            self.pipes = []
            self.init_random_data() # 生成图元targets
            self.locked_indices = self._pick_two_locks() # 带锁图元的索引
            self.text_labels = self._generate_text_labels() # 每个图元的4至6位编号
            try:
                self.label_boxes = self._place_labels() # 编号的矩形框(tx, ty, tw, th)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError("连续100次仍无法生成可合法放置全部编号的场景")
        self._generate_pipes() # 生成管线

        # 使用定时器延迟保存，确保界面完全绘制
        if auto_save_enabled:
            QTimer.singleShot(AUTO_SAVE_DELAY_MS, self.auto_save)

    def _pick_two_locks(self):
        if len(self.targets) < 2:
            return []
        return random.sample(range(len(self.targets)), 2)

    def _generate_text_labels(self):
        """按 1:1:8 的比例生成4位、5位、6位数字编号。"""
        labels = []
        for _ in self.targets:
            length = random.choices([4, 5, 6], weights=[1, 1, 8], k=1)[0]
            labels.append("".join(random.choices("0123456789", k=length)))
        return labels

    def _place_labels(self):
        """按混合角度分布放置编号标签，并避开图元和其他编号。"""
        placed = []  # 已放置的标签矩形 (tx, ty, tw, th)
        boxes = []
        img_w, img_h = self.width(), self.height()
        font_metrics = QFontMetrics(self.label_font)  # 使用真实字体尺寸计算编号框
        label_padding = 3  # 避让时给编号四周留出安全距离

        # 所有图元的矩形（扩大5像素作为安全间距）
        target_rects = []
        for _, cx, cy, size in self.targets:
            half = size / 2 + 5   # 加5像素缓冲
            target_rects.append((cx - half, cy - half, cx + half, cy + half))

        for idx, (class_id, cx, cy, size) in enumerate(self.targets):
            label = self.text_labels[idx]
            tw = font_metrics.horizontalAdvance(label)  # 使用真实绘制宽度
            th = font_metrics.height()                  # 使用真实绘制高度

            # 50%在四个主方向附近±20°，50%在0至360°内连续随机。
            near_cardinal = random.random() < 0.5
            candidate_angles = []
            if near_cardinal:
                cardinal_angle = random.choice([0.0, 90.0, 180.0, 270.0])
                candidate_angles.append(
                    cardinal_angle + random.uniform(-20.0, 20.0)
                )
                candidate_angles.extend(
                    cardinal_angle + random.uniform(-20.0, 20.0)
                    for _ in range(23)
                )
            else:
                candidate_angles.extend(
                    random.uniform(0.0, 360.0)
                    for _ in range(24)
                )

            # 保持原数据集“编号贴近图元”的观感：文字框与图元外缘约留 10 px。
            # 若当前位置发生碰撞，只允许小幅外移；仍放不下时由外层逻辑重建整张图，
            # 避免为了强行放置而把编号推到离所属图元很远的位置。
            edge_gap = random.uniform(9.0, 13.0)
            candidates = []
            for extra_distance in (0.0, 4.0, 8.0):
                for angle in candidate_angles:
                    radians = math.radians(angle)
                    cosine = abs(math.cos(radians))
                    sine = abs(math.sin(radians))
                    horizontal_clearance = size / 2 + tw / 2
                    vertical_clearance = size / 2 + th / 2
                    horizontal_distance = (
                        horizontal_clearance / cosine
                        if cosine > 1e-6 else float("inf")
                    )
                    vertical_distance = (
                        vertical_clearance / sine
                        if sine > 1e-6 else float("inf")
                    )
                    distance = (
                        min(horizontal_distance, vertical_distance)
                        + edge_gap
                        + extra_distance
                    )
                    label_center_x = cx + math.cos(radians) * distance
                    label_center_y = cy + math.sin(radians) * distance
                    candidates.append(
                        (
                            label_center_x - tw / 2,
                            label_center_y - th / 2,
                        )
                    )

            best = None
            for tx, ty in candidates:
                # 边界检查
                if tx < 10 or tx + tw > img_w - 10:
                    continue
                if ty < 65 or ty + th > img_h - 10:
                    continue

                # 检查与所有图元矩形是否相交
                overlap_with_target = False
                for rx_min, ry_min, rx_max, ry_max in target_rects:
                    # 标签矩形范围，避让判断使用带安全边距的真实编号框
                    lx_min, ly_min = tx - label_padding, ty - label_padding
                    lx_max, ly_max = tx + tw + label_padding, ty + th + label_padding
                    if not (lx_max <= rx_min or lx_min >= rx_max or ly_max <= ry_min or ly_min >= ry_max):
                        overlap_with_target = True
                        break
                if overlap_with_target:
                    continue

                # 检查与已放置标签的重叠
                overlap_with_label = False
                for px, py, pw, ph in placed:
                    # 编号之间也按安全边距避让
                    if not (tx + tw + label_padding <= px - label_padding or
                            tx - label_padding >= px + pw + label_padding or
                            ty + th + label_padding <= py - label_padding or
                            ty - label_padding >= py + ph + label_padding):
                        overlap_with_label = True
                        break
                if not overlap_with_label:
                    best = (tx, ty)
                    break

            if best is None:
                raise RuntimeError(f"无法为第 {idx} 个图元放置编号")

            tx, ty = best
            placed.append((tx, ty, tw, th))
            boxes.append((tx, ty, tw, th))

        return boxes

    def is_overlap(self, cx, cy, size, existing_targets):
        for _, ex_cx, ex_cy, ex_size in existing_targets:
            dist = ((cx - ex_cx) ** 2 + (cy - ex_cy) ** 2) ** 0.5
            if dist < (size + ex_size) / 2 + 5:
                return True
        return False

    def generate_random_pipe(self):
        """生成一条不与任何编号矩形相交的管线，最多尝试 20 次"""
        max_tries = 20
        for _ in range(max_tries):
            pipe_type = random.choice(['horizontal', 'vertical', 'diagonal'])
            w = self.width()
            h = self.height()
            margin = 80
            thickness = random.randint(1, 3)

            if pipe_type == 'horizontal':
                y = random.randint(100, h - margin)
                x1 = random.randint(margin, w // 3)
                x2 = random.randint(w // 2, w - margin)
                x1, y1, x2, y2 = x1, y, x2, y
            elif pipe_type == 'vertical':
                x = random.randint(margin, w - margin)
                y1 = random.randint(100, h // 2)
                y2 = random.randint(h // 2, h - margin)
                x1, y1, x2, y2 = x, y1, x, y2
            else:  # diagonal
                x1 = random.randint(margin, w // 2)
                y1 = random.randint(100, h // 2)
                x2 = random.randint(w // 2, w - margin)
                y2 = random.randint(h // 2, h - margin)

            # 检查是否与任何编号矩形相交，按管线线宽扩展编号矩形，避免编号与管线边缘重合。
            # 修改：按管线线宽和编号安全边距扩展编号矩形，避免编号与管线重合。
            label_padding = 3
            intersect = any(
                self.line_intersects_rect(x1, y1, x2, y2,
                                          rx - thickness - label_padding,
                                          ry - thickness - label_padding,
                                          rw + (thickness + label_padding) * 2,
                                          rh + (thickness + label_padding) * 2)
                for rx, ry, rw, rh in self.label_boxes
            )
            if not intersect:
                return (x1, y1, x2, y2, thickness)
        return None  # 实在生成不了就跳过

    def init_random_data(self):
        # 只生成 targets
        num_targets = random.randint(5, 10)
        attempts = 0
        max_attempts = 500
        while len(self.targets) < num_targets and attempts < max_attempts:
            class_id = random.randint(0, 3)
            cx = random.randint(100, self.width() - 100)
            cy = random.randint(120, self.height() - 80)
            size = random.randint(20, 35)
            if not self.is_overlap(cx, cy, size, self.targets):
                self.targets.append((class_id, cx, cy, size))
            attempts += 1

    def _generate_pipes(self):
        """生成随机管线"""
        num_pipes = random.randint(5, 10)
        self.pipes = []
        for _ in range(num_pipes):
            pipe = self.generate_random_pipe()
            if pipe is not None:
                self.pipes.append(pipe)

    @staticmethod
    def line_intersects_rect(x1, y1, x2, y2, rx, ry, rw, rh):
        """
        判断线段 (x1,y1)-(x2,y2) 是否与矩形 (rx,ry,rw,rh) 相交
        使用 Cohen-Sutherland 编码的简化版：若两端点都在矩形同一侧，则不相交
        """
        # 计算端点的区域码
        def code(x, y):
            c = 0
            if x < rx:          c |= 1  # left
            elif x > rx + rw:   c |= 2  # right
            if y < ry:          c |= 4  # top
            elif y > ry + rh:   c |= 8  # bottom
            return c
        c1 = code(x1, y1)
        c2 = code(x2, y2)
        # 如果按位与不为0，说明在同一侧外部，肯定不相交
        if c1 & c2 != 0:
            return False
        # 否则认为可能相交（保守判断，即使恰好擦边也视为相交，重试即可）
        return True

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. 背景
        painter.fillRect(self.rect(), self.COLOR_BG)

        # 2. 顶部标题栏
        title_rect = QRectF(0, 0, self.width(), 55)
        painter.fillRect(title_rect, self.COLOR_TITLE_BG)
        painter.setPen(self.COLOR_TEXT)
        font = QFont("Arial", 12, QFont.Bold)
        painter.setFont(font)
        painter.drawText(QRectF(20, 10, 700, 24), Qt.AlignLeft | Qt.AlignVCenter, "中国三峡 - 柔性直流输电示范工程")

        # 3. 左侧竖排文字
        painter.save()
        painter.translate(40, self.height() / 2)
        painter.rotate(-90)
        painter.setPen(self.COLOR_TEXT)
        font_left = QFont("Arial", 13)
        painter.setFont(font_left)
        painter.drawText(QRectF(-self.height()/2, -20, self.height(), 40), Qt.AlignCenter, "陆上站交流场")
        painter.restore()

        # 4. 随机管线
        for x1, y1, x2, y2, width in self.pipes:
            pen_pipe = QPen(self.COLOR_TEXT, width)
            painter.setPen(pen_pipe)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # 5. 核心目标
        for class_id, cx, cy, size in self.targets:
            if class_id in [0, 1]:
                color = self.COLOR_RED
            else:
                color = self.COLOR_GREEN
            pen_outline = QPen(color.lighter(150), 2)
            painter.setPen(pen_outline)
            painter.setBrush(color)
            rect = QRectF(cx - size/2, cy - size/2, size, size)
            if class_id % 2 == 0:
                path = QPainterPath()
                path.moveTo(cx, cy - size/2)
                path.lineTo(cx + size/2, cy)
                path.lineTo(cx, cy + size/2)
                path.lineTo(cx - size/2, cy)
                path.closeSubpath()
                painter.drawPath(path)
            else:
                painter.drawRect(rect)

        # 6. 绘制黄色锁
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 0))   # 纯黄色
        for idx in self.locked_indices:
            if idx >= len(self.targets):
                continue
            _, cx, cy, size = self.targets[idx]
            lock_size = size * 0.45
            lock_rect = QRectF(cx - lock_size/2, cy - lock_size/2, lock_size, lock_size)
            painter.drawRect(lock_rect)

        # 7. 绘制图元编号
        painter.setFont(self.label_font)  # 绘制字体与避让/标注计算保持一致
        font_metrics = QFontMetrics(self.label_font)  # 用真实字体基线绘制编号
        painter.setPen(QColor(0, 0, 0))  # 黑色
        for idx, (tx, ty, tw, th) in enumerate(self.label_boxes):
            label = self.text_labels[idx]
            # 绘制文本（注意 drawText(x, y, str) 的 y 是基线，所以用 ty+th 作为基线）
            painter.drawText(int(tx), int(ty + font_metrics.ascent()), label)

    def auto_save(self):
        """自动保存截图和扩展 COCO 实体标签。"""
        # 图片编号在生成随机内容前已经确定，保证每张图使用独立固定种子。
        idx = self.image_index
        split = get_split(idx)
        image_dir = os.path.join(SAVE_DIR, "images", split)
        ensure_dir(image_dir)
        img_path = os.path.join(image_dir, f"{idx}.png")

        # 截图
        pixmap = self.grab()  # 捕获窗口内容
        if not pixmap.save(img_path, "PNG"):
            raise RuntimeError(f"图片保存失败：{img_path}")

        entities = []
        for target_index, (class_id, cx, cy, size) in enumerate(self.targets):
            tx, ty, tw, th = self.label_boxes[target_index]
            entities.append({
                "symbol_class": class_id,
                "symbol_bbox": [
                    round(cx - size / 2, 2),
                    round(cy - size / 2, 2),
                    size,
                    size,
                ],
                "locked": int(target_index in self.locked_indices),
                "text_bbox": [round(tx, 2), round(ty, 2), tw, th],
                "identifier": self.text_labels[target_index],
            })

        annotation_path = save_annotation(split, idx, entities)

        # 输出信息
        if os.environ.get("MOCK_UI_SILENT") != "1":
            print(f"✅ 已保存：{img_path}")
            print(f"✅ 已保存：{annotation_path}")
            print(f"图元数量：{len(self.targets)}，管线数量：{len(self.pipes)}")

        # 关闭窗口（可选，方便连续运行）
        self.close()


app = QApplication(sys.argv)
font = QFont("Microsoft YaHei", 10)
app.setFont(font)

if os.environ.get("MOCK_UI_BATCH") == "1":
    app.setQuitOnLastWindowClosed(False)
    batch_start = int(os.environ.get("MOCK_UI_BATCH_START", "0"))
    batch_end = int(os.environ.get("MOCK_UI_BATCH_END", "500"))
    if batch_start < 0 or batch_end > 500 or batch_start >= batch_end:
        raise ValueError(f"批量生成范围无效：[{batch_start}, {batch_end})")

    for image_index in range(batch_start, batch_end):
        random.seed(DATASET_RANDOM_SEED + image_index)
        window = MockSCADAScreen(image_index, auto_save_enabled=False)
        window.move(-10000, -10000)
        window.show()
        app.processEvents()
        window.auto_save()
        window.deleteLater()
        app.processEvents()
        if (image_index + 1) % 50 == 0:
            print(f"已完成 {image_index + 1}/{batch_end}")
    sys.exit(0)

requested_image_index = os.environ.get("MOCK_UI_IMAGE_INDEX")
if requested_image_index is None:
    image_index = get_next_counter()
else:
    image_index = int(requested_image_index)
    if image_index < 0 or image_index >= 500:
        raise ValueError(f"指定图片编号超出范围：{image_index}")

random.seed(DATASET_RANDOM_SEED + image_index)
window = MockSCADAScreen(image_index)
if os.environ.get("MOCK_UI_HIDDEN") == "1":
    window.move(-10000, -10000)
window.show()
sys.exit(app.exec_())

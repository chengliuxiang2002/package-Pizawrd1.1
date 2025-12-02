import os
from PIL import Image
from typing import List, Tuple


class ImageBlock:
    def __init__(self, img_path: str):
        self.img = Image.open(img_path)
        self.width, self.height = self.img.size
        self.area = self.width * self.height
        self.x = 0  # 放置x坐标
        self.y = 0  # 放置y坐标


class Shelf:
    """表示一行（架子）的类"""

    def __init__(self, y: int, height: int):
        self.y = y  # 架子的y坐标
        self.height = height  # 架子的高度
        self.current_x = 0  # 当前放置位置的x坐标
        self.remaining_height = height  # 剩余高度（用于后续调整）


def bottom_left_layout(blocks: List[ImageBlock], gap: int = 10) -> Tuple[int, int]:
    """
    左下角算法进行图像合并
    gap: 图片之间的间隙（像素）
    """
    if not blocks:
        return 0, 0

    # 按高度降序排序（先放高的图片）
    blocks.sort(key=lambda x: -x.height)

    shelves = []  # 存储所有架子
    canvas_width = 0
    canvas_height = 0

    for block in blocks:
        placed = False

        # 尝试将图片放入现有的架子中
        for shelf in shelves:
            # 检查架子高度是否足够且宽度有余量
            if shelf.height >= block.height and shelf.current_x + block.width <= canvas_width:
                # 可以放入这个架子
                block.x = shelf.current_x
                block.y = shelf.y + (shelf.height - block.height)  # 底部对齐
                shelf.current_x += block.width + gap
                placed = True
                break

        # 如果无法放入现有架子，创建新架子
        if not placed:
            # 新架子放在当前画布的最下方
            new_shelf_y = canvas_height
            new_shelf_height = block.height

            # 创建新架子
            new_shelf = Shelf(new_shelf_y, new_shelf_height)
            new_shelf.current_x = block.width + gap

            # 放置图片
            block.x = 0
            block.y = new_shelf_y

            # 更新画布尺寸
            canvas_width = max(canvas_width, block.width)
            canvas_height += block.height + gap

            shelves.append(new_shelf)
        else:
            # 更新画布宽度（如果当前架子延伸超出了原有宽度）
            canvas_width = max(canvas_width, block.x + block.width)

    return canvas_width, canvas_height - gap  # 减去最后一个间隙


def merge_all_images_bottom_left(image_paths: List[str], output_path: str = "all_merged_bottom_left.png",
                                 gap: int = 10):
    """使用左下角算法合并所有图片"""
    # 过滤无效路径
    valid_paths = [p for p in image_paths if os.path.exists(p)]
    if not valid_paths:
        print("没有有效图片路径")
        return

    # 创建图片块
    blocks = [ImageBlock(p) for p in valid_paths]

    # 使用左下角算法布局
    canvas_w, canvas_h = bottom_left_layout(blocks, gap)

    # 创建画布并粘贴所有图片
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))  # 白色背景
    for block in blocks:
        canvas.paste(block.img, (block.x, block.y))

    # 保存结果
    canvas.save(output_path)
    print(f"所有图片已使用左下角算法合并至 {output_path}，尺寸: {canvas_w}x{canvas_h}（包含{gap}px间隙）")


# 使用示例
if __name__ == "__main__":
    image_paths = [
        "cropped_regions/25_region_7_figure_title.png",
        "cropped_regions/25_region_8_figure_title.png",
        "cropped_regions/25_region_9_figure_title.png",
        "cropped_regions/25_region_10_figure_title.png",
        "cropped_regions/25_region_11_figure_title.png"
        # 可以添加更多图片路径
    ]

    # 调用左下角算法合并图片，可指定间隙大小
    merge_all_images_bottom_left(image_paths, gap=10)
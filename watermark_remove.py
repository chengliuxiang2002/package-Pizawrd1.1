"""去除pdf文档水印"""
import fitz

PAGE_SIZE = (612, 792)

def get_img_size_list(page):
    """获取指定页图像大小信息列表"""
    img_list = page.get_images()
    if (len(img_list) >= 40):
        return []
    img_size_list = []  # 存放该页的图像大小列表
    for img in img_list:
        if ((img[2] > 32) or (img[3] > 32)):      # 图像如果过小，则判断为水印点
            img_size = (img[2], img[3])
            img_size_list.append(img_size)

    return img_size_list


def get_x_pos_list(page):
    """获取指定页xobject位置信息列表"""
    x_list = page.get_xobjects()
    if (len(x_list) >= 40):
        return []
    x_pos_list = []  # 存放该页的xobject位置信息列表
    for x in x_list:
        x_pos = x[3]
        if ((((x_pos[2] - x_pos[0]) > 64) and ((x_pos[2] - x_pos[0]) < PAGE_SIZE[0]))
                or ((x_pos[3] - x_pos[1]) > 64) and ((x_pos[3] - x_pos[1]) < PAGE_SIZE[1])):
            x_pos_list.append(x_pos)

    return x_pos_list


def get_com_list(list1, list2, list3):
    return list(set(list1) & set(list2) & set(list3))


def get_watermark_img_xref_list(page, watermark_img_size_list):
    """获取该页上符合水印图片大小的xref"""
    img_list = page.get_images()
    watermark_img_xref_list = []  # 存放该页符合水印图片大小xref列表
    for img in img_list:
        if ((img[2] > 32) or (img[3] > 32)):
            a = (img[2], img[3])
            if a in watermark_img_size_list:
                watermark_img_xref_list.append(img[0])

    return watermark_img_xref_list


def get_watermark_x_xref_list(page, watermark_x_pos_list):
    """获取该页上符合水印xobject位置信息的xref"""
    x_list = page.get_xobjects()
    watermark_x_xref_list = []  # 存放该页符合水印xobject位置信息xref列表
    for x in x_list:
        a = x[3]
        if (((a[2] - a[0]) > 64) or ((a[3] - a[1]) > 64)):    # 若xobject 对象尺寸太小，则抛弃
            if a in watermark_x_pos_list:
                watermark_x_xref_list.append(x[0])

    return watermark_x_xref_list


def remove_annot_watermark(path):
    """去除文档注释类水印"""

    with fitz.open(path) as doc:
        # 取一页进行注释类水印判断
        annots = list(doc[int(doc.page_count / 2)].annots())
        if (len(annots)):
            # 进行注释类水印去除
            for page in doc:
                for annot in page.annots():
                    if (annot.type[0] == 25):  # 判断注释是否为水印类型注释
                        page.delete_annot(annot)

            if doc.can_save_incrementally():
                doc.save(path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
            else:
                doc.save(path, clean=True, garbage=4)

def remove_img_watermark(path):
    """去除img类水印"""
    with fitz.open(path) as doc:
        # 是否含img类水印判断
        if (len(get_img_size_list(doc[int(doc.page_count / 2)]))):
            page1 = doc[2]  # 避免封面和目录
            page2 = doc[-3]  # 避免空白页
            page3 = doc[(doc.page_count - 1) // 2]
            watermark_img_size_list = get_com_list(get_img_size_list(page1), get_img_size_list(page2),
                                                                            get_img_size_list(page3))
            if len(watermark_img_size_list):
                for page in doc:
                    watermark_img_xref_list = get_watermark_img_xref_list(page, watermark_img_size_list)
                    for i in watermark_img_xref_list:
                        page.delete_image(i)

                if doc.can_save_incrementally():
                    doc.save(path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
                else:
                    doc.save(path, clean=True, garbage=4)


def remove_x_watermark(path):
    """去除xobject类水印"""

    with fitz.open(path) as doc:
        # 是否含xobject类型水印判断
        if (len(get_x_pos_list(doc[int(doc.page_count / 2)]))):
            page1 = doc[2]
            page2 = doc[-3]
            page3 = doc[(doc.page_count - 1) // 2]
            watermark_x_pos_list = get_com_list(get_x_pos_list(page1),
                                                get_x_pos_list(page2), get_x_pos_list(page3))

            if len(watermark_x_pos_list):
                for page in doc:
                    watermark_x_xref_list = get_watermark_x_xref_list(page, watermark_x_pos_list)
                    for i in watermark_x_xref_list:
                        doc.update_stream(i, b"")

                if doc.can_save_incrementally():
                    doc.save(path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
                else:
                    doc.save(path, clean=True, garbage=4)


def watermark_remove(path):
    """对于页数少于3页的pdf文档不进行水印处理"""
    flag = 0
    global PAGE_SIZE
    with fitz.open(path) as doc:
        if (doc.page_count < 3):
            flag = 1
    if (flag == 0):
        with fitz.open(path) as doc:
            page = doc[0]
            PAGE_SIZE = (page.rect[2], page.rect[3])

        remove_annot_watermark(path)
        remove_img_watermark(path)
        remove_x_watermark(path)




import fitz
import json
from shapely.geometry import Polygon


def judge_other_package(page, source_package_data):
    """判断该页上是否有多个封装图，若有2个以上封装图，则返回True，否则返回False"""
    count = 0
    for item in source_package_data:
        if page == item['page']:
            count += 1
    return count > 1


def judge_package(package):
    """判断该封装图信息是否满足封装图对象"""
    match_state1 = 1 if package['keyview'] is not None else 0
    match_state2 = 1 if package['Top'] is not None else 0
    match_state3 = 1 if len(package['Side']) > 0 else 0
    match_state4 = 1 if len(package['Detail']) > 0 else 0
    return (match_state1 + match_state2 + match_state3 + match_state4) >= 2


def get_rects_d(rect1_coords, rect2_coords):
    """输入两个矩形坐标，返回这两个矩形之间的最短距离"""
    rect1 = Polygon([(rect1_coords[0], rect1_coords[1]),
                     (rect1_coords[0], rect1_coords[3]),
                     (rect1_coords[2], rect1_coords[3]),
                     (rect1_coords[2], rect1_coords[1])])

    rect2 = Polygon([(rect2_coords[0], rect2_coords[1]),
                     (rect2_coords[0], rect2_coords[3]),
                     (rect2_coords[2], rect2_coords[3]),
                     (rect2_coords[2], rect2_coords[1])])

    return rect1.distance(rect2)


def package_match_view(detr_result):
    """封装图匹配视图"""
    # 获取DETR检测结果
    source_package_data = detr_result.get('source_package_data', [])
    source_keyview_data_temp = detr_result.get('source_keyview_data', [])
    source_Top_data_temp = detr_result.get('source_Top_data', [])
    source_Side_data_temp = detr_result.get('source_Side_data', [])
    source_Detail_data_temp = detr_result.get('source_Detail_data', [])
    source_Note_data = detr_result.get('source_Note_data', [])
    source_Form_data = detr_result.get('source_Form_data', [])

    # 重新分类整理数据
    source_keyview_data = []
    source_Top_data = []
    source_Side_data = []
    source_Detail_data = []

    # 处理原始keyview数据并根据type重新分配
    all_view_data = (
            source_keyview_data_temp +
            source_Top_data_temp +
            source_Side_data_temp +
            source_Detail_data_temp
    )

    for item in all_view_data:
        type = item.get('type', '')
        if type in ['TOPVIEW', 'Top']:
            source_Top_data.append(item)
        elif type in ['SIDEVIEW', 'Side']:
            source_Side_data.append(item)
        elif type in ['DETAIL', 'Detail']:
            source_Detail_data.append(item)
        else:
            source_keyview_data.append(item)

    # 对源数据进行排序

    source_package_data = sorted(source_package_data, key=lambda x: (x['page'], x['pos'][1]))
    source_keyview_data = sorted(source_keyview_data, key=lambda x: (x['page'], x['pos'][1]))
    source_Top_data = sorted(source_Top_data, key=lambda x: (x['page'], x['pos'][1]))
    source_Side_data = sorted(source_Side_data, key=lambda x: (x['page'], x['pos'][1]))
    source_Detail_data = sorted(source_Detail_data, key=lambda x: (x['page'], x['pos'][1]))
    source_Note_data = sorted(source_Note_data, key=lambda x: (x['page'], x['pos'][1]))
    source_Form_data = sorted(source_Form_data, key=lambda x: (x['page'], x['pos'][1]))
    package_data = []

    for i in range(len(source_package_data)):
        package_data.append({
            'page': source_package_data[i]['page'],
            'pos': source_package_data[i]['pos'],
            'conf': source_package_data[i]['conf'],
            'package_type': None,
            'keyview': None,
            'Top': None,
            'Side': [],
            'Detail': [],
            'Note': [],
            'Form': [{}, {}, {}]
        })

    # 封装图与关键特征视图信息匹配
    for i in range(len(source_package_data)):
        keyview_data = []
        Top_data = []
        current_page_num = source_package_data[i]['page']
        package_pos = source_package_data[i]['pos']

        # 匹配关键特征视图
        for j in range(len(source_keyview_data)):
            match_state = source_keyview_data[j]['match_state']
            keyview_page = source_keyview_data[j]['page']
            keyview_pos = source_keyview_data[j]['pos']
            if (match_state < 0 and
                    keyview_page == current_page_num and
                    int(get_rects_d(package_pos, keyview_pos)) == 0):
                keyview_data.append({
                    'package_type': source_keyview_data[j]['detr_type'],
                    'pos': keyview_pos,
                    'conf': source_keyview_data[j]['conf'],
                    'index': j
                })
            if keyview_page > current_page_num:
                break

        # 处理关键特征视图匹配结果
        if len(keyview_data) == 0:
            if source_package_data[i].get('type') != 'package' and source_package_data[i].get('type') is not None:
                package_data[i]['package_type'] = source_package_data[i]['type']

        elif len(keyview_data) == 1:
            # 根据source_package_data[i]['type']的值来决定package_type
            if source_package_data[i].get('type') != 'package' and source_package_data[i].get('type') is not None:
                package_data[i]['package_type'] = source_package_data[i]['type']
            else:
                # 使用source_keyview_data[j]['detr_type']
                package_data[i]['package_type'] = keyview_data[0]['package_type']
            package_data[i]['keyview'] = {
                'pos': keyview_data[0]['pos'],
                'conf': keyview_data[0]['conf']
            }
            source_keyview_data[keyview_data[0]['index']]['match_state'] = 1
        elif len(keyview_data) > 1:
            max_item = max(keyview_data, key=lambda x: x['conf'])
            if source_package_data[i].get('type') != 'package' and source_package_data[i].get('type') is not None:
                package_data[i]['package_type'] = source_package_data[i]['type']
            else:
                package_data[i]['package_type'] = max_item['package_type']
            package_data[i]['keyview'] = {
                'pos': max_item['pos'],
                'conf': max_item['conf']
            }
            source_keyview_data[max_item['index']]['match_state'] = 1

        # 匹配Top视图
        for j in range(len(source_Top_data)):
            match_state = source_Top_data[j]['match_state']
            Top_page = source_Top_data[j]['page']
            Top_pos = source_Top_data[j]['pos']
            if (match_state < 0 and
                    Top_page == current_page_num and
                    int(get_rects_d(Top_pos, package_pos)) == 0):
                Top_data.append({
                    'pos': Top_pos,
                    'conf': source_Top_data[j]['conf'],
                    'index': j
                })
            if Top_page > current_page_num:
                break

        # 处理Top视图匹配结果
        if len(Top_data) == 1:
            package_data[i]['Top'] = {
                'pos': Top_data[0]['pos'],
                'conf': Top_data[0]['conf']
            }
            source_Top_data[Top_data[0]['index']]['match_state'] = 1
        elif len(Top_data) > 1:
            max_item = max(Top_data, key=lambda x: x['conf'])
            package_data[i]['Top'] = {
                'pos': max_item['pos'],
                'conf': max_item['conf']
            }
            source_Top_data[max_item['index']]['match_state'] = 1

        # 匹配Side视图
        for j in range(len(source_Side_data)):
            match_state = source_Side_data[j]['match_state']
            Side_page = source_Side_data[j]['page']
            Side_pos = source_Side_data[j]['pos']
            if (match_state < 0 and
                    Side_page == current_page_num and
                    int(get_rects_d(Side_pos, package_pos)) == 0):
                package_data[i]['Side'].append({
                    'pos': Side_pos,
                    'conf': source_Side_data[j]['conf']
                })
                source_Side_data[j]['match_state'] = 1
            if Side_page > current_page_num:
                break

        # 匹配Detail视图
        for j in range(len(source_Detail_data)):
            match_state = source_Detail_data[j]['match_state']
            Detail_page = source_Detail_data[j]['page']
            Detail_pos = source_Detail_data[j]['pos']
            if (match_state < 0 and
                    Detail_page == current_page_num and
                    int(get_rects_d(Detail_pos, package_pos)) == 0):
                package_data[i]['Detail'].append({
                    'pos': Detail_pos,
                    'conf': source_Detail_data[j]['conf']
                })
                source_Detail_data[j]['match_state'] = 1
            if Detail_page > current_page_num:
                break

        # 匹配Note视图
        for j in range(len(source_Note_data)):
            match_state = source_Note_data[j]['match_state']
            Note_page = source_Note_data[j]['page']
            Note_pos = source_Note_data[j]['pos']
            if (match_state < 0 and
                    Note_page == current_page_num and
                    int(get_rects_d(Note_pos, package_pos)) == 0):
                package_data[i]['Note'].append({
                    'pos': Note_pos,
                    'conf': source_Note_data[j]['conf']
                })
                source_Note_data[j]['match_state'] = 1
            if Note_page > current_page_num:
                break

    # 封装图数据预筛选
    del_index = []
    for i in range(len(package_data)):
        # 修改筛选条件：必须存在keyview
        if not judge_package(package_data[i]):
            del_index.append(i)

    package_data = [item for index, item in enumerate(package_data) if index not in del_index]
    source_package_data = [item for index, item in enumerate(source_package_data) if index not in del_index]

    # 匹配Form
    for i in range(len(source_package_data)):
        current_page_num = source_package_data[i]['page']
        package_pos = source_package_data[i]['pos']

        # 上页封装表匹配
        if not judge_other_package(current_page_num, source_package_data):
            pre_Form_data = []
            pre_package_page = current_page_num - 1
            for j in range(len(source_Form_data)):
                match_state = source_Form_data[j]['match_state']
                Form_page = source_Form_data[j]['page']
                if match_state < 0 and Form_page == pre_package_page:
                    pre_Form_data.append({
                        'pos': source_Form_data[j]['pos'],
                        'conf': source_Form_data[j]['conf'],
                        'index': j
                    })
                if Form_page > pre_package_page:
                    break

            if len(pre_Form_data) == 1:
                package_data[i]['Form'][0] = {
                    'pos': pre_Form_data[0]['pos'],
                    'page': pre_package_page,
                    'conf': pre_Form_data[0]['conf']
                }
                source_Form_data[pre_Form_data[0]['index']]['match_state'] = -1
            elif len(pre_Form_data) > 1:
                max_item = max(pre_Form_data, key=lambda x: x['pos'][3])
                package_data[i]['Form'][0] = {
                    'pos': max_item['pos'],
                    'page': pre_package_page,
                    'conf': max_item['conf']
                }
                source_Form_data[max_item['index']]['match_state'] = -1

        # 下页封装表匹配
        if not judge_other_package(current_page_num, source_package_data):
            next_Form_data = []
            next_package_page = current_page_num + 1
            for j in range(len(source_Form_data)):
                match_state = source_Form_data[j]['match_state']
                Form_page = source_Form_data[j]['page']
                if match_state < 0 and Form_page == next_package_page:
                    next_Form_data.append({
                        'pos': source_Form_data[j]['pos'],
                        'conf': source_Form_data[j]['conf'],
                        'index': j
                    })
                if Form_page > next_package_page:
                    break

            if len(next_Form_data) == 1:
                package_data[i]['Form'][2] = {
                    'pos': next_Form_data[0]['pos'],
                    'page': next_package_page,
                    'conf': next_Form_data[0]['conf']
                }
                source_Form_data[next_Form_data[0]['index']]['match_state'] = -1
            elif len(next_Form_data) > 1:
                min_item = min(next_Form_data, key=lambda x: x['pos'][1])
                package_data[i]['Form'][2] = {
                    'pos': min_item['pos'],
                    'page': next_package_page,
                    'conf': min_item['conf']
                }
                source_Form_data[min_item['index']]['match_state'] = -1

        # 当前页封装表匹配
        cur_Form_data = []
        for j in range(len(source_Form_data)):
            match_state = source_Form_data[j]['match_state']
            Form_page = source_Form_data[j]['page']
            if match_state < 0 and Form_page == current_page_num:
                cur_Form_data.append({
                    'pos': source_Form_data[j]['pos'],
                    'conf': source_Form_data[j]['conf'],
                    'index': j
                })
            if Form_page > current_page_num:
                break

        if len(cur_Form_data) == 1:
            package_data[i]['Form'][1] = {
                'pos': cur_Form_data[0]['pos'],
                'page': current_page_num,
                'conf': cur_Form_data[0]['conf']
            }
            source_Form_data[cur_Form_data[0]['index']]['match_state'] = -1
        elif len(cur_Form_data) > 1:
            min_item = min(cur_Form_data, key=lambda x: get_rects_d(x['pos'], package_pos))
            package_data[i]['Form'][1] = {
                'pos': min_item['pos'],
                'page': current_page_num,
                'conf': min_item['conf']
            }
            source_Form_data[min_item['index']]['match_state'] = -1

    return package_data, detr_result


def get_new_pos(data, pdf_path):
    """生成pdf坐标"""
    with fitz.open(pdf_path) as doc:
        for i in range(len(data)):
            cur_page_num = data[i]['page']
            page = doc[cur_page_num ]
            rotation = page.rotation
            cur_page_width = page.rect[2]

            # 处理package_new_pos
            package_new_pos = list(data[i]['pos'])  # 复制原始坐标
            if rotation != 0:
                package_new_pos = [
                    data[i]['pos'][1],
                    cur_page_width - data[i]['pos'][0],
                    data[i]['pos'][3],
                    cur_page_width - data[i]['pos'][2]
                ]
            data[i]['new_pos'] = package_new_pos

            # 处理keyview_new_pos
            if data[i]['keyview'] is not None:
                keyview_new_pos = list(data[i]['keyview']['pos'])
                if rotation != 0:
                    keyview_new_pos = [
                        data[i]['keyview']['pos'][1],
                        cur_page_width - data[i]['keyview']['pos'][0],
                        data[i]['keyview']['pos'][3],
                        cur_page_width - data[i]['keyview']['pos'][2]
                    ]
                data[i]['keyview']['new_pos'] = keyview_new_pos

            # 处理Top_new_pos
            if data[i]['Top'] is not None:
                top_new_pos = list(data[i]['Top']['pos'])
                if rotation != 0:
                    top_new_pos = [
                        data[i]['Top']['pos'][1],
                        cur_page_width - data[i]['Top']['pos'][0],
                        data[i]['Top']['pos'][3],
                        cur_page_width - data[i]['Top']['pos'][2]
                    ]
                data[i]['Top']['new_pos'] = top_new_pos

            # 处理Side_new_pos
            for j in range(len(data[i]['Side'])):
                side_new_pos = list(data[i]['Side'][j]['pos'])
                if rotation != 0:
                    side_new_pos = [
                        data[i]['Side'][j]['pos'][1],
                        cur_page_width - data[i]['Side'][j]['pos'][0],
                        data[i]['Side'][j]['pos'][3],
                        cur_page_width - data[i]['Side'][j]['pos'][2]
                    ]
                data[i]['Side'][j]['new_pos'] = side_new_pos

            # 处理Detail_new_pos
            for j in range(len(data[i]['Detail'])):
                detail_new_pos = list(data[i]['Detail'][j]['pos'])
                if rotation != 0:
                    detail_new_pos = [
                        data[i]['Detail'][j]['pos'][1],
                        cur_page_width - data[i]['Detail'][j]['pos'][0],
                        data[i]['Detail'][j]['pos'][3],
                        cur_page_width - data[i]['Detail'][j]['pos'][2]
                    ]
                data[i]['Detail'][j]['new_pos'] = detail_new_pos

            # 处理Note_new_pos
            for j in range(len(data[i]['Note'])):
                note_new_pos = list(data[i]['Note'][j]['pos'])
                if rotation != 0:
                    note_new_pos = [
                        data[i]['Note'][j]['pos'][1],
                        cur_page_width - data[i]['Note'][j]['pos'][0],
                        data[i]['Note'][j]['pos'][3],
                        cur_page_width - data[i]['Note'][j]['pos'][2]
                    ]
                data[i]['Note'][j]['new_pos'] = note_new_pos

            # 处理Form_new_pos
            # 上页Form
            if data[i]['Form'][0]:
                pre_page_num = data[i]['Form'][0]['page']
                pre_page = doc[pre_page_num - 1]
                pre_rotation = pre_page.rotation
                pre_page_width = pre_page.rect[2]
                pre_form_new_pos = list(data[i]['Form'][0]['pos'])
                if pre_rotation != 0:
                    pre_form_new_pos = [
                        data[i]['Form'][0]['pos'][1],
                        pre_page_width - data[i]['Form'][0]['pos'][0],
                        data[i]['Form'][0]['pos'][3],
                        pre_page_width - data[i]['Form'][0]['pos'][2]
                    ]
                data[i]['Form'][0]['new_pos'] = pre_form_new_pos

            # 当前页Form
            if data[i]['Form'][1]:
                cur_form_page_num = data[i]['Form'][1]['page']
                cur_form_page = doc[cur_form_page_num - 1]
                cur_form_rotation = cur_form_page.rotation
                cur_form_width = cur_form_page.rect[2]
                cur_form_new_pos = list(data[i]['Form'][1]['pos'])
                if cur_form_rotation != 0:
                    cur_form_new_pos = [
                        data[i]['Form'][1]['pos'][1],
                        cur_form_width - data[i]['Form'][1]['pos'][0],
                        data[i]['Form'][1]['pos'][3],
                        cur_form_width - data[i]['Form'][1]['pos'][2]
                    ]
                data[i]['Form'][1]['new_pos'] = cur_form_new_pos

            # 下页Form
            if data[i]['Form'][2]:
                next_page_num = data[i]['Form'][2]['page']
                next_page = doc[next_page_num - 1]
                next_rotation = next_page.rotation
                next_page_width = next_page.rect[2]
                next_form_new_pos = list(data[i]['Form'][2]['pos'])
                if next_rotation != 0:
                    next_form_new_pos = [
                        data[i]['Form'][2]['pos'][1],
                        next_page_width - data[i]['Form'][2]['pos'][0],
                        data[i]['Form'][2]['pos'][3],
                        next_page_width - data[i]['Form'][2]['pos'][2]
                    ]
                data[i]['Form'][2]['new_pos'] = next_form_new_pos

    return data


def data2json(data2):
    """将匹配好的信息写入json"""
    with open('package_viewinfo.json', 'w', encoding='utf-8') as f:
        json.dump(data2, f, ensure_ascii=False, indent=2)


def transfer_data(data):
    """将封装图对象数据转为指定格式"""
    have_page = []
    data2 = []
    for i in range(len(data)):
        have_page.append(data[i]['page'])
        part_content = []

        # 添加关键特征视图
        if data[i]['package_type'] and data[i]['keyview'] is not None:
            part_content.append({
                'part_name': data[i]['package_type'],
                'page': data[i]['page'] ,
                'rect': data[i]['keyview']['pos'],
                'new_rect': data[i]['keyview']['new_pos']
            })

        # 添加Top视图
        if data[i]['Top']:
            part_content.append({
                'part_name': 'Top',
                'page': data[i]['page'],
                'rect': data[i]['Top']['pos'],
                'new_rect': data[i]['Top']['new_pos']
            })

        # 添加Side视图
        for j in range(len(data[i]['Side'])):
            part_content.append({
                'part_name': 'Side',
                'page': data[i]['page'],
                'rect': data[i]['Side'][j]['pos'],
                'new_rect': data[i]['Side'][j]['new_pos']
            })

        # 添加Detail视图
        for j in range(len(data[i]['Detail'])):
            part_content.append({
                'part_name': 'Detail',
                'page': data[i]['page'],
                'rect': data[i]['Detail'][j]['pos'],
                'new_rect': data[i]['Detail'][j]['new_pos']
            })

        # 添加Note
        for j in range(len(data[i]['Note'])):
            part_content.append({
                'part_name': 'Note',
                'page': data[i]['page'],
                'rect': data[i]['Note'][j]['pos'],
                'new_rect': data[i]['Note'][j]['new_pos']
            })

        # 添加Form
        # 上页Form
        if data[i]['Form'][0]:
            have_page.append(data[i]['page'] - 1)
            part_content.append({
                'part_name': 'Form',
                'page': data[i]['page'] - 1,
                'rect': data[i]['Form'][0]['pos'],
                'new_rect': data[i]['Form'][0]['new_pos']
            })

        # 当前页Form
        if data[i]['Form'][1]:
            part_content.append({
                'part_name': 'Form',
                'page': data[i]['page'] ,
                'rect': data[i]['Form'][1]['pos'],
                'new_rect': data[i]['Form'][1]['new_pos']
            })

        # 下页Form
        if data[i]['Form'][2]:
            have_page.append(data[i]['page'])
            part_content.append({
                'part_name': 'Form',
                'page': data[i]['page']+1,
                'rect': data[i]['Form'][2]['pos'],
                'new_rect': data[i]['Form'][2]['new_pos']
            })

        data_dict = {
            'page': data[i]['page'] ,
            'type': 'img',
            'rect': data[i]['pos'],
            'new_rect': data[i]['new_pos'],
            'package_type': data[i]['package_type'],
            'source': 'manual',
            'part_content': part_content if part_content else None,
            'reco_content': None
        }
        data2.append(data_dict)
    have_page = sorted(list(set(have_page)))
    print("数据data2结果：",data2)
    return data2, have_page


def clean_data(data2, detr_result):
    """
    数据清洗函数：
    1. 当part_content不存在关键特征视图时，将top视图的名称改为package_type
    2. 当part_content中不存在side视图时，找到原本detr检测到的side视图与其中坐标相同的数据，并将其名称改为detr_type
    3. 标准化package_type名称：'BALL GRID ARRAY'->'BGA', 'Quad Flat Package'->'QFP'
    """
    # 首先获取原始DETR检测结果
    if not detr_result or 'source_Side_data' not in detr_result:
        source_Side_data = []
    else:
        source_Side_data = detr_result.get('source_Side_data', [])

    cleaned_data = []

    for item in data2:
        # 创建副本以避免修改原始数据
        cleaned_item = item.copy()
        part_content = cleaned_item.get('part_content', [])

        if part_content:
            # 检查是否存在关键特征视图
            package_type = cleaned_item.get('package_type')
            has_keyview = any(
                part.get('part_name') == package_type
                for part in part_content
            )

            # 如果不存在关键特征视图且package_type存在，则修改Top视图名称
            if not has_keyview and package_type:
                for part in part_content:
                    if part.get('part_name') == 'Top':
                        part['part_name'] = package_type

            # 检查part_content中是否存在Side视图
            has_side_in_part_content = any(
                part.get('part_name') == 'Side'
                for part in part_content
            )

            # 如果part_content中不存在Side视图，则查找原始DETR检测结果中坐标相同的Side视图
            if not has_side_in_part_content:
                page_num = cleaned_item.get('page')
                for part in part_content:
                    # 对于part_content中的每个视图，检查是否能在原始Side数据中找到坐标相同的项
                    for side_item in source_Side_data:
                        if (side_item.get('page') == page_num and
                                side_item.get('pos') == part.get('rect')):
                            if side_item.get('detr_type'):
                                part['part_name'] = side_item['detr_type']
                            break

        # 标准化package_type名称，并同时修改相关的关键字特征视图名称
        original_package_type = cleaned_item.get('package_type')
        if original_package_type == 'BALL GRID ARRAY':
            cleaned_item['package_type'] = 'BGA'
            # 同时修改part_content中与原package_type相同的关键字特征视图名称
            if part_content:
                for part in part_content:
                    if part.get('part_name') == original_package_type:
                        part['part_name'] = 'BGA'
        elif original_package_type == 'Quad Flat Package':
            cleaned_item['package_type'] = 'QFP'
            # 同时修改part_content中与原package_type相同的关键字特征视图名称
            if part_content:
                for part in part_content:
                    if part.get('part_name') == original_package_type:
                        part['part_name'] = 'QFP'
        elif original_package_type == 'DFN_SON':
            cleaned_item['package_type'] = 'SON'
            # 同时修改part_content中与原package_type相同的关键字特征视图名称
            if part_content:
                for part in part_content:
                    if part.get('part_name') == original_package_type:
                        part['part_name'] = 'SON'

        cleaned_data.append(cleaned_item)

    return cleaned_data



def process_package_matching(pdf_path, detr_results):
    """处理封装图匹配的主函数"""
    # 执行匹配
    package_data, detr_result = package_match_view(detr_results)

    # 生成新坐标
    package_data = get_new_pos(package_data, pdf_path)

    # 转换数据格式
    data2, have_page = transfer_data(package_data)
    # 数据清洗
    data2 = clean_data(data2, detr_result)
    # 保存到JSON
    data2json(data2)

    return package_data, data2, have_page, detr_result

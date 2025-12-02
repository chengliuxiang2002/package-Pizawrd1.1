from other_match_dbnet import other_match_boxes_by_overlap
from pin_match_dbnet import PINnum_find_matching_boxes
from angle_match_dbnet import angle_find_matching_boxes
from num_match_dbnet import num_match_size_boxes
from merge_box_and_ocr import process_all_variables
from data_process_with_Absolutely import process_recognized_strings
import copy
import json
import os

def save_box_to_txt(boxes, filename):
    """保存框坐标到txt文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        for box in boxes:
            if isinstance(box, dict) and 'location' in box:
                # 处理字典格式的框
                loc = box['location']
                f.write(f"{loc[0]} {loc[1]} {loc[2]} {loc[3]}\n")
            else:
                # 处理列表格式的框 [x1,y1,x2,y2]
                f.write(f"{box[0]} {box[1]} {box[2]} {box[3]}\n")

def save_small_boxes_to_txt(data_list, base_filename):
    """保存small_boxes到单独的txt文件"""
    for i, item in enumerate(data_list):
        if 'small_boxes' in item and item['small_boxes']:
            filename = f"{base_filename}_small_boxes_{i}.txt"
            save_box_to_txt(item['small_boxes'], filename)

def test_all(package_classes, img_path, dbnet, other, yolox_serial_num, angle_pairs, yolox_num, BGA_serial_num=None, BGA_serial_letter=None):
    """
    测试所有匹配函数
    
    Args:
        package_classes: 封装类型 ('BGA' 或其他)
        dbnet: dbnet识别到的文本框列表
        other: 其他大框列表
        yolox_serial_num: YOLOX识别的PIN序列号框
        angle_pairs: 角度外框列表
        yolox_num: 尺寸外框列表
        BGA_serial_num: BGA序列号数字框 (BGA特有)
        BGA_serial_letter: BGA序列号字母框 (BGA特有)
    """
    # 创建结果目录
    
    # 使用深拷贝避免修改原始数据
    dbnet_copy = copy.deepcopy(dbnet)
    other_copy = copy.deepcopy(other)
    yolox_serial_num_copy = copy.deepcopy(yolox_serial_num)
    angle_pairs_copy = copy.deepcopy(angle_pairs)
    yolox_num_copy = copy.deepcopy(yolox_num)
    BGA_serial_num_copy = copy.deepcopy(BGA_serial_num)
    BGA_serial_letter_copy = copy.deepcopy(BGA_serial_letter)
    
    # 测试 testF4_1
    print("Testing testF4_1...")
    new_dbnet_1, new_other = other_match_boxes_by_overlap(dbnet_copy, other_copy)
    
    # 保存 testF4_1 结果
    # save_box_to_txt(new_dbnet_1, "D:\\BaiduNetdiskDownload\\post0\\0910\\data\\testF4_1_new_dbnet_data.txt")
    # save_small_boxes_to_txt(new_other, "D:\\BaiduNetdiskDownload\\post0\\0910\\data\\testF4_1")
    
    # 测试 testF4_2
    print("Testing testF4_2...")
    new_dbnet_2, new_yolox_serial_num = PINnum_find_matching_boxes(new_dbnet_1, yolox_serial_num_copy)
    
    # BGA特有类型处理
    new_BGA_serial_num = None
    new_BGA_serial_letter = None
    
    if package_classes == 'BGA':
        # 处理BGA序列号数字
        
        new_dbnet_2_1, new_BGA_serial_num = PINnum_find_matching_boxes(new_dbnet_2, BGA_serial_num_copy)
        
        # 处理BGA序列号字母
        
        new_dbnet_2_2, new_BGA_serial_letter = PINnum_find_matching_boxes(new_dbnet_2_1, BGA_serial_letter_copy)
        new_dbnet_2 = new_dbnet_2_2
    
    # 保存 testF4_2 结果
    # save_box_to_txt(new_dbnet_2, "D:\\BaiduNetdiskDownload\\post0\\0910\\data\\testF4_2_new_dbnet_data.txt")
    # save_small_boxes_to_txt(new_yolox_serial_num, "D:\\BaiduNetdiskDownload\\post0\\0910\\data\\testF4_2")
    
    # 保存BGA特有类型结果
    if package_classes == 'BGA':
        if new_BGA_serial_num:
            save_box_to_txt(new_dbnet_2_1, "D:\\BaiduNetdiskDownload\\post0\\0910\\data\\testF4_2_1_new_dbnet_data.txt")
            save_small_boxes_to_txt(new_BGA_serial_num, "D:\\BaiduNetdiskDownload\\post0\\0910\\data\\testF4_2_1")
        if new_BGA_serial_letter:
            save_box_to_txt(new_dbnet_2_2, "D:\\BaiduNetdiskDownload\\post0\\0910\\data\\testF4_2_2_new_dbnet_data.txt")
            save_small_boxes_to_txt(new_BGA_serial_letter, "D:\\BaiduNetdiskDownload\\post0\\0910\\data\\testF4_2_2")
    
    # 测试 testF4_3
    print("Testing testF4_3...")
    angle_boxes_dicts, new_dbnet_3, new_angle_pairs = angle_find_matching_boxes(new_dbnet_2, angle_pairs_copy)
    
    # 保存 testF4_3 结果
    # save_box_to_txt(new_dbnet_3, "D:\\BaiduNetdiskDownload\\post0\\0910\\data\\testF4_3_new_dbnet_data.txt")
    # save_box_to_txt(new_angle_pairs, "D:\\BaiduNetdiskDownload\\post0\\0910\\data\\testF4_3_new_angle_pairs.txt")
    
    # 测试 testF4_4
    print("Testing testF4_4...")
    new_dbnet_4, new_yolox_num = num_match_size_boxes(new_dbnet_3, yolox_num_copy)
    
    # 保存 testF4_4 结果
    # save_box_to_txt(new_dbnet_4, "D:\\BaiduNetdiskDownload\\post0\\0910\\data\\testF4_4_new_dbnet_data.txt")

    # 确保数据可以被JSON序列化
    def convert_to_serializable(obj):
        """将对象转换为可JSON序列化的格式"""
        if isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif hasattr(obj, '__dict__'):
            # 如果是自定义对象，尝试转换为字典
            return convert_to_serializable(obj.__dict__)
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            # 对于其他类型，尝试转换为字符串
            return str(obj)

    try:
        # 转换数据为可序列化格式
        serializable_data = convert_to_serializable(new_yolox_num)
        
        with open("D:\\BaiduNetdiskDownload\\post0\\0910\\data\\testF4_4_new_yolox_num.json", 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print("JSON保存成功!")
    except Exception as e:
        print(f"JSON保存失败: {e}")
        # 尝试保存为文本格式作为备选
        try:
            with open("D:\\BaiduNetdiskDownload\\post0\\0910\\data\\testF4_4_new_yolox_num_fallback.txt", 'w', encoding='utf-8') as f:
                f.write(str(new_yolox_num))
            print("已保存为备选文本格式")
        except Exception as e2:
            print(f"备选保存也失败: {e2}")

    save_small_boxes_to_txt(new_yolox_num, "D:\\BaiduNetdiskDownload\\post0\\0910\\data\\testF4_4.txt")
    
    # 测试 testF4_5
    print("Testing testF4_5...")
    
    # 准备参数
    process_params = {
        'image_path': img_path,
        'new_other': new_other,
        'new_yolox_serial_num': new_yolox_serial_num,
        'angle_boxes_dicts': angle_boxes_dicts,
        'new_yolox_num': new_yolox_num
    }
    
    # 添加BGA特有类型参数
    if package_classes == 'BGA':
        process_params['BGA_serial_num'] = new_BGA_serial_num if new_BGA_serial_num else []
        process_params['BGA_serial_letter'] = new_BGA_serial_letter if new_BGA_serial_letter else []
    
    results = process_all_variables(**process_params)

    # 然后分别提取结果
    new_other_processed = results['new_other']
    new_yolox_serial_num_processed = results['new_yolox_serial_num']
    angle_boxes_dicts_processed = results['angle_boxes_dicts']
    new_yolox_num_processed = results['new_yolox_num']
    
    # BGA特有类型结果
    BGA_serial_num_processed = results.get('BGA_serial_num', [])
    BGA_serial_letter_processed = results.get('BGA_serial_letter', [])
    
    # 保存 testF4_5 结果
    # 查看所有变量的处理结果
    print("=== OCR识别结果汇总 ===")
    for var_name, var_data in results.items():
        print(f"\n--- {var_name} ---")
        print(f"元素数量: {len(var_data)}")
        
        for i, item in enumerate(var_data):
            print(f"  元素 {i}:")
            print(f"    位置: {item['location']}")
            print(f"    小框数量: {len(item.get('small_boxes', []))}")
            print(f"    OCR结果: {item['ocr_strings']}")
            print(f"    绝对信息: {item['Absolutely']}")
            print("    " + "-" * 30)
    
    # 测试 testF4_6
    print("Testing testF4_6...")
    print("=== 输入数据检查 ===")
    print(f"new_other_processed 类型: {type(new_other_processed)}, 内容: {new_other_processed}")
    print(f"new_yolox_serial_num_processed 类型: {type(new_yolox_serial_num_processed)}, 内容: {new_yolox_serial_num_processed}")
    print(f"angle_boxes_dicts_processed 类型: {type(angle_boxes_dicts_processed)}, 内容: {angle_boxes_dicts_processed}")
    print(f"new_yolox_num_processed 类型: {type(new_yolox_num_processed)}, 内容: {new_yolox_num_processed}")
    
    # 准备testF4_6参数
    f4_6_params = {
        'new_other': new_other_processed,
        'new_yolox_serial_num': new_yolox_serial_num_processed,
        'angle_boxes_dicts': angle_boxes_dicts_processed,
        'new_yolox_num': new_yolox_num_processed
    }
    
    # 添加BGA特有类型参数
    if package_classes == 'BGA':
        f4_6_params['BGA_serial_num'] = BGA_serial_num_processed
        f4_6_params['BGA_serial_letter'] = BGA_serial_letter_processed
    
    result = process_recognized_strings(**f4_6_params)
    # 保存testF4_6结果到文件
    def save_testF4_6_results(result, filename):
        """保存testF4_6的处理结果到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=== testF4_6 处理结果 ===\n\n")
                for key, value in result.items():
                    f.write(f"--- {key} ---\n")
                    if isinstance(value, list):
                        f.write(f"元素数量: {len(value)}\n")
                        for i, item in enumerate(value):
                            f.write(f"  元素 {i}:\n")
                            if isinstance(item, dict):
                                for k, v in item.items():
                                    if k == 'location':
                                        f.write(f"    位置: {v}\n")
                                    elif k == 'ocr_strings':
                                        f.write(f"    OCR结果: {v}\n")
                                    elif k == 'Absolutely':
                                        f.write(f"    类别: {v}\n")
                                    elif k == 'max_nom_min':
                                        f.write(f"    数值处理结果: {v}\n")
                                    elif k == 'small_boxes':
                                        f.write(f"    小框数量: {len(v) if v else 0}\n")
                                    else:
                                        f.write(f"    {k}: {v}\n")
                            else:
                                f.write(f"    {item}\n")
                            f.write("    " + "-" * 30 + "\n")
                    else:
                        f.write(f"{value}\n")
                    f.write("\n")
            print(f"testF4_6结果已保存到: {filename}")
        except Exception as e:
            print(f"保存testF4_6结果失败: {e}")

    # 保存testF4_6结果
    # testF4_6_filename = "D:\\BaiduNetdiskDownload\\post0\\0910\\data\\testF4_6_result.txt"
    # save_testF4_6_results(result, testF4_6_filename)

    # 同时保存为JSON格式
    def save_testF4_6_json(result, filename):
        """保存testF4_6结果为JSON格式"""
        try:
            # 使用之前定义的convert_to_serializable函数
            serializable_result = convert_to_serializable(result)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)
            print(f"testF4_6 JSON结果已保存到: {filename}")
        except Exception as e:
            print(f"保存testF4_6 JSON结果失败: {e}")

    testF4_6_json_filename = "D:\\BaiduNetdiskDownload\\post0\\0910\\data\\testF4_6_result.json"
    save_testF4_6_json(result, testF4_6_json_filename)

    # 控制台打印结果（保持不变）
    print("=== testF4_6 处理结果 ===")
    for key, value in result.items():
        print(f"{key}: {value}")

    print("All tests completed! Results saved.")
    # 打印结果
    for key, value in result.items():
        print(f"{key}: {value}")
    
    print("All tests completed! Results saved.")
    
    # 返回最终结果
    final_results = {
        'final_dbnet_data': new_dbnet_4,
        'testF4_1_results': new_other,
        'testF4_2_results': new_yolox_serial_num,
        'testF4_3_results': angle_boxes_dicts,
        'testF4_4_results': new_yolox_num
    }
    
    # 添加BGA特有类型结果
    if package_classes == 'BGA':
        final_results['BGA_serial_num'] = new_BGA_serial_num
        final_results['BGA_serial_letter'] = new_BGA_serial_letter
    
    return final_results
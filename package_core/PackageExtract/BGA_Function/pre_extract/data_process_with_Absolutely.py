
# import re
# import numpy as np

# def process_recognized_strings(new_other, new_yolox_serial_num, angle_boxes_dicts, new_yolox_num, 
#                               BGA_serial_num=None, BGA_serial_letter=None):
#     """
#     对识别好的字符串内容进行整理，根据Absolutely类别进行不同处理
#     支持处理列表中的多个字典项，包括BGA特有类型
#     """
    
#     def ensure_string(text):
#         """确保输入是字符串类型"""
#         if text is None:
#             return ""
#         if isinstance(text, (list, tuple)):
#             # 如果是列表或元组，连接所有元素
#             return ' '.join([ensure_string(item) for item in text])
#         if isinstance(text, (int, float)):
#             # 如果是数字，转换为字符串
#             return str(text)
#         if not isinstance(text, str):
#             # 其他类型尝试转换为字符串
#             try:
#                 return str(text)
#             except:
#                 return ""
#         return text
    
#     def preprocess_text(text):
#         """
#         预处理文本，处理逗号作为小数点的情况
#         例如："9，60" -> "9.60"
#         """
#         text = ensure_string(text)
#         if not text:
#             return text
            
#         # 处理逗号作为小数点的情况
#         # 模式：数字 + 逗号 + 数字
#         pattern = r'(\d+)[，,](\d+)'
#         text = re.sub(pattern, r'\1.\2', text)
        
#         return text
    
#     def extract_and_process_numbers(text):
#         """
#         提取并处理数字，特别处理乘号X后的数字和可能被错误识别的等号
#         例如："0.80X8.6.40" -> [0.8, 8, 6.4]
#         """
#         text = ensure_string(text)
#         if not text:
#             return []
            
#         # 预处理文本，处理逗号作为小数点的情况
#         text = preprocess_text(text)
        
#         # 检查是否包含乘号X
#         if 'X' in text or 'x' in text:
#             # 使用X或x分割字符串
#             parts = re.split(r'[Xx]', text)
#             if len(parts) >= 2:
#                 numbers = []
                
#                 # 处理第一部分（乘号前）
#                 first_part = parts[0].strip()
#                 first_numbers = extract_simple_numbers(first_part)
#                 numbers.extend(first_numbers)
                
#                 # 处理第二部分（乘号后）
#                 second_part = parts[1].strip()
                
#                 # 检查第二部分是否有多个点（可能等号被识别为点）
#                 dots_count = second_part.count('.')
#                 if dots_count > 1:
#                     # 有多个点，可能是等号被识别为点
#                     # 例如 "8.6.40" 应该分为 "8" 和 "6.40"
#                     dot_parts = second_part.split('.')
                    
#                     # 第一个点前的部分是一个整数
#                     if dot_parts[0]:
#                         try:
#                             numbers.append(float(dot_parts[0]))
#                         except:
#                             pass
                    
#                     # 剩下的部分尝试组合成小数
#                     if len(dot_parts) > 2:
#                         # 从第二个点开始，将后面的部分组合成一个小数
#                         decimal_part = '.'.join(dot_parts[1:])
#                         try:
#                             numbers.append(float(decimal_part))
#                         except:
#                             pass
#                 else:
#                     # 只有一个点或没有点，正常提取数字
#                     second_numbers = extract_simple_numbers(second_part)
#                     numbers.extend(second_numbers)
                
#                 # 按从大到小排序
#                 numbers.sort(reverse=True)
#                 return numbers
        
#         # 如果没有乘号，使用常规方法提取数字
#         return extract_simple_numbers(text)
    
#     def extract_simple_numbers(text):
#         """
#         简单的数字提取，不包括乘号X的特殊处理
#         """
#         text = ensure_string(text)
#         if not text:
#             return []
#         try:
#             # 改进的正则表达式，能正确处理各种数字格式
#             pattern = r'[-+]?[±]?(\d*\.\d+|\d+\.?\d*)'
#             matches = re.findall(pattern, text)
#             return [float(match) for match in matches if match]
#         except Exception as e:
#             print(f"提取数字时出错: {e}, 文本: {text}")
#             return []
    
#     def preprocess_number_string(num_str):
#         """
#         预处理数字字符串，处理前导0的情况
#         例如：050 -> 0.50, 025 -> 0.25
#         """
#         num_str = ensure_string(num_str)
#         if not num_str:
#             return num_str
            
#         # 如果字符串以0开头且长度大于1，且没有小数点，则添加小数点
#         if num_str.startswith('0') and len(num_str) > 1 and '.' not in num_str:
#             # 检查是否已经是小数形式（如0.50）
#             if not re.match(r'^\d+\.\d+$', num_str):
#                 # 将前导0后面的部分作为小数部分
#                 num_str = '0.' + num_str[1:]
        
#         return num_str
    
#     def extract_max_min_values(text):
#         """
#         从文本中提取MAX和MIN值（不区分大小写）
#         返回格式: (max_value, min_value) 或 (None, None)
#         """
#         text = ensure_string(text)
#         if not text:
#             return None, None
            
#         # 预处理文本，处理逗号作为小数点的情况
#         text = preprocess_text(text)
#         text_lower = text.lower()
#         max_value = None
#         min_value = None
        
#         # 改进的正则表达式，能处理MAX/MIN在数字前后的情况
#         # 情况1: MAX/MIN在数字前面 (如 "MAX 0.5", "Min.0.25")
#         max_pattern_before = r'max[\.\s:：]*([-+]?\d*\.\d+|\d+\.?\d*)'
#         min_pattern_before = r'min[\.\s:：]*([-+]?\d*\.\d+|\d+\.?\d*)'
        
#         # 情况2: MAX/MIN在数字后面 (如 "0.5 MAX", "0.25 MIN")
#         max_pattern_after = r'([-+]?\d*\.\d+|\d+\.?\d*)[\.\s:：]*max'
#         min_pattern_after = r'([-+]?\d*\.\d+|\d+\.?\d*)[\.\s:：]*min'
        
#         # 尝试匹配MAX在数字前面的情况
#         max_match_before = re.search(max_pattern_before, text_lower)
#         if max_match_before:
#             try:
#                 # 提取匹配的数字部分
#                 num_str = max_match_before.group(1)
#                 # 预处理数字字符串，处理前导0的情况
#                 num_str = preprocess_number_string(num_str)
#                 # 处理特殊情况，如"Min.0.25"中的".0.25"
#                 if num_str.startswith('.') and len(num_str) > 1:
#                     num_str = '0' + num_str
#                 max_value = float(num_str)
#                 print(f"检测到MAX在数字前: {max_match_before.group(0)} -> {num_str} -> {max_value}")
#             except Exception as e:
#                 print(f"提取MAX值失败: {e}")
#         else:
#             # 如果MAX在数字前面没匹配到，尝试匹配MAX在数字后面的情况
#             max_match_after = re.search(max_pattern_after, text_lower)
#             if max_match_after:
#                 try:
#                     # 提取匹配的数字部分
#                     num_str = max_match_after.group(1)
#                     # 预处理数字字符串，处理前导0的情况
#                     num_str = preprocess_number_string(num_str)
#                     # 处理特殊情况，如"Min.0.25"中的".0.25"
#                     if num_str.startswith('.') and len(num_str) > 1:
#                         num_str = '0' + num_str
#                     max_value = float(num_str)
#                     print(f"检测到MAX在数字后: {max_match_after.group(0)} -> {num_str} -> {max_value}")
#                 except Exception as e:
#                     print(f"提取MAX值失败: {e}")
        
#         # 尝试匹配MIN在数字前面的情况
#         min_match_before = re.search(min_pattern_before, text_lower)
#         if min_match_before:
#             try:
#                 # 提取匹配的数字部分
#                 num_str = min_match_before.group(1)
#                 # 预处理数字字符串，处理前导0的情况
#                 num_str = preprocess_number_string(num_str)
#                 # 处理特殊情况，如"Min.0.25"中的".0.25"
#                 if num_str.startswith('.') and len(num_str) > 1:
#                     num_str = '0' + num_str
#                 min_value = float(num_str)
#                 print(f"检测到MIN在数字前: {min_match_before.group(0)} -> {num_str} -> {min_value}")
#             except Exception as e:
#                 print(f"提取MIN值失败: {e}")
#         else:
#             # 如果MIN在数字前面没匹配到，尝试匹配MIN在数字后面的情况
#             min_match_after = re.search(min_pattern_after, text_lower)
#             if min_match_after:
#                 try:
#                     # 提取匹配的数字部分
#                     num_str = min_match_after.group(1)
#                     # 预处理数字字符串，处理前导0的情况
#                     num_str = preprocess_number_string(num_str)
#                     # 处理特殊情况，如"Min.0.25"中的".0.25"
#                     if num_str.startswith('.') and len(num_str) > 1:
#                         num_str = '0' + num_str
#                     min_value = float(num_str)
#                     print(f"检测到MIN在数字后: {min_match_after.group(0)} -> {num_str} -> {min_value}")
#                 except Exception as e:
#                     print(f"提取MIN值失败: {e}")
        
#         return max_value, min_value
    
#     def process_single_item(item):
#         """处理单个字典项"""
#         if not item or not isinstance(item, dict):
#             return item
            
#         # 复制数据避免修改原数据
#         result_item = item.copy()
        
#         # 获取类别
#         absolutely_list = result_item.get('Absolutely', [])
#         category = absolutely_list[0] if absolutely_list and len(absolutely_list) > 0 else 'other'
        
#         # 根据类别处理
#         if category == 'num':
#             numbers = []
            
#             # 获取OCR文本并确保是字符串
#             ocr_strings = result_item.get('ocr_strings', '')
#             ocr_text = ensure_string(ocr_strings)
#             print(f"处理num类别文本: {ocr_text}")
            
#             # 如果有小框，优先处理小框中的数字
#             small_boxes = result_item.get('small_boxes', [])
#             if small_boxes and len(small_boxes) > 0:
#                 # 如果有小框，我们可能需要处理每个小框的OCR结果
#                 # 但根据之前的设计，小框的OCR结果可能已经合并到主OCR字符串中
#                 # 所以这里我们主要从主OCR字符串中提取
#                 numbers.extend(extract_and_process_numbers(ocr_text))
#             else:
#                 # 处理主OCR字符串
#                 numbers.extend(extract_and_process_numbers(ocr_text))
            
#             # 检查是否包含±符号
#             if '±' in ocr_text:
#                 pattern = r'([-+]?\d*\.\d+|\d+\.?\d*)\s*±\s*([-+]?\d*\.\d+|\d+\.?\d*)'
#                 match = re.search(pattern, ocr_text)
#                 if match:
#                     try:
#                         nom_str = match.group(1)
#                         tolerance_str = match.group(2)
#                         # 预处理数字字符串，处理前导0的情况
#                         nom_str = preprocess_number_string(nom_str)
#                         tolerance_str = preprocess_number_string(tolerance_str)
#                         # 处理特殊情况，如".25"变为"0.25"
#                         if nom_str.startswith('.') and len(nom_str) > 1:
#                             nom_str = '0' + nom_str
#                         if tolerance_str.startswith('.') and len(tolerance_str) > 1:
#                             tolerance_str = '0' + tolerance_str
                            
#                         nom = float(nom_str)
#                         tolerance = float(tolerance_str)
#                         max_val = nom + tolerance
#                         min_val = nom - tolerance
#                         result_item['max_nom_min'] = [max_val, nom, min_val]
#                         print(f"检测到±符号: {nom} ± {tolerance}")
#                         return result_item
#                     except Exception as e:
#                         print(f"处理±符号失败: {e}")
            
#             # 如果没有±，检查是否有MAX和MIN关键字
#             max_value, min_value = extract_max_min_values(ocr_text)
            
#             if max_value is not None and min_value is not None:
#                 # 同时有MAX和MIN，计算平均值作为nom值
#                 nom_value = (max_value + min_value) / 2
#                 result_item['max_nom_min'] = [max_value, nom_value, min_value]
#                 print(f"检测到MAX和MIN: MAX={max_value}, MIN={min_value}, NOM={nom_value}")
            
#             elif max_value is not None:
#                 # 只有MAX，只提取MAX值
#                 result_item['max_nom_min'] = [max_value]
#                 print(f"检测到MAX: {max_value}")
            
#             elif min_value is not None:
#                 # 只有MIN，只提取MIN值
#                 result_item['max_nom_min'] = [min_value]
#                 print(f"检测到MIN: {min_value}")
            
#             else:
#                 # 没有±、MAX、MIN，只提取纯数字，按由大到小排序
#                 if numbers:
#                     # 注意：extract_and_process_numbers已经排序了
#                     result_item['max_nom_min'] = numbers
#                     print(f"提取到数字: {numbers}")
#                 else:
#                     result_item['max_nom_min'] = []
#                     print("未提取到任何数字")
                    
#         elif category == 'angle':
#             numbers = []
#             ocr_strings = result_item.get('ocr_strings', '')
#             ocr_text = ensure_string(ocr_strings)
            
#             # 检查是否包含°符号
#             if '°' in ocr_text:
#                 pattern = r'([-+]?\d*\.\d+|\d+\.?\d*)\s*°'
#                 matches = re.findall(pattern, ocr_text)
#                 numbers = [float(match) for match in matches if match]
#             else:
#                 numbers = extract_and_process_numbers(ocr_text)
            
#             result_item['max_nom_min'] = numbers
            
#         elif category == 'PIN_num':
#             # PIN_num类别不做处理，只新增字段
#             result_item['max_nom_min'] = []
            
#         elif category == 'other':
#             numbers = []
#             ocr_strings = result_item.get('ocr_strings', '')
#             ocr_text = ensure_string(ocr_strings)
#             numbers = extract_and_process_numbers(ocr_text)
#             result_item['max_nom_min'] = numbers
            
#         else:
#             # 未知类别，按other处理
#             numbers = []
#             ocr_strings = result_item.get('ocr_strings', '')
#             ocr_text = ensure_string(ocr_strings)
#             numbers = extract_and_process_numbers(ocr_text)
#             result_item['max_nom_min'] = numbers
        
#         return result_item
    
#     def process_input_data(data):
#         """处理输入数据，支持列表或单个字典"""
#         if isinstance(data, list):
#             # 如果是列表，处理每个元素
#             return [process_single_item(item) for item in data]
#         elif isinstance(data, dict):
#             # 如果是单个字典，直接处理
#             return process_single_item(data)
#         else:
#             # 其他类型原样返回
#             return data
    
#     # 处理每个输入变量
#     results = {
#         'new_other': process_input_data(new_other),
#         'new_yolox_serial_num': process_input_data(new_yolox_serial_num),
#         'angle_boxes_dicts': process_input_data(angle_boxes_dicts),
#         'new_yolox_num': process_input_data(new_yolox_num)
#     }
    
#     # 处理BGA特有类型
#     if BGA_serial_num is not None:
#         results['BGA_serial_num'] = process_input_data(BGA_serial_num)
    
#     if BGA_serial_letter is not None:
#         results['BGA_serial_letter'] = process_input_data(BGA_serial_letter)
    
#     return results





import re
import numpy as np

def process_recognized_strings(new_other, new_yolox_serial_num, angle_boxes_dicts, new_yolox_num, 
                              BGA_serial_num=None, BGA_serial_letter=None):
    """
    对识别好的字符串内容进行整理，根据Absolutely类别进行不同处理
    支持处理列表中的多个字典项，包括BGA特有类型
    """
    
    def ensure_string(text):
        """确保输入是字符串类型"""
        if text is None:
            return ""
        if isinstance(text, (list, tuple)):
            # 如果是列表或元组，连接所有元素
            return ' '.join([ensure_string(item) for item in text])
        if isinstance(text, (int, float)):
            # 如果是数字，转换为字符串
            return str(text)
        if not isinstance(text, str):
            # 其他类型尝试转换为字符串
            try:
                return str(text)
            except:
                return ""
        return text
    
    def preprocess_text(text):
        """
        预处理文本，处理逗号作为小数点的情况
        例如："9，60" -> "9.60"
        """
        text = ensure_string(text)
        if not text:
            return text
            
        # 处理逗号作为小数点的情况
        # 模式：数字 + 逗号 + 数字
        pattern = r'(\d+)[，,](\d+)'
        text = re.sub(pattern, r'\1.\2', text)
        
        return text
    
    def extract_and_process_numbers(text):
        """
        提取并处理数字，特别处理乘号X后的数字和可能被错误识别的等号
        例如："0.80X8.6.40" -> [0.8, 8, 6.4]
        """
        text = ensure_string(text)
        if not text:
            return []
            
        # 预处理文本，处理逗号作为小数点的情况
        text = preprocess_text(text)
        
        # 检查是否包含乘号X
        if 'X' in text or 'x' in text:
            # 使用X或x分割字符串
            parts = re.split(r'[Xx]', text)
            if len(parts) >= 2:
                numbers = []
                
                # 处理第一部分（乘号前）
                first_part = parts[0].strip()
                first_numbers = extract_simple_numbers(first_part)
                numbers.extend(first_numbers)
                
                # 处理第二部分（乘号后）
                second_part = parts[1].strip()
                
                # 检查第二部分是否有多个点（可能等号被识别为点）
                dots_count = second_part.count('.')
                if dots_count > 1:
                    # 有多个点，可能是等号被识别为点
                    # 例如 "8.6.40" 应该分为 "8" 和 "6.40"
                    dot_parts = second_part.split('.')
                    
                    # 第一个点前的部分是一个整数
                    if dot_parts[0]:
                        try:
                            numbers.append(float(dot_parts[0]))
                        except:
                            pass
                    
                    # 剩下的部分尝试组合成小数
                    if len(dot_parts) > 2:
                        # 从第二个点开始，将后面的部分组合成一个小数
                        decimal_part = '.'.join(dot_parts[1:])
                        try:
                            numbers.append(float(decimal_part))
                        except:
                            pass
                else:
                    # 只有一个点或没有点，正常提取数字
                    second_numbers = extract_simple_numbers(second_part)
                    numbers.extend(second_numbers)
                
                # 按从大到小排序
                numbers.sort(reverse=True)
                return numbers
        
        # 如果没有乘号，使用常规方法提取数字
        return extract_simple_numbers(text)
    
    def extract_simple_numbers(text):
        """
        简单的数字提取，不包括乘号X的特殊处理
        """
        text = ensure_string(text)
        if not text:
            return []
        try:
            # 改进的正则表达式，能正确处理各种数字格式
            pattern = r'[-+]?[±]?(\d*\.\d+|\d+\.?\d*)'
            matches = re.findall(pattern, text)
            return [float(match) for match in matches if match]
        except Exception as e:
            print(f"提取数字时出错: {e}, 文本: {text}")
            return []
    
    def preprocess_number_string(num_str):
        """
        预处理数字字符串，处理前导0的情况
        例如：050 -> 0.50, 025 -> 0.25
        """
        num_str = ensure_string(num_str)
        if not num_str:
            return num_str
            
        # 如果字符串以0开头且长度大于1，且没有小数点，则添加小数点
        if num_str.startswith('0') and len(num_str) > 1 and '.' not in num_str:
            # 检查是否已经是小数形式（如0.50）
            if not re.match(r'^\d+\.\d+$', num_str):
                # 将前导0后面的部分作为小数部分
                num_str = '0.' + num_str[1:]
        
        return num_str
    
    def extract_max_min_values(text):
        """
        从文本中提取MAX和MIN值（不区分大小写）
        返回格式: (max_value, min_value) 或 (None, None)
        """
        text = ensure_string(text)
        if not text:
            return None, None
            
        # 预处理文本，处理逗号作为小数点的情况
        text = preprocess_text(text)
        text_lower = text.lower()
        max_value = None
        min_value = None
        
        # 改进的正则表达式，能处理MAX/MIN在数字前后的情况
        # 情况1: MAX/MIN在数字前面 (如 "MAX 0.5", "Min.0.25")
        max_pattern_before = r'max[\.\s:：]*([-+]?\d*\.\d+|\d+\.?\d*)'
        min_pattern_before = r'min[\.\s:：]*([-+]?\d*\.\d+|\d+\.?\d*)'
        
        # 情况2: MAX/MIN在数字后面 (如 "0.5 MAX", "0.25 MIN")
        max_pattern_after = r'([-+]?\d*\.\d+|\d+\.?\d*)[\.\s:：]*max'
        min_pattern_after = r'([-+]?\d*\.\d+|\d+\.?\d*)[\.\s:：]*min'
        
        # 尝试匹配MAX在数字前面的情况
        max_match_before = re.search(max_pattern_before, text_lower)
        if max_match_before:
            try:
                # 提取匹配的数字部分
                num_str = max_match_before.group(1)
                # 预处理数字字符串，处理前导0的情况
                num_str = preprocess_number_string(num_str)
                # 处理特殊情况，如"Min.0.25"中的".0.25"
                if num_str.startswith('.') and len(num_str) > 1:
                    num_str = '0' + num_str
                max_value = float(num_str)
                print(f"检测到MAX在数字前: {max_match_before.group(0)} -> {num_str} -> {max_value}")
            except Exception as e:
                print(f"提取MAX值失败: {e}")
        else:
            # 如果MAX在数字前面没匹配到，尝试匹配MAX在数字后面的情况
            max_match_after = re.search(max_pattern_after, text_lower)
            if max_match_after:
                try:
                    # 提取匹配的数字部分
                    num_str = max_match_after.group(1)
                    # 预处理数字字符串，处理前导0的情况
                    num_str = preprocess_number_string(num_str)
                    # 处理特殊情况，如"Min.0.25"中的".0.25"
                    if num_str.startswith('.') and len(num_str) > 1:
                        num_str = '0' + num_str
                    max_value = float(num_str)
                    print(f"检测到MAX在数字后: {max_match_after.group(0)} -> {num_str} -> {max_value}")
                except Exception as e:
                    print(f"提取MAX值失败: {e}")
        
        # 尝试匹配MIN在数字前面的情况
        min_match_before = re.search(min_pattern_before, text_lower)
        if min_match_before:
            try:
                # 提取匹配的数字部分
                num_str = min_match_before.group(1)
                # 预处理数字字符串，处理前导0的情况
                num_str = preprocess_number_string(num_str)
                # 处理特殊情况，如"Min.0.25"中的".0.25"
                if num_str.startswith('.') and len(num_str) > 1:
                    num_str = '0' + num_str
                min_value = float(num_str)
                print(f"检测到MIN在数字前: {min_match_before.group(0)} -> {num_str} -> {min_value}")
            except Exception as e:
                print(f"提取MIN值失败: {e}")
        else:
            # 如果MIN在数字前面没匹配到，尝试匹配MIN在数字后面的情况
            min_match_after = re.search(min_pattern_after, text_lower)
            if min_match_after:
                try:
                    # 提取匹配的数字部分
                    num_str = min_match_after.group(1)
                    # 预处理数字字符串，处理前导0的情况
                    num_str = preprocess_number_string(num_str)
                    # 处理特殊情况，如"Min.0.25"中的".0.25"
                    if num_str.startswith('.') and len(num_str) > 1:
                        num_str = '0' + num_str
                    min_value = float(num_str)
                    print(f"检测到MIN在数字后: {min_match_after.group(0)} -> {num_str} -> {min_value}")
                except Exception as e:
                    print(f"提取MIN值失败: {e}")
        
        return max_value, min_value
    
    def is_predominantly_letters(text):
        """
        判断文本是否主要是字母（而不是数字）
        如果文本中字母字符的数量超过数字字符的数量，则认为是字母
        """
        if not text:
            return False
        
        letter_count = len(re.findall(r'[A-Za-z]', text))
        digit_count = len(re.findall(r'\d', text))
        
        return letter_count > digit_count
    
    def excel_column_to_number(col):
        """将Excel列名转换为数字（A=1, B=2, ..., Z=26, AA=27, AB=28, ...）"""
        num = 0
        for c in col.upper():
            if c < 'A' or c > 'Z':
                return 0
            num = num * 26 + (ord(c) - ord('A') + 1)
        return num
    
    def sort_excel_columns(columns):
        """按照Excel列名顺序排序"""
        return sorted(columns, key=lambda x: excel_column_to_number(x))
    
    def process_single_item(item):
        """处理单个字典项"""
        if not item or not isinstance(item, dict):
            return item
            
        # 复制数据避免修改原数据
        result_item = item.copy()
        
        # 获取类别
        absolutely_list = result_item.get('Absolutely', [])
        category = absolutely_list[0] if absolutely_list and len(absolutely_list) > 0 else 'other'
        
        # 根据类别处理
        if category == 'num':
            numbers = []
            
            # 获取OCR文本并确保是字符串
            ocr_strings = result_item.get('ocr_strings', '')
            ocr_text = ensure_string(ocr_strings)
            print(f"处理num类别文本: {ocr_text}")
            
            # 如果有小框，优先处理小框中的数字
            small_boxes = result_item.get('small_boxes', [])
            if small_boxes and len(small_boxes) > 0:
                # 如果有小框，我们可能需要处理每个小框的OCR结果
                # 但根据之前的设计，小框的OCR结果可能已经合并到主OCR字符串中
                # 所以这里我们主要从主OCR字符串中提取
                numbers.extend(extract_and_process_numbers(ocr_text))
            else:
                # 处理主OCR字符串
                numbers.extend(extract_and_process_numbers(ocr_text))
            
            # 检查是否包含±符号
            if '±' in ocr_text:
                pattern = r'([-+]?\d*\.\d+|\d+\.?\d*)\s*±\s*([-+]?\d*\.\d+|\d+\.?\d*)'
                match = re.search(pattern, ocr_text)
                if match:
                    try:
                        nom_str = match.group(1)
                        tolerance_str = match.group(2)
                        # 预处理数字字符串，处理前导0的情况
                        nom_str = preprocess_number_string(nom_str)
                        tolerance_str = preprocess_number_string(tolerance_str)
                        # 处理特殊情况，如".25"变为"0.25"
                        if nom_str.startswith('.') and len(nom_str) > 1:
                            nom_str = '0' + nom_str
                        if tolerance_str.startswith('.') and len(tolerance_str) > 1:
                            tolerance_str = '0' + tolerance_str
                            
                        nom = float(nom_str)
                        tolerance = float(tolerance_str)
                        max_val = nom + tolerance
                        min_val = nom - tolerance
                        result_item['max_nom_min'] = [max_val, nom, min_val]
                        print(f"检测到±符号: {nom} ± {tolerance}")
                        return result_item
                    except Exception as e:
                        print(f"处理±符号失败: {e}")
            
            # 如果没有±，检查是否有MAX和MIN关键字
            max_value, min_value = extract_max_min_values(ocr_text)
            
            if max_value is not None and min_value is not None:
                # 同时有MAX和MIN，计算平均值作为nom值
                nom_value = (max_value + min_value) / 2
                result_item['max_nom_min'] = [max_value, nom_value, min_value]
                print(f"检测到MAX和MIN: MAX={max_value}, MIN={min_value}, NOM={nom_value}")
            
            elif max_value is not None:
                # 只有MAX，只提取MAX值
                result_item['max_nom_min'] = [max_value]
                print(f"检测到MAX: {max_value}")
            
            elif min_value is not None:
                # 只有MIN，只提取MIN值
                result_item['max_nom_min'] = [min_value]
                print(f"检测到MIN: {min_value}")
            
            else:
                # 没有±、MAX、MIN，只提取纯数字，按由大到小排序
                if numbers:
                    # 注意：extract_and_process_numbers已经排序了
                    result_item['max_nom_min'] = numbers
                    print(f"提取到数字: {numbers}")
                else:
                    result_item['max_nom_min'] = []
                    print("未提取到任何数字")
                    
        elif category == 'angle':
            numbers = []
            ocr_strings = result_item.get('ocr_strings', '')
            ocr_text = ensure_string(ocr_strings)
            
            # 检查是否包含°符号
            if '°' in ocr_text:
                pattern = r'([-+]?\d*\.\d+|\d+\.?\d*)\s*°'
                matches = re.findall(pattern, ocr_text)
                numbers = [float(match) for match in matches if match]
            else:
                numbers = extract_and_process_numbers(ocr_text)
            
            result_item['max_nom_min'] = numbers
            
        elif category == 'PIN_num':
            # 处理PIN_num类别
            ocr_strings = result_item.get('ocr_strings', '')
            ocr_text = ensure_string(ocr_strings)
            print(f"处理PIN_num类别文本: {ocr_text}")
            
            # 首先判断文本是否主要是字母
            if is_predominantly_letters(ocr_text):
                # 如果是字母，按Excel列名顺序处理
                # 提取所有字母（包括多个字母组合如AF）
                letters = re.findall(r'[A-Za-z]+', ocr_text)
                if letters:
                    # 去重
                    unique_letters = list(set(letters))
                    
                    # 按照Excel列名顺序排序
                    sorted_letters = sort_excel_columns(unique_letters)
                    
                    if len(sorted_letters) >= 2:
                        # 如果有多个字母，取最大和最小
                        max_letter = sorted_letters[-1]  # 最大的字母
                        min_letter = sorted_letters[0]   # 最小的字母
                        result_item['max_nom_min'] = [max_letter, min_letter]
                        print(f"PIN_num字母: 最大={max_letter}, 最小={min_letter}")
                    else:
                        # 如果只有一个字母，重复使用它作为最大和最小值
                        single_letter = sorted_letters[0]
                        result_item['max_nom_min'] = [single_letter, single_letter]
                        print(f"PIN_num单个字母: {single_letter}")
                else:
                    result_item['max_nom_min'] = []
                    print("PIN_num: 未提取到字母")
            else:
                # 如果不是主要字母，则按数字处理
                numbers = extract_simple_numbers(ocr_text)
                if numbers:
                    # 提取最大最小值
                    max_val = max(numbers) if numbers else None
                    min_val = min(numbers) if numbers else None
                    if max_val is not None and min_val is not None:
                        result_item['max_nom_min'] = [max_val, min_val]
                        print(f"PIN_num数字: 最大值={max_val}, 最小值={min_val}")
                    else:
                        result_item['max_nom_min'] = []
                else:
                    result_item['max_nom_min'] = []
                    print("PIN_num: 未提取到数字")
            
        elif category == 'other':
            numbers = []
            ocr_strings = result_item.get('ocr_strings', '')
            ocr_text = ensure_string(ocr_strings)
            numbers = extract_and_process_numbers(ocr_text)
            result_item['max_nom_min'] = numbers
            
        else:
            # 未知类别，按other处理
            numbers = []
            ocr_strings = result_item.get('ocr_strings', '')
            ocr_text = ensure_string(ocr_strings)
            numbers = extract_and_process_numbers(ocr_text)
            result_item['max_nom_min'] = numbers
        
        return result_item
    
    def process_input_data(data):
        """处理输入数据，支持列表或单个字典"""
        if isinstance(data, list):
            # 如果是列表，处理每个元素
            return [process_single_item(item) for item in data]
        elif isinstance(data, dict):
            # 如果是单个字典，直接处理
            return process_single_item(data)
        else:
            # 其他类型原样返回
            return data
    
    # 处理每个输入变量
    results = {
        'new_other': process_input_data(new_other),
        'new_yolox_serial_num': process_input_data(new_yolox_serial_num),
        'angle_boxes_dicts': process_input_data(angle_boxes_dicts),
        'new_yolox_num': process_input_data(new_yolox_num)
    }
    
    # 处理BGA特有类型
    if BGA_serial_num is not None:
        results['BGA_serial_num'] = process_input_data(BGA_serial_num)
    
    if BGA_serial_letter is not None:
        results['BGA_serial_letter'] = process_input_data(BGA_serial_letter)
    
    return results
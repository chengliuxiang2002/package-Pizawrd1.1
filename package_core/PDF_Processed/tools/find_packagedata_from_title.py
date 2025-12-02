import re
import math

# 查找“x”的数量，若num=1，找到“x”前后紧邻的数字，若num=2，匹配数字x数字x数字
def find_x_and_numbers(text):
    # 匹配形如 数字 x 数字 的模式，允许有空格和小数
    match_1 = re.search(r'(\d+(?:\.\d+)?)\s*[x]\s*(\d+(?:\.\d+)?)', text)
    result = {}
    if match_1:
        n1 = match_1.group(1)
        n2 = match_1.group(2)
        # 检查“数字 x 数字”后是否紧跟“Exposed Pad”，允许有可选空格和可选mm
        after_pos = match_1.end()
        after_text = text[after_pos:]
        if re.match(r'\s*(mm)?\s*Exposed\s*Pad', after_text, re.IGNORECASE):
            return None
        # 检查是否有“-数字”紧跟其后
        match_height = re.match(r'\s*-\s*(\d+(?:\.\d+)?)', after_text)
        if match_height:
            n3 = match_height.group(1)
            if float(n1) == float(n2) or float(n1) > float(n2):
                result['length'] = float(n1)
                result['width'] = float(n2)
                result['height'] = float(n3)
            elif float(n1) < float(n2):
                result['length'] = float(n2)
                result['width'] = float(n1)
                result['height'] = float(n3)
        else:
            if float(n1) == float(n2) or float(n1) > float(n2):
                result['length'] = float(n1)
                result['width'] = float(n2)
            elif float(n1) < float(n2):
                result['length'] = float(n2)
                result['width'] = float(n1)
    match_2 = re.search(r'(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if match_2 :
        n1, n2, n3 = match_2.group(1), match_2.group(2), match_2.group(3)
        if float(n1) == float(n2) or float(n1) > float(n2):
            result['length'] = float(n1)
            result['width'] = float(n2)
            result['height'] = float(n3)
        elif float(n1) < float(n2):
            result['length'] = float(n2)
            result['width'] = float(n1)
            result['height'] = float(n3)
    # 检查“数字Body Thickness”并直接加入height
    match_body_thickness = re.search(r'(\d+(?:\.\d+)?)\s*Body\s*Thickness', text, re.IGNORECASE)
    if match_body_thickness:
        result['height'] = float(match_body_thickness.group(1))

    return result if result else None

# 只计算QFP横向和竖向pin数量
def calc_QFP_horizontal_and_vertical_pins(total_pin, length, width):
    if length and width :
        if length == width:
            horizontal_pin = total_pin / 4
            vertical_pin = total_pin /4
        else:
            vertical_pin = int(total_pin * width / (2 * (length + width)))
            horizontal_pin = total_pin / 2 - vertical_pin
    else:
        horizontal_pin = total_pin / 4
        vertical_pin = total_pin / 4

    return horizontal_pin,vertical_pin

# 只计算QFN横向和竖向pin数量
def calc_QFN_horizontal_and_vertical_pins(total_pin, length, width):
    if total_pin % 2 == 0:
        if length and width:
            if length == width:
                horizontal_pin = total_pin / 4
                vertical_pin = total_pin / 4
            elif float(length).is_integer() and float(width).is_integer():
                horizontal_pin = int(total_pin * width / (2 * (length + width)))
                vertical_pin = total_pin / 2 - horizontal_pin
            else:
                horizontal_pin = 2
                vertical_pin = total_pin / 2 - horizontal_pin
        else:
            horizontal_pin = total_pin / 4
            vertical_pin = total_pin / 4
    else:
        return None,None
    return horizontal_pin, vertical_pin

def calc_BGA_horizontal_and_vertical_pins(total_pin, length, width):
    if length == width:
        sqrt_val = math.isqrt(total_pin)
        if sqrt_val * sqrt_val == total_pin:
            horizontal_pin =  sqrt_val
            vertical_pin = sqrt_val
        else:
            return None,None
    else:
        return None, None
    return horizontal_pin, vertical_pin



def calc_DFN_SON_horizontal_and_vertical_pins(total_pin):
    if total_pin % 2 == 0:
        horizontal_pin = 2
        vertical_pin = total_pin / 2
    else:
        return None,None
    return horizontal_pin, vertical_pin


def clean_text(text):
    text = re.sub(r'×', 'x', text)
    text = re.sub(r'\*', 'x', text)
    text = re.sub(r'\', 'x', text)
    text = re.sub(r'(\d+(?:\.\d+)?)\s*mm', r'\1', text, flags=re.IGNORECASE)
    # 将数字X数字或数字X数字X数字中的大写X换成小写x
    text = re.sub(r'(\d+)\s*X\s*(\d+)', r'\1x\2', text)
    text = re.sub(r'(\d+)\s*X\s*(\d+)\s*X\s*(\d+)', r'\1x\2x\3', text)
    return text

def remove_dash_between_keyword_and_number(text, keywords=None):
    # 先处理关键字-数字 → 关键字数字
    for kw in keywords:
        # 只处理关键字-数字，且数字为整数（不含小数点）
        text = re.sub(rf'({kw})-(\d+)(?!\.)', rf'\1\2', text, flags=re.IGNORECASE)
    # 处理“数字-关键字”变为“关键字数字”，只允许连字符，不处理有空格的情况
    for kw in keywords:
        text = re.sub(rf'(\d+)-({kw})', rf'\2\1 ', text, flags=re.IGNORECASE)
    # 处理“数字L 关键字”变为“关键字数字”类型，如'32L QFP' -> 'QFP32'
    for kw in keywords:
        text = re.sub(rf'(\d+)L\s+{kw}', rf'{kw}\1', text, flags=re.IGNORECASE)
    # 如果已存在“关键字数字”形式，则不做处理
    for kw in keywords:
        if re.search(rf'{kw}+\d', text, re.IGNORECASE):
            return text
    # 只处理“数字 关键字”变为“关键字数字”，只允许空格
    def replace_even_number(match):
        num = match.group(1)
        kw = match.group(2)
        if  6 <= int(num) <= 500:
            return f'{kw}{num}'
        else:
            return match.group(0)
    for kw in keywords:
        text = re.sub(rf'(?<![\d.])(\d+)\s+({kw})(?!\.)', replace_even_number, text, flags=re.IGNORECASE)

    return text

def check_keywords_and_numbers(text):
    keywords = ['QFP', 'QFN', 'BGA', 'DFN', 'SON','SOP','LQFP','TQFP','MQFP','QFPN','Quad Flatpack']
    text = clean_text(text)
    text = remove_dash_between_keyword_and_number(text, keywords)
    keyword = None
    for kw in keywords:
        if kw in text:
            keyword = kw
            break
    # print(keyword)
    if keyword and re.search(r'\d+', text):
        # 检查是否存在带连字符或不带连字符的pin/lead
        if re.search(r'\d+\s*[-]?\s*(?:pin|lead|Pin|Lead|PIN|LEAD|Terminal|pad|Ball|balls)', text, re.IGNORECASE):
            if re.search(r'\d+\s*x\s*\d+', text, re.IGNORECASE):
                # 提取紧邻'-pin'、'-lead'、'-Lead'前的数字，支持带连字符和不带连字符
                match = re.search(r'(\d+)\s*[-]?\s*(?:pin|lead|Pin|Lead|PIN|LEAD|Terminal|pad|Ball|balls)', text, re.IGNORECASE)
                pin = match.group(1) if match else None
                dims = find_x_and_numbers(text)
                result = {'pin': pin}
                if dims:
                    result.update(dims)
                    # 计算横向和竖向pin数量
                    if pin and (keyword == 'QFP' or keyword == 'Quad Flat Package' or keyword =='Quad Flatpack'):
                        hpin, vpin = calc_QFP_horizontal_and_vertical_pins(int(pin), dims['length'], dims['width'])
                        result['horizontal_pin'] = hpin
                        result['vertical_pin'] = vpin
                    elif pin and (keyword == 'QFN' or keyword == 'QFPN'):
                        hpin, vpin = calc_QFN_horizontal_and_vertical_pins(int(pin), dims['length'], dims['width'])
                        result['horizontal_pin'] = hpin
                        result['vertical_pin'] = vpin
                    elif pin and 'length' in dims and 'width' in dims and (keyword == 'SON' or keyword == 'DFN'):
                        hpin, vpin = calc_DFN_SON_horizontal_and_vertical_pins(int(pin))
                        result['horizontal_pin'] = hpin
                        result['vertical_pin'] = vpin
                    # elif pin and 'length' in dims and 'width' in dims and (keyword == 'BGA'):
                    #     hpin, vpin = calc_BGA_horizontal_and_vertical_pins(int(pin), dims['length'], dims['width'])
                    #     result['horizontal_pin'] = hpin
                    #     result['vertical_pin'] = vpin
                return result
            else:
                # 提取紧邻'-pin'、'-lead'、'-Lead'前的数字，支持带连字符和不带连字符
                match = re.search(r'(\d+)\s*[-]?\s*(?:pin|lead|Pin|Lead|PIN|LEAD|Terminal|pad|Ball|balls)', text, re.IGNORECASE)
                pin = match.group(1) if match else None
                result = {'pin': pin}
                if pin and (keyword == 'SON' or keyword == 'DFN'):
                    hpin, vpin = calc_DFN_SON_horizontal_and_vertical_pins(int(pin))
                    result['horizontal_pin'] = hpin
                    result['vertical_pin'] = vpin
                elif pin and (keyword == 'QFP' or keyword == 'Quad Flat Package' or keyword =='Quad Flatpack'):
                    hpin, vpin = calc_QFP_horizontal_and_vertical_pins(int(pin),length=None,width=None)
                    result['horizontal_pin'] = hpin
                    result['vertical_pin'] = vpin
                elif pin and (keyword == 'QFN' ):
                    hpin, vpin = calc_QFN_horizontal_and_vertical_pins(int(pin),length=None,width=None)
                    result['horizontal_pin'] = hpin
                    result['vertical_pin'] = vpin
                return result

        else:
            if re.search(r'\d+\s*x\s*\d+', text, re.IGNORECASE):
                # 提取紧邻关键字后的数字（无空格）
                for kw in keywords:
                    m = re.search(rf'{kw}\s*(\d+)', text, re.IGNORECASE)
                    if m:
                        pin = m.group(1)
                        break
                else:
                    pin = None
                dims = find_x_and_numbers(text)
                result = {'pin': pin}
                if dims:
                    result.update(dims)
                    if pin and (keyword == 'QFP' or keyword == 'Quad Flat Package'or keyword =='Quad Flatpack'):
                        hpin, vpin = calc_QFP_horizontal_and_vertical_pins(int(pin), dims['length'], dims['width'])
                        result['horizontal_pin'] = hpin
                        result['vertical_pin'] = vpin
                    elif pin and (keyword == 'QFN' ):
                        hpin, vpin = calc_QFN_horizontal_and_vertical_pins(int(pin), dims['length'], dims['width'])
                        result['horizontal_pin'] = hpin
                        result['vertical_pin'] = vpin
                    elif pin and 'length' in dims and 'width' in dims and (keyword == 'SON' or keyword == 'DFN'):
                        hpin, vpin = calc_DFN_SON_horizontal_and_vertical_pins(int(pin))
                        result['horizontal_pin'] = hpin
                        result['vertical_pin'] = vpin
                    # elif pin and 'length' in dims and 'width' in dims and (keyword == 'BGA'):
                    #     hpin, vpin = calc_BGA_horizontal_and_vertical_pins(int(pin), dims['length'], dims['width'])
                    #     result['horizontal_pin'] = hpin
                    #     result['vertical_pin'] = vpin
                return result
            else:
                # 提取紧邻关键字后的数字（无空格）
                for kw in keywords:
                    m = re.search(rf'{kw}\s*(\d+)', text, re.IGNORECASE)
                    if m:
                        pin = m.group(1)
                        break
                else:
                    pin = None
                result = {'pin': pin}
                if pin and (keyword == 'SON' or keyword == 'DFN'):
                    hpin, vpin = calc_DFN_SON_horizontal_and_vertical_pins(int(pin))
                    result['horizontal_pin'] = hpin
                    result['vertical_pin'] = vpin
                elif pin and (keyword == 'QFP' or keyword == 'Quad Flat Package'or keyword =='Quad Flatpack'):
                    hpin, vpin = calc_QFP_horizontal_and_vertical_pins(int(pin),length=None,width=None)
                    result['horizontal_pin'] = hpin
                    result['vertical_pin'] = vpin
                elif pin and (keyword == 'QFN' ):
                    hpin, vpin = calc_QFN_horizontal_and_vertical_pins(int(pin),length=None,width=None)
                    result['horizontal_pin'] = hpin
                    result['vertical_pin'] = vpin
                return result

    else:
        return {}

def clean_result(result):
    """
    对check_keywords_and_numbers得到的result进行数据清理：
    """
    def clean_pin(pin_val):
        try:
            pin = int(pin_val)
            if  6 <= pin <= 500:
                return pin
            else:
                return None
        except (TypeError, ValueError):
            return None
    def clean_height(height_val):
        try:
            h = float(height_val)
            if h < 5:
                return h
            else:
                return None
        except (TypeError, ValueError):
            return None
    def to_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    cleaned = dict(result)
    cleaned['pin'] = clean_pin(result.get('pin'))
    cleaned['height'] = clean_height(result.get('height'))
    for k in ['length', 'width', 'horizontal_pin', 'vertical_pin']:
        if k in cleaned:
            cleaned[k] = to_float(result.get(k))
    return cleaned


if __name__ == "__main__":
    text = '100-Pin Thin Plastic Quad Flatpack (14 x 20 x 1.4 mm), 51-85050'
    result = check_keywords_and_numbers(text)
    cleaned = clean_result(result)
    # print(result)
    print(cleaned)


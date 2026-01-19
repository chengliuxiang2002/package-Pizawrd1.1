"""
文字方向分类器 - 使用ONNX模型
支持检测图像中文字的方向：0°, 90°, 180°, 270°
"""

import numpy as np
import onnxruntime as ort
from PIL import Image

from package_core.PackageExtract.yolox_onnx_py.model_paths import model_path


class OrientationClassifier:
    LABELS = ["0°", "180°", "90°", "270°"]

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = (64, 64)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """预处理图像"""
        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 调整大小
        image = image.resize(self.input_size, Image.BILINEAR)

        # 转换为numpy数组并归一化
        img_array = np.array(image, dtype=np.float32) / 255.0

        # 标准化 (ImageNet均值和标准差)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std

        # HWC -> CHW
        img_array = img_array.transpose(2, 0, 1)

        # 添加batch维度
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image: Image.Image) -> tuple[str, float, list[float]]:
        """
        预测图像方向

        返回:
            - orientation: 方向标签 ("0°", "90°", "180°", "270°")
            - confidence: 置信度
            - probabilities: 各方向的概率
        """
        # 预处理
        input_tensor = self.preprocess(image)

        # 推理
        outputs = self.session.run(None, {self.input_name: input_tensor})
        logits = outputs[0][0]

        # Softmax计算概率
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()

        # 获取预测结果
        pred_idx = np.argmax(probabilities)
        orientation = self.LABELS[pred_idx]
        confidence = float(probabilities[pred_idx])

        return orientation, confidence, probabilities.tolist()

    def predict_and_correct(self, image: Image.Image) -> tuple[Image.Image, str, float]:
        """
        预测方向并自动校正图像

        返回:
            - corrected_image: 校正后的图像
            - original_orientation: 原始方向
            - confidence: 置信度
        """
        orientation, confidence, _ = self.predict(image)

        # 根据检测到的方向进行旋转校正
        rotation_map = {
            "0°": 0,
            "180°": 180,
            "90°": -90,
            "270°": 90
        }

        rotation_angle = rotation_map[orientation]
        if rotation_angle != 0:
            corrected = image.rotate(rotation_angle, expand=True)
        else:
            corrected = image.copy()

        return corrected, orientation, confidence


def main():
    import os

    # ===== 在这里修改配置 =====
    image_path = r"D:\HuaweiMoveData\Users\LNQ\xwechat_files\wxid_6s02ovglnbms22_6e06\msg\file\2026-01\onnx_orientation\onnx_orientation\img.png"  # 修改为你的图片路径
    do_correct = True        # 是否自动校正并保存
    # ==========================

    # 模型路径
    onnx_model_path = model_path("orientation_model","resnet_orientation.onnx")
    # 初始化分类器
    print("加载模型...")
    classifier = OrientationClassifier(onnx_model_path)
    print("模型加载完成!")

    # 加载图像
    print(f"\n处理图像: {image_path}")
    image = Image.open(image_path)

    # 预测
    orientation, confidence, probs = classifier.predict(image)

    print(f"\n检测结果:")
    print(f"  方向: {orientation}")
    print(f"  置信度: {confidence:.2%}")

    # 校正图像
    # if do_correct and orientation != "0°":
    #     corrected, _, _ = classifier.predict_and_correct(image)
    #     base, ext = os.path.splitext(image_path)
    #     output_path = f"{base}_corrected{ext}"
    #     corrected.save(output_path)
    #     print(f"\n保存校正后的图像: {output_path}")


if __name__ == "__main__":
    main()

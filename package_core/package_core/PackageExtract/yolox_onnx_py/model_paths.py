"""提供统一的模型路径解析工具，确保不同模块可在任意工作目录下找到 ONNX 资源。"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Iterable

def _get_repo_root() -> Path:
    """获取项目根目录，兼容开发环境和打包后环境。"""
    # PyInstaller 打包后的临时目录
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # 打包后，模型文件应该在 exe 同目录的 model 文件夹中
        return Path(sys.executable).parent
    
    # 开发环境：从当前位置回溯到项目根目录
    # model_paths.py 位于 package_core/PackageExtract/yolox_onnx_py/model_paths.py
    # 需要回溯 3 级：yolox_onnx_py -> PackageExtract -> package_core -> 根目录
    return Path(__file__).resolve().parents[3]

def get_project_root() -> Path:
    """获取项目根目录路径（兼容开发环境和打包后环境）。
    
    这是统一的根目录获取函数，所有需要访问 Result 目录的代码都应该使用此函数。
    """
    return _get_repo_root()

# 项目根目录
_REPO_ROOT = _get_repo_root()

# 模型主目录以及常用子目录
MODEL_ROOT = _REPO_ROOT / "model"
OCR_MODEL_ROOT = MODEL_ROOT / "ocr_model"
YOLO_MODEL_ROOT = MODEL_ROOT / "yolo_model"


def _join_path(base: Path, parts: Iterable[str]) -> Path:
    """在给定基础目录上拼接路径片段。"""
    return base.joinpath(*parts)


def model_path(*parts: str) -> str:
    """返回 ``model`` 目录下资源的绝对路径字符串。"""
    return str(_join_path(MODEL_ROOT, parts))


def ocr_model_path(*parts: str) -> str:
    """返回 ``model/ocr_model`` 子目录下资源的绝对路径字符串。"""
    return str(_join_path(OCR_MODEL_ROOT, parts))


def yolo_model_path(*parts: str) -> str:
    """返回 ``model/yolo_model`` 子目录下资源的绝对路径字符串。"""
    return str(_join_path(YOLO_MODEL_ROOT, parts))

def result_path(*parts: str) -> str:
    """返回 ``Result`` 目录下资源的绝对路径字符串。"""
    return str(_REPO_ROOT / "Result" / Path(*parts))

if __name__ == '__main__':
    print(model_path("ocr_model", "ocr_rec.onnx"))
    print(ocr_model_path("ocr_rec.onnx"))
    print(yolo_model_path("yolov5s.onnx"))
    print(result_path("Package_extract", "data"))
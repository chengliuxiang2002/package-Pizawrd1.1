import shutil
import os

try:
    from package_core.PackageExtract.yolox_onnx_py.model_paths import result_path
except ModuleNotFoundError:
    from pathlib import Path
    def result_path(*parts):
        return str(Path(__file__).resolve().parents[2] / 'Result' / Path(*parts))

TEMP_DIRECTORY = result_path('temp')  # 临时存放pdf文件夹
PDF_NAME = os.path.join(TEMP_DIRECTORY, 'detect.pdf')
PDF_NAME_MINI = os.path.join(TEMP_DIRECTORY, 'detect_mini.pdf')
def remove_dir(dir_path):
    """
        删除dir_path文件夹（包括其所有子文件夹及文件）
    """
    shutil.rmtree(dir_path)

def create_dir(dir_path):
    """
        创建dir_path空文件夹（若存在该文件夹则清空该文件夹）
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
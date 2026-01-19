"""全局模型缓存模块

该模块提供 ONNX/YOLO 模型的全局缓存机制，避免重复加载模型带来的性能损失。
每个模型只在首次使用时加载，后续调用直接返回缓存的实例。

使用方法:
    from package_core.PackageExtract.yolox_onnx_py.model_cache import get_onnx_session, get_yolo_model

    # 获取 ONNX Session（带缓存）
    session = get_onnx_session(model_path)

    # 获取 YOLO 模型（带缓存）
    model = get_yolo_model(model_path)
"""

import onnxruntime
from threading import Lock

# ==================== 全局缓存存储 ====================
_ONNX_SESSION_CACHE = {}
_YOLO_MODEL_CACHE = {}
_cache_lock = Lock()  # 线程安全锁


def get_onnx_session(model_path: str, providers: list = None) -> onnxruntime.InferenceSession:
    """获取缓存的 ONNX InferenceSession

    Args:
        model_path: ONNX 模型文件路径
        providers: 执行提供者列表，默认使用 CPU

    Returns:
        缓存的 InferenceSession 实例
    """
    global _ONNX_SESSION_CACHE

    # 规范化路径作为缓存键
    import os
    cache_key = os.path.normpath(os.path.abspath(model_path))

    with _cache_lock:
        if cache_key not in _ONNX_SESSION_CACHE:
            print(f"[模型缓存] 首次加载 ONNX 模型: {os.path.basename(model_path)}")

            if providers is None:
                providers = ['CPUExecutionProvider']

            # 创建优化的 Session 配置
            sess_options = onnxruntime.SessionOptions()
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 1
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True

            _ONNX_SESSION_CACHE[cache_key] = onnxruntime.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            print(f"[模型缓存] ONNX 模型已缓存: {os.path.basename(model_path)}")

        return _ONNX_SESSION_CACHE[cache_key]


def get_yolo_model(model_path: str):
    """获取缓存的 YOLO 模型

    Args:
        model_path: YOLO 模型文件路径

    Returns:
        缓存的 YOLO 模型实例
    """
    global _YOLO_MODEL_CACHE

    import os
    cache_key = os.path.normpath(os.path.abspath(model_path))

    with _cache_lock:
        if cache_key not in _YOLO_MODEL_CACHE:
            print(f"[模型缓存] 首次加载 YOLO 模型: {os.path.basename(model_path)}")

            from ultralytics import YOLO
            _YOLO_MODEL_CACHE[cache_key] = YOLO(model_path)

            print(f"[模型缓存] YOLO 模型已缓存: {os.path.basename(model_path)}")

        return _YOLO_MODEL_CACHE[cache_key]


def clear_cache():
    """清空所有模型缓存（用于测试或内存释放）"""
    global _ONNX_SESSION_CACHE, _YOLO_MODEL_CACHE

    with _cache_lock:
        _ONNX_SESSION_CACHE.clear()
        _YOLO_MODEL_CACHE.clear()
        print("[模型缓存] 所有缓存已清空")


def get_cache_stats():
    """获取缓存统计信息"""
    return {
        "onnx_sessions": len(_ONNX_SESSION_CACHE),
        "yolo_models": len(_YOLO_MODEL_CACHE),
        "onnx_keys": list(_ONNX_SESSION_CACHE.keys()),
        "yolo_keys": list(_YOLO_MODEL_CACHE.keys()),
    }

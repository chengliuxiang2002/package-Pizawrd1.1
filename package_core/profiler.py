"""
性能分析模块 - PackageWizard 时间统计系统

用于统计：
1. 步骤用时 - 各主要处理步骤的执行时间
2. 函数用时 - 函数调用次数和总时间
3. 模型推理时间 - DETR/YOLO/DBNet/SVTR等模型的推理时间
4. 函数调用栈分析 - 调用层级和热点函数识别
5. 模型推理详细指标 - 吞吐量、内存使用等

使用方法：
    from package_core.profiler import profiler, step_timer, func_timer, model_timer

    # 步骤计时
    with step_timer("PDF预处理"):
        do_preprocess()

    # 函数装饰器
    @func_timer
    def my_function():
        pass

    # 模型推理计时（带详细信息）
    with model_timer("DETR", {"input_size": "640x640", "batch_size": 1}):
        model.predict()

    # 函数调用栈追踪
    @call_tracker
    def traced_function():
        pass

    # 生成报告
    profiler.generate_report("performance_report.json")
"""

import time
import json
import threading
import traceback
import sys
from contextlib import contextmanager
from functools import wraps
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import os


@dataclass
class TimingRecord:
    """单次计时记录"""
    name: str
    duration: float  # 秒
    timestamp: str
    category: str  # 'step', 'function', 'model'
    extra_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedStats:
    """聚合统计信息"""
    name: str
    category: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0

    def update(self, duration: float):
        self.call_count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.avg_time = self.total_time / self.call_count


@dataclass
class ModelInferenceStats:
    """模型推理详细统计"""
    model_name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    # 详细指标
    total_samples: int = 0  # 处理的样本总数
    throughput: float = 0.0  # 样本/秒
    preprocess_time: float = 0.0  # 预处理时间
    inference_time: float = 0.0  # 纯推理时间
    postprocess_time: float = 0.0  # 后处理时间
    input_sizes: List[str] = field(default_factory=list)  # 输入尺寸记录

    def update(self, duration: float, batch_size: int = 1,
               preprocess: float = 0.0, inference: float = 0.0,
               postprocess: float = 0.0, input_size: str = None):
        self.call_count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.avg_time = self.total_time / self.call_count
        self.total_samples += batch_size
        self.throughput = self.total_samples / self.total_time if self.total_time > 0 else 0
        self.preprocess_time += preprocess
        self.inference_time += inference
        self.postprocess_time += postprocess
        if input_size and input_size not in self.input_sizes:
            self.input_sizes.append(input_size)


@dataclass
class FunctionCallRecord:
    """函数调用记录（包含调用栈）"""
    func_name: str
    duration: float
    timestamp: str
    caller: str  # 调用者函数名
    call_stack: List[str]  # 完整调用栈
    args_info: str = ""  # 参数摘要信息


class CallGraphNode:
    """调用图节点"""
    def __init__(self, name: str):
        self.name = name
        self.call_count = 0
        self.total_time = 0.0
        self.self_time = 0.0  # 自身耗时（排除子调用）
        self.children: Dict[str, 'CallGraphNode'] = {}
        self.callers: Dict[str, int] = {}  # 调用者 -> 调用次数

    def add_call(self, duration: float, caller: str = None):
        self.call_count += 1
        self.total_time += duration
        if caller:
            self.callers[caller] = self.callers.get(caller, 0) + 1

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'call_count': self.call_count,
            'total_time': round(self.total_time, 4),
            'self_time': round(self.self_time, 4),
            'avg_time': round(self.total_time / self.call_count, 4) if self.call_count > 0 else 0,
            'callers': self.callers,
            'children': [child.to_dict() for child in self.children.values()]
        }


class PerformanceProfiler:
    """性能分析器主类"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.records: List[TimingRecord] = []
        self.stats: Dict[str, AggregatedStats] = {}
        self.step_stack: List[str] = []  # 用于嵌套步骤
        self.session_start: Optional[float] = None
        self.session_info: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self.enabled = True

        # 新增：模型推理详细统计
        self.model_stats: Dict[str, ModelInferenceStats] = {}
        # 新增：函数调用记录
        self.function_calls: List[FunctionCallRecord] = []
        # 新增：调用图
        self.call_graph: Dict[str, CallGraphNode] = {}
        # 新增：当前调用栈（用于追踪嵌套调用）
        self._call_stack: List[str] = []
        # 新增：热点函数阈值
        self.hotspot_threshold = 0.05  # 占总时间5%以上视为热点

    def start_session(self, info: Optional[Dict[str, Any]] = None):
        """开始一个新的分析会话"""
        self.records = []
        self.stats = {}
        self.step_stack = []
        self.session_start = time.time()
        self.session_info = info or {}
        self.session_info['start_time'] = datetime.now().isoformat()
        # 重置新增的统计
        self.model_stats = {}
        self.function_calls = []
        self.call_graph = {}
        self._call_stack = []

    def end_session(self):
        """结束当前会话"""
        if self.session_start:
            self.session_info['end_time'] = datetime.now().isoformat()
            self.session_info['total_duration'] = time.time() - self.session_start

    def record(self, name: str, duration: float, category: str,
               extra_info: Optional[Dict[str, Any]] = None):
        """记录一次计时"""
        if not self.enabled:
            return

        with self._lock:
            record = TimingRecord(
                name=name,
                duration=duration,
                timestamp=datetime.now().isoformat(),
                category=category,
                extra_info=extra_info or {}
            )
            self.records.append(record)

            # 更新统计
            key = f"{category}:{name}"
            if key not in self.stats:
                self.stats[key] = AggregatedStats(name=name, category=category)
            self.stats[key].update(duration)

    def record_model_inference(self, model_name: str, duration: float,
                               batch_size: int = 1,
                               preprocess_time: float = 0.0,
                               inference_time: float = 0.0,
                               postprocess_time: float = 0.0,
                               input_size: str = None,
                               extra_info: Optional[Dict[str, Any]] = None):
        """记录模型推理（带详细信息）"""
        if not self.enabled:
            return

        with self._lock:
            # 记录到通用记录
            info = extra_info or {}
            info.update({
                'batch_size': batch_size,
                'preprocess_time': preprocess_time,
                'inference_time': inference_time,
                'postprocess_time': postprocess_time,
                'input_size': input_size
            })

            record = TimingRecord(
                name=model_name,
                duration=duration,
                timestamp=datetime.now().isoformat(),
                category='model',
                extra_info=info
            )
            self.records.append(record)

            # 更新通用统计
            key = f"model:{model_name}"
            if key not in self.stats:
                self.stats[key] = AggregatedStats(name=model_name, category='model')
            self.stats[key].update(duration)

            # 更新模型详细统计
            if model_name not in self.model_stats:
                self.model_stats[model_name] = ModelInferenceStats(model_name=model_name)
            self.model_stats[model_name].update(
                duration=duration,
                batch_size=batch_size,
                preprocess=preprocess_time,
                inference=inference_time,
                postprocess=postprocess_time,
                input_size=input_size
            )

    def record_function_call(self, func_name: str, duration: float,
                             caller: str = None, call_stack: List[str] = None,
                             args_info: str = ""):
        """记录函数调用（带调用栈）"""
        if not self.enabled:
            return

        with self._lock:
            # 记录函数调用
            record = FunctionCallRecord(
                func_name=func_name,
                duration=duration,
                timestamp=datetime.now().isoformat(),
                caller=caller or "unknown",
                call_stack=call_stack or [],
                args_info=args_info
            )
            self.function_calls.append(record)

            # 更新通用统计
            key = f"function:{func_name}"
            if key not in self.stats:
                self.stats[key] = AggregatedStats(name=func_name, category='function')
            self.stats[key].update(duration)

            # 更新调用图
            if func_name not in self.call_graph:
                self.call_graph[func_name] = CallGraphNode(func_name)
            self.call_graph[func_name].add_call(duration, caller)

            # 更新调用者的children
            if caller and caller in self.call_graph:
                if func_name not in self.call_graph[caller].children:
                    self.call_graph[caller].children[func_name] = self.call_graph[func_name]

    def push_call_stack(self, func_name: str):
        """入栈：记录当前调用"""
        self._call_stack.append(func_name)

    def pop_call_stack(self) -> Optional[str]:
        """出栈：弹出当前调用"""
        if self._call_stack:
            return self._call_stack.pop()
        return None

    def get_current_caller(self) -> Optional[str]:
        """获取当前调用者"""
        if len(self._call_stack) >= 1:
            return self._call_stack[-1]
        return None

    def get_call_stack(self) -> List[str]:
        """获取当前调用栈副本"""
        return self._call_stack.copy()

    def get_step_stats(self) -> List[Dict]:
        """获取步骤统计"""
        return [
            {
                'name': s.name,
                'call_count': s.call_count,
                'total_time': round(s.total_time, 4),
                'avg_time': round(s.avg_time, 4),
                'min_time': round(s.min_time, 4) if s.min_time != float('inf') else 0,
                'max_time': round(s.max_time, 4),
            }
            for key, s in self.stats.items() if s.category == 'step'
        ]

    def get_function_stats(self) -> List[Dict]:
        """获取函数统计"""
        result = [
            {
                'name': s.name,
                'call_count': s.call_count,
                'total_time': round(s.total_time, 4),
                'avg_time': round(s.avg_time, 4),
                'min_time': round(s.min_time, 4) if s.min_time != float('inf') else 0,
                'max_time': round(s.max_time, 4),
            }
            for key, s in self.stats.items() if s.category == 'function'
        ]
        # 按总时间降序排序
        return sorted(result, key=lambda x: x['total_time'], reverse=True)

    def get_model_stats(self) -> List[Dict]:
        """获取模型推理统计"""
        result = [
            {
                'name': s.name,
                'call_count': s.call_count,
                'total_time': round(s.total_time, 4),
                'avg_time': round(s.avg_time, 4),
                'min_time': round(s.min_time, 4) if s.min_time != float('inf') else 0,
                'max_time': round(s.max_time, 4),
            }
            for key, s in self.stats.items() if s.category == 'model'
        ]
        return sorted(result, key=lambda x: x['total_time'], reverse=True)

    def get_model_detailed_stats(self) -> List[Dict]:
        """获取模型推理详细统计（包含吞吐量、预处理/推理/后处理耗时分解）"""
        result = []
        for name, stats in self.model_stats.items():
            result.append({
                'name': name,
                'call_count': stats.call_count,
                'total_time': round(stats.total_time, 4),
                'avg_time': round(stats.avg_time, 4),
                'min_time': round(stats.min_time, 4) if stats.min_time != float('inf') else 0,
                'max_time': round(stats.max_time, 4),
                'total_samples': stats.total_samples,
                'throughput': round(stats.throughput, 2),  # 样本/秒
                'preprocess_time': round(stats.preprocess_time, 4),
                'inference_time': round(stats.inference_time, 4),
                'postprocess_time': round(stats.postprocess_time, 4),
                'input_sizes': stats.input_sizes,
                # 计算各阶段占比
                'preprocess_pct': round(stats.preprocess_time / stats.total_time * 100, 1) if stats.total_time > 0 else 0,
                'inference_pct': round(stats.inference_time / stats.total_time * 100, 1) if stats.total_time > 0 else 0,
                'postprocess_pct': round(stats.postprocess_time / stats.total_time * 100, 1) if stats.total_time > 0 else 0,
            })
        return sorted(result, key=lambda x: x['total_time'], reverse=True)

    def get_function_call_analysis(self) -> Dict:
        """获取函数调用分析（包含调用关系和热点识别）"""
        total_time = sum(s.total_time for s in self.stats.values() if s.category == 'function')

        # 热点函数识别
        hotspots = []
        for key, s in self.stats.items():
            if s.category == 'function':
                time_pct = s.total_time / total_time if total_time > 0 else 0
                if time_pct >= self.hotspot_threshold:
                    hotspots.append({
                        'name': s.name,
                        'call_count': s.call_count,
                        'total_time': round(s.total_time, 4),
                        'time_percentage': round(time_pct * 100, 2),
                        'avg_time': round(s.avg_time, 4),
                    })
        hotspots.sort(key=lambda x: x['total_time'], reverse=True)

        # 调用频率最高的函数
        most_called = sorted(
            [{'name': s.name, 'call_count': s.call_count, 'total_time': round(s.total_time, 4)}
             for s in self.stats.values() if s.category == 'function'],
            key=lambda x: x['call_count'],
            reverse=True
        )[:20]

        # 调用者-被调用者关系统计
        caller_callee_pairs = defaultdict(int)
        for record in self.function_calls:
            if record.caller and record.caller != "unknown":
                pair = f"{record.caller} -> {record.func_name}"
                caller_callee_pairs[pair] += 1

        top_call_pairs = sorted(
            [{'pair': k, 'count': v} for k, v in caller_callee_pairs.items()],
            key=lambda x: x['count'],
            reverse=True
        )[:30]

        # 调用图摘要
        call_graph_summary = [node.to_dict() for node in self.call_graph.values()
                              if node.call_count > 0][:50]

        return {
            'total_function_time': round(total_time, 4),
            'total_function_calls': len(self.function_calls),
            'unique_functions': len([s for s in self.stats.values() if s.category == 'function']),
            'hotspots': hotspots,
            'most_called': most_called,
            'top_call_pairs': top_call_pairs,
            'call_graph_summary': call_graph_summary,
        }

    def get_hotspot_functions(self, top_n: int = 10) -> List[Dict]:
        """获取热点函数列表"""
        total_time = sum(s.total_time for s in self.stats.values() if s.category == 'function')

        result = []
        for key, s in self.stats.items():
            if s.category == 'function':
                time_pct = s.total_time / total_time if total_time > 0 else 0
                result.append({
                    'name': s.name,
                    'call_count': s.call_count,
                    'total_time': round(s.total_time, 4),
                    'avg_time': round(s.avg_time, 4),
                    'time_percentage': round(time_pct * 100, 2),
                    'is_hotspot': time_pct >= self.hotspot_threshold,
                })
        result.sort(key=lambda x: x['total_time'], reverse=True)
        return result[:top_n]

    def get_summary(self) -> Dict:
        """获取汇总统计"""
        step_total = sum(s.total_time for s in self.stats.values() if s.category == 'step')
        func_total = sum(s.total_time for s in self.stats.values() if s.category == 'function')
        model_total = sum(s.total_time for s in self.stats.values() if s.category == 'model')

        # 计算模型推理详细汇总
        total_samples = sum(s.total_samples for s in self.model_stats.values())
        avg_throughput = total_samples / model_total if model_total > 0 else 0

        return {
            'session_info': self.session_info,
            'total_records': len(self.records),
            'step_count': len([s for s in self.stats.values() if s.category == 'step']),
            'step_total_time': round(step_total, 4),
            'function_count': len([s for s in self.stats.values() if s.category == 'function']),
            'function_total_time': round(func_total, 4),
            'function_call_count': len(self.function_calls),
            'model_count': len([s for s in self.stats.values() if s.category == 'model']),
            'model_total_time': round(model_total, 4),
            'model_total_samples': total_samples,
            'model_avg_throughput': round(avg_throughput, 2),
            'hotspot_count': len(self.get_hotspot_functions()),
        }

    def generate_report(self, output_path: Optional[str] = None) -> Dict:
        """生成完整报告"""
        self.end_session()

        report = {
            'summary': self.get_summary(),
            'steps': self.get_step_stats(),
            'functions': self.get_function_stats(),
            'models': self.get_model_stats(),
            'models_detailed': self.get_model_detailed_stats(),
            'function_analysis': self.get_function_call_analysis(),
            'hotspots': self.get_hotspot_functions(20),
            'timeline': [
                {
                    'name': r.name,
                    'category': r.category,
                    'duration': round(r.duration, 4),
                    'timestamp': r.timestamp,
                    **r.extra_info
                }
                for r in self.records
            ]
        }

        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"性能报告已保存至: {output_path}")

        return report

    def print_summary(self):
        """打印简要统计到控制台"""
        print("\n" + "="*70)
        print("性能分析报告")
        print("="*70)

        summary = self.get_summary()
        print(f"\n会话信息:")
        print(f"  开始时间: {summary['session_info'].get('start_time', 'N/A')}")
        print(f"  总耗时: {summary['session_info'].get('total_duration', 0):.2f}秒")

        print(f"\n步骤统计 (共{summary['step_count']}个步骤):")
        print("-"*70)
        print(f"{'步骤名称':<30} {'次数':>6} {'总时间(秒)':>12} {'平均(秒)':>10}")
        print("-"*70)
        for s in self.get_step_stats():
            print(f"{s['name']:<30} {s['call_count']:>6} {s['total_time']:>12.4f} {s['avg_time']:>10.4f}")

        print(f"\n模型推理统计 (共{summary['model_count']}个模型):")
        print("-"*70)
        print(f"{'模型名称':<20} {'次数':>6} {'总时间(秒)':>10} {'平均(秒)':>8} {'样本数':>8} {'吞吐量':>10}")
        print("-"*70)
        for m in self.get_model_detailed_stats():
            print(f"{m['name']:<20} {m['call_count']:>6} {m['total_time']:>10.4f} "
                  f"{m['avg_time']:>8.4f} {m['total_samples']:>8} {m['throughput']:>8.2f}/s")

        # 显示模型耗时分解
        detailed_stats = self.get_model_detailed_stats()
        if detailed_stats:
            print(f"\n模型耗时分解:")
            print("-"*70)
            print(f"{'模型名称':<20} {'预处理':>12} {'推理':>12} {'后处理':>12} {'其他':>12}")
            print("-"*70)
            for m in detailed_stats:
                other_pct = 100 - m['preprocess_pct'] - m['inference_pct'] - m['postprocess_pct']
                print(f"{m['name']:<20} {m['preprocess_pct']:>10.1f}% {m['inference_pct']:>10.1f}% "
                      f"{m['postprocess_pct']:>10.1f}% {other_pct:>10.1f}%")

        print(f"\n热点函数 (Top 10):")
        print("-"*70)
        print(f"{'函数名称':<40} {'次数':>6} {'总时间(秒)':>10} {'占比':>8}")
        print("-"*70)
        for f in self.get_hotspot_functions(10):
            name = f['name'][:38] + '..' if len(f['name']) > 40 else f['name']
            hotspot_mark = "🔥" if f['is_hotspot'] else "  "
            print(f"{hotspot_mark}{name:<38} {f['call_count']:>6} {f['total_time']:>10.4f} {f['time_percentage']:>6.1f}%")

        print(f"\n函数调用统计 (Top 20):")
        print("-"*70)
        print(f"{'函数名称':<40} {'次数':>6} {'总时间(秒)':>12}")
        print("-"*70)
        for f in self.get_function_stats()[:20]:
            name = f['name'][:38] + '..' if len(f['name']) > 40 else f['name']
            print(f"{name:<40} {f['call_count']:>6} {f['total_time']:>12.4f}")

        print("="*70 + "\n")


# 全局单例
profiler = PerformanceProfiler()


@contextmanager
def step_timer(step_name: str, extra_info: Optional[Dict[str, Any]] = None):
    """步骤计时上下文管理器

    Args:
        step_name: 步骤名称
        extra_info: 额外信息

    Usage:
        with step_timer("PDF预处理"):
            preprocess()
    """
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        profiler.record(step_name, duration, 'step', extra_info)
        print(f"[Profiler] 步骤 '{step_name}' 完成, 耗时: {duration:.4f}秒")


@contextmanager
def model_timer(model_name: str, extra_info: Optional[Dict[str, Any]] = None):
    """模型推理计时上下文管理器

    Args:
        model_name: 模型名称 (如 DETR, YOLO, DBNet, SVTR)
        extra_info: 额外信息 (如输入尺寸, batch size等)

    Usage:
        with model_timer("DETR", {"input_size": "640x640"}):
            model.predict(image)
    """
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        profiler.record(model_name, duration, 'model', extra_info)


@contextmanager
def model_timer_detailed(model_name: str, batch_size: int = 1,
                         input_size: str = None,
                         extra_info: Optional[Dict[str, Any]] = None):
    """模型推理详细计时上下文管理器（支持分阶段计时）

    Args:
        model_name: 模型名称 (如 DETR, YOLO, DBNet, SVTR)
        batch_size: 批次大小
        input_size: 输入尺寸描述 (如 "640x640")
        extra_info: 额外信息

    Usage:
        with model_timer_detailed("DETR", batch_size=1, input_size="640x640") as timer:
            timer.mark_preprocess_start()
            preprocessed = preprocess(image)
            timer.mark_preprocess_end()

            timer.mark_inference_start()
            output = model.predict(preprocessed)
            timer.mark_inference_end()

            timer.mark_postprocess_start()
            result = postprocess(output)
            timer.mark_postprocess_end()
    """
    timer = DetailedModelTimer(model_name, batch_size, input_size, extra_info)
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


class DetailedModelTimer:
    """详细模型计时器，支持预处理/推理/后处理分阶段计时"""

    def __init__(self, model_name: str, batch_size: int = 1,
                 input_size: str = None, extra_info: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.input_size = input_size
        self.extra_info = extra_info or {}

        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

        self._preprocess_start: Optional[float] = None
        self._preprocess_end: Optional[float] = None
        self._inference_start: Optional[float] = None
        self._inference_end: Optional[float] = None
        self._postprocess_start: Optional[float] = None
        self._postprocess_end: Optional[float] = None

    def start(self):
        self._start_time = time.time()

    def stop(self):
        self._end_time = time.time()
        self._record()

    def mark_preprocess_start(self):
        self._preprocess_start = time.time()

    def mark_preprocess_end(self):
        self._preprocess_end = time.time()

    def mark_inference_start(self):
        self._inference_start = time.time()

    def mark_inference_end(self):
        self._inference_end = time.time()

    def mark_postprocess_start(self):
        self._postprocess_start = time.time()

    def mark_postprocess_end(self):
        self._postprocess_end = time.time()

    def _record(self):
        if self._start_time is None or self._end_time is None:
            return

        total_duration = self._end_time - self._start_time
        preprocess_time = 0.0
        inference_time = 0.0
        postprocess_time = 0.0

        if self._preprocess_start and self._preprocess_end:
            preprocess_time = self._preprocess_end - self._preprocess_start
        if self._inference_start and self._inference_end:
            inference_time = self._inference_end - self._inference_start
        if self._postprocess_start and self._postprocess_end:
            postprocess_time = self._postprocess_end - self._postprocess_start

        profiler.record_model_inference(
            model_name=self.model_name,
            duration=total_duration,
            batch_size=self.batch_size,
            preprocess_time=preprocess_time,
            inference_time=inference_time,
            postprocess_time=postprocess_time,
            input_size=self.input_size,
            extra_info=self.extra_info
        )


def func_timer(func=None, *, name: str = None):
    """函数计时装饰器

    Args:
        func: 被装饰的函数
        name: 自定义名称, 默认使用函数名

    Usage:
        @func_timer
        def my_function():
            pass

        @func_timer(name="自定义名称")
        def another_function():
            pass
    """
    def decorator(fn):
        fn_name = name or f"{fn.__module__}.{fn.__qualname__}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                duration = time.time() - start
                profiler.record(fn_name, duration, 'function')

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def call_tracker(func=None, *, name: str = None, track_args: bool = False):
    """带调用栈追踪的函数计时装饰器

    Args:
        func: 被装饰的函数
        name: 自定义名称, 默认使用函数名
        track_args: 是否记录参数信息

    Usage:
        @call_tracker
        def my_function():
            pass

        @call_tracker(track_args=True)
        def another_function(x, y):
            pass
    """
    def decorator(fn):
        fn_name = name or f"{fn.__module__}.{fn.__qualname__}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # 获取调用者信息
            caller = profiler.get_current_caller()
            call_stack = profiler.get_call_stack()

            # 记录参数信息
            args_info = ""
            if track_args:
                args_repr = [repr(a)[:50] for a in args[:3]]  # 只记录前3个参数
                kwargs_repr = [f"{k}={repr(v)[:30]}" for k, v in list(kwargs.items())[:3]]
                args_info = ", ".join(args_repr + kwargs_repr)

            # 入栈
            profiler.push_call_stack(fn_name)

            start = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                duration = time.time() - start
                # 记录函数调用
                profiler.record_function_call(
                    func_name=fn_name,
                    duration=duration,
                    caller=caller,
                    call_stack=call_stack,
                    args_info=args_info
                )
                # 出栈
                profiler.pop_call_stack()

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def auto_profile_class(cls):
    """类装饰器：自动为类的所有公共方法添加计时

    Usage:
        @auto_profile_class
        class MyModel:
            def predict(self, x):
                pass
    """
    for attr_name in dir(cls):
        if attr_name.startswith('_'):
            continue
        attr = getattr(cls, attr_name)
        if callable(attr):
            setattr(cls, attr_name, call_tracker(attr, name=f"{cls.__name__}.{attr_name}"))
    return cls


class TimerContext:
    """可复用的计时器类

    Usage:
        timer = TimerContext("模型加载")
        timer.start()
        load_model()
        timer.stop()
        print(f"耗时: {timer.elapsed}秒")
    """

    def __init__(self, name: str, category: str = 'step', auto_record: bool = True):
        self.name = name
        self.category = category
        self.auto_record = auto_record
        self._start: Optional[float] = None
        self._elapsed: float = 0.0

    def start(self):
        self._start = time.time()
        return self

    def stop(self):
        if self._start is not None:
            self._elapsed = time.time() - self._start
            if self.auto_record:
                profiler.record(self.name, self._elapsed, self.category)
            self._start = None
        return self._elapsed

    @property
    def elapsed(self) -> float:
        if self._start is not None:
            return time.time() - self._start
        return self._elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def generate_html_report(report: Dict, output_path: str):
    """生成HTML格式的可视化报告"""

    html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PackageWizard 性能分析报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f7fa; color: #333; line-height: 1.6; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 3px solid #3498db; }}
        h2 {{ color: #34495e; margin: 30px 0 15px; display: flex; align-items: center; gap: 10px; }}
        h3 {{ color: #5d6d7e; margin: 20px 0 10px; font-size: 16px; }}
        .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); transition: transform 0.2s; }}
        .card:hover {{ transform: translateY(-2px); }}
        .card-title {{ font-size: 13px; color: #7f8c8d; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .card-value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
        .card-unit {{ font-size: 14px; color: #95a5a6; }}
        .card-subtitle {{ font-size: 12px; color: #95a5a6; margin-top: 5px; }}
        .card.step {{ border-left: 4px solid #3498db; }}
        .card.model {{ border-left: 4px solid #e74c3c; }}
        .card.function {{ border-left: 4px solid #27ae60; }}
        .card.hotspot {{ border-left: 4px solid #f39c12; }}
        table {{ width: 100%; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        th {{ background: #34495e; color: white; font-weight: 600; font-size: 13px; text-transform: uppercase; }}
        tr:hover {{ background: #f8f9fa; }}
        tr:last-child td {{ border-bottom: none; }}
        .bar-container {{ width: 100%; background: #ecf0f1; border-radius: 3px; height: 8px; }}
        .bar {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
        .bar.step {{ background: linear-gradient(90deg, #3498db, #2980b9); }}
        .bar.model {{ background: linear-gradient(90deg, #e74c3c, #c0392b); }}
        .bar.function {{ background: linear-gradient(90deg, #27ae60, #1e8449); }}
        .bar.hotspot {{ background: linear-gradient(90deg, #f39c12, #d68910); }}
        .section {{ margin-bottom: 40px; }}
        .timestamp {{ color: #7f8c8d; font-size: 12px; text-align: center; margin-top: 30px; }}
        .hotspot-badge {{ background: #f39c12; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 5px; }}
        .grid-2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
        .phase-bar {{ display: flex; height: 24px; border-radius: 4px; overflow: hidden; margin: 5px 0; }}
        .phase-bar div {{ display: flex; align-items: center; justify-content: center; font-size: 11px; color: white; }}
        .phase-preprocess {{ background: #3498db; }}
        .phase-inference {{ background: #e74c3c; }}
        .phase-postprocess {{ background: #27ae60; }}
        .phase-other {{ background: #95a5a6; }}
        .legend {{ display: flex; gap: 20px; margin: 10px 0; font-size: 12px; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .legend-color {{ width: 12px; height: 12px; border-radius: 2px; }}
        .tabs {{ display: flex; gap: 5px; margin-bottom: 20px; }}
        .tab {{ padding: 10px 20px; background: #ecf0f1; border: none; border-radius: 5px 5px 0 0; cursor: pointer; font-size: 14px; }}
        .tab.active {{ background: white; font-weight: 600; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        @media (max-width: 768px) {{
            .grid-2 {{ grid-template-columns: 1fr; }}
            .summary-cards {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PackageWizard 性能分析报告</h1>

        <div class="summary-cards">
            <div class="card step">
                <div class="card-title">步骤总数</div>
                <div class="card-value">{step_count}</div>
            </div>
            <div class="card step">
                <div class="card-title">步骤总耗时</div>
                <div class="card-value">{step_time:.2f}<span class="card-unit">秒</span></div>
            </div>
            <div class="card model">
                <div class="card-title">模型推理次数</div>
                <div class="card-value">{model_count}</div>
            </div>
            <div class="card model">
                <div class="card-title">模型推理总耗时</div>
                <div class="card-value">{model_time:.2f}<span class="card-unit">秒</span></div>
            </div>
            <div class="card model">
                <div class="card-title">平均吞吐量</div>
                <div class="card-value">{model_throughput:.1f}<span class="card-unit">样本/秒</span></div>
            </div>
            <div class="card function">
                <div class="card-title">函数调用数</div>
                <div class="card-value">{func_count}</div>
            </div>
            <div class="card hotspot">
                <div class="card-title">热点函数</div>
                <div class="card-value">{hotspot_count}</div>
                <div class="card-subtitle">&gt;5%总耗时</div>
            </div>
            <div class="card function">
                <div class="card-title">总运行时间</div>
                <div class="card-value">{total_time:.2f}<span class="card-unit">秒</span></div>
            </div>
        </div>

        <div class="section">
            <h2>步骤耗时统计</h2>
            <table>
                <thead>
                    <tr>
                        <th>步骤名称</th>
                        <th>调用次数</th>
                        <th>总耗时(秒)</th>
                        <th>平均耗时(秒)</th>
                        <th>占比</th>
                    </tr>
                </thead>
                <tbody>
                    {step_rows}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>模型推理详细统计</h2>
            <table>
                <thead>
                    <tr>
                        <th>模型名称</th>
                        <th>调用次数</th>
                        <th>样本数</th>
                        <th>总耗时(秒)</th>
                        <th>平均(秒)</th>
                        <th>吞吐量</th>
                        <th>耗时分解</th>
                    </tr>
                </thead>
                <tbody>
                    {model_detailed_rows}
                </tbody>
            </table>
            <div class="legend">
                <div class="legend-item"><div class="legend-color phase-preprocess"></div>预处理</div>
                <div class="legend-item"><div class="legend-color phase-inference"></div>推理</div>
                <div class="legend-item"><div class="legend-color phase-postprocess"></div>后处理</div>
                <div class="legend-item"><div class="legend-color phase-other"></div>其他</div>
            </div>
        </div>

        <div class="section">
            <h2>热点函数分析</h2>
            <p style="color: #7f8c8d; margin-bottom: 15px;">以下函数占用了超过5%的总执行时间，是性能优化的重点目标</p>
            <table>
                <thead>
                    <tr>
                        <th>函数名称</th>
                        <th>调用次数</th>
                        <th>总耗时(秒)</th>
                        <th>平均耗时(秒)</th>
                        <th>时间占比</th>
                    </tr>
                </thead>
                <tbody>
                    {hotspot_rows}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>函数调用耗时统计 (Top 30)</h2>
            <table>
                <thead>
                    <tr>
                        <th>函数名称</th>
                        <th>调用次数</th>
                        <th>总耗时(秒)</th>
                        <th>平均耗时(秒)</th>
                        <th>占比</th>
                    </tr>
                </thead>
                <tbody>
                    {func_rows}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>高频调用函数 (Top 20)</h2>
            <table>
                <thead>
                    <tr>
                        <th>函数名称</th>
                        <th>调用次数</th>
                        <th>总耗时(秒)</th>
                    </tr>
                </thead>
                <tbody>
                    {most_called_rows}
                </tbody>
            </table>
        </div>

        <p class="timestamp">报告生成时间: {timestamp}</p>
    </div>
</body>
</html>
    """

    summary = report['summary']
    step_time = summary.get('step_total_time', 0)
    model_time = summary.get('model_total_time', 0)
    func_time = summary.get('function_total_time', 0)
    total_time = summary.get('session_info', {}).get('total_duration', step_time + model_time)
    model_throughput = summary.get('model_avg_throughput', 0)
    hotspot_count = summary.get('hotspot_count', 0)

    def make_rows(items, category, max_time):
        rows = []
        for item in items:
            pct = (item['total_time'] / max_time * 100) if max_time > 0 else 0
            rows.append(f"""
                <tr>
                    <td>{item['name']}</td>
                    <td>{item['call_count']}</td>
                    <td>{item['total_time']:.4f}</td>
                    <td>{item['avg_time']:.4f}</td>
                    <td>
                        <div class="bar-container">
                            <div class="bar {category}" style="width: {min(pct, 100)}%"></div>
                        </div>
                        {pct:.1f}%
                    </td>
                </tr>
            """)
        return '\n'.join(rows)

    def make_model_detailed_rows(items):
        rows = []
        for item in items:
            pre_pct = item.get('preprocess_pct', 0)
            inf_pct = item.get('inference_pct', 0)
            post_pct = item.get('postprocess_pct', 0)
            other_pct = max(0, 100 - pre_pct - inf_pct - post_pct)

            phase_bar = f"""
                <div class="phase-bar">
                    <div class="phase-preprocess" style="width: {pre_pct}%">{pre_pct:.0f}%</div>
                    <div class="phase-inference" style="width: {inf_pct}%">{inf_pct:.0f}%</div>
                    <div class="phase-postprocess" style="width: {post_pct}%">{post_pct:.0f}%</div>
                    <div class="phase-other" style="width: {other_pct}%">{other_pct:.0f}%</div>
                </div>
            """

            rows.append(f"""
                <tr>
                    <td>{item['name']}</td>
                    <td>{item['call_count']}</td>
                    <td>{item.get('total_samples', item['call_count'])}</td>
                    <td>{item['total_time']:.4f}</td>
                    <td>{item['avg_time']:.4f}</td>
                    <td>{item.get('throughput', 0):.2f}/s</td>
                    <td style="min-width: 200px">{phase_bar}</td>
                </tr>
            """)
        return '\n'.join(rows)

    def make_hotspot_rows(items):
        rows = []
        for item in items:
            is_hotspot = item.get('is_hotspot', False)
            badge = '<span class="hotspot-badge">HOTSPOT</span>' if is_hotspot else ''
            pct = item.get('time_percentage', 0)
            rows.append(f"""
                <tr>
                    <td>{item['name']}{badge}</td>
                    <td>{item['call_count']}</td>
                    <td>{item['total_time']:.4f}</td>
                    <td>{item['avg_time']:.4f}</td>
                    <td>
                        <div class="bar-container">
                            <div class="bar hotspot" style="width: {min(pct, 100)}%"></div>
                        </div>
                        {pct:.1f}%
                    </td>
                </tr>
            """)
        return '\n'.join(rows)

    def make_most_called_rows(items):
        rows = []
        for item in items:
            rows.append(f"""
                <tr>
                    <td>{item['name']}</td>
                    <td>{item['call_count']}</td>
                    <td>{item['total_time']:.4f}</td>
                </tr>
            """)
        return '\n'.join(rows)

    step_rows = make_rows(report.get('steps', []), 'step', step_time)

    # 使用详细的模型统计（如果有）
    models_detailed = report.get('models_detailed', report.get('models', []))
    model_detailed_rows = make_model_detailed_rows(models_detailed)

    func_rows = make_rows(report.get('functions', [])[:30], 'function', func_time)

    # 热点函数
    hotspots = report.get('hotspots', [])
    hotspot_rows = make_hotspot_rows(hotspots)

    # 高频调用函数
    func_analysis = report.get('function_analysis', {})
    most_called = func_analysis.get('most_called', [])[:20]
    most_called_rows = make_most_called_rows(most_called)

    func_calls = summary.get('function_call_count', sum(f['call_count'] for f in report.get('functions', [])))

    html = html_template.format(
        step_count=summary.get('step_count', 0),
        step_time=step_time,
        model_count=sum(m['call_count'] for m in report.get('models', [])),
        model_time=model_time,
        model_throughput=model_throughput,
        func_count=func_calls,
        hotspot_count=hotspot_count,
        total_time=total_time,
        step_rows=step_rows or '<tr><td colspan="5">暂无数据</td></tr>',
        model_detailed_rows=model_detailed_rows or '<tr><td colspan="7">暂无数据</td></tr>',
        hotspot_rows=hotspot_rows or '<tr><td colspan="5">暂无热点函数</td></tr>',
        func_rows=func_rows or '<tr><td colspan="5">暂无数据</td></tr>',
        most_called_rows=most_called_rows or '<tr><td colspan="3">暂无数据</td></tr>',
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML报告已保存至: {output_path}")


if __name__ == '__main__':
    # 测试代码
    profiler.start_session({'test': True})

    with step_timer("测试步骤1"):
        time.sleep(0.1)

    with step_timer("测试步骤2"):
        time.sleep(0.2)

    # 测试简单的模型计时
    with model_timer("DETR"):
        time.sleep(0.05)

    with model_timer("YOLO"):
        time.sleep(0.03)

    # 测试详细的模型计时（包含分阶段）
    with model_timer_detailed("DBNet", batch_size=2, input_size="640x480") as timer:
        timer.mark_preprocess_start()
        time.sleep(0.01)  # 模拟预处理
        timer.mark_preprocess_end()

        timer.mark_inference_start()
        time.sleep(0.03)  # 模拟推理
        timer.mark_inference_end()

        timer.mark_postprocess_start()
        time.sleep(0.005)  # 模拟后处理
        timer.mark_postprocess_end()

    # 测试函数计时装饰器
    @func_timer
    def test_func():
        time.sleep(0.01)

    for _ in range(5):
        test_func()

    # 测试带调用栈追踪的装饰器
    @call_tracker
    def outer_func():
        time.sleep(0.01)
        inner_func()

    @call_tracker
    def inner_func():
        time.sleep(0.005)

    for _ in range(3):
        outer_func()

    profiler.print_summary()
    report = profiler.generate_report("test_performance_report.json")
    generate_html_report(report, "test_performance_report.html")

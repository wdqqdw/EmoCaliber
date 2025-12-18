import os
import re
import json
import wandb
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.calibration import calibration_curve

def load_jsonl_file(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行JSON解析错误: {e}")
                continue
    
    print(f"成功加载 {len(data)} 条数据")
    return data

def evaluate_response(ground_truth: str, response: str) -> bool:
    """评估模型响应是否正确 (不区分大小写的正则匹配)"""
    # 标准化ground_truth (去掉标点、转为小写)
    truth_normalized = re.sub(r'[^\w]', '', ground_truth.lower())
    
    # 检查是否在response中找到 (不区分大小写)
    pattern = re.compile(r'\b' + re.escape(truth_normalized) + r'\b', re.IGNORECASE)
    return bool(pattern.search(response))

def calculate_metrics_by_task(data: List[Dict]) -> Tuple[Dict, Dict]:
    """按task分组计算评估指标"""
    task_data = {}
    
    # 按task分组
    for item in data:
        task = item.get('task', 'unknown')
        if task not in task_data:
            task_data[task] = []
        task_data[task].append(item)
    
    # 计算每个task的指标
    task_metrics = {}
    for task, items in task_data.items():
        truths = []
        preds = []
        for item in items:
            truth = item.get('ground_truth', '').lower()
            response = item.get('response', '').lower()
            candidates = item.get('candidates')
            
            # 尝试从<answer>标签中提取预测结果
            answer_content = None
            answer_match = re.search(r'<answer>(.*?)</answer>', response)
            if answer_match:
                answer_content = answer_match.group(1).lower()
            
            # 优先检查<answer>标签中的内容
            pred = "unknown"
            
            # 如果有<answer>标签，先检查其中的内容
            if answer_content:
                for emotion in candidates:
                    if emotion.lower() in answer_content:
                        pred = emotion.lower()
                        break
            else:            
                for emotion in candidates:
                    if emotion.lower() in response:
                        pred = emotion.lower()
                        break
            
            truths.append(truth)
            preds.append(pred)
        
        # 计算指标
        acc = accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted')
        
        task_metrics[task] = {
            "accuracy": acc,
            "f1_score": f1,
            "total_samples": len(items),
            "correct_samples": sum(1 for t, p in zip(truths, preds) if t == p)
        }
    
    # 计算总体指标
    all_truths = []
    all_preds = []
    for item in data:
        truth = item.get('ground_truth', '').lower()
        response = item.get('response', '').lower()
        candidates = item.get('candidates')
        
        found = False
        for emotion in candidates:
            if emotion in response:
                pred = emotion
                found = True
                break
        
        if not found:
            pred = "unknown"
        
        all_truths.append(truth)
        all_preds.append(pred)
    
    overall_metrics = {
        "accuracy": accuracy_score(all_truths, all_preds),
        "f1_score": f1_score(all_truths, all_preds, average='weighted'),
        "total_samples": len(data),
        "correct_samples": sum(1 for t, p in zip(all_truths, all_preds) if t == p)
    }
    
    return task_metrics, overall_metrics

def calculate_confidence_metrics_by_group(data: List[Dict]) -> Tuple[List[str], List[List[str]]]:
    """按三大任务组计算不同置信度阈值下的指标"""
    
    # 定义任务分组
    task_groups = {
        'ID VSA': ['FI-2', 'WebEmo-2'],
        'ID VER': ['FI-8', 'EmoSet-8', 'WebEmo-7', 'WebEmo-25'],
        'OOD VER': ['Abstract-8', 'Artphoto-8', 'UnbiasedEmo-6']
    }
    
    # 按组分类数据
    group_data = {group_name: [] for group_name in task_groups.keys()}
    for item in data:
        task = item.get('task', 'unknown')
        for group_name, tasks in task_groups.items():
            if task in tasks:
                group_data[group_name].append(item)
                break
    
    # 置信度阈值
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # 准备结果
    excel_headers = []
    excel_data = []
    
    # 生成表头
    for group_name in ['ID VSA', 'ID VER', 'OOD VER']:
        excel_headers.extend([f"{group_name} Acc", f"{group_name} F1", f"{group_name} Coverage"])
    
    # 为每个阈值计算指标
    for threshold in thresholds:
        row_data = []
        
        for group_name in ['ID VSA', 'ID VER', 'OOD VER']:
            items = group_data[group_name]
            total_samples = len(items)
            
            if total_samples == 0:
                row_data.extend(["N/A", "N/A", "N/A"])
                continue
            
            # 筛选置信度 >= 阈值的样本
            high_conf_items = []
            for item in items:
                confidence = item.get('confidence')
                if confidence is not None and confidence >= threshold:
                    high_conf_items.append(item)
            
            coverage = len(high_conf_items) / total_samples if total_samples > 0 else 0
            
            if len(high_conf_items) > 0:
                # 计算高置信度样本的准确率和F1
                truths = []
                preds = []
                for item in high_conf_items:
                    truth = item.get('ground_truth', '').lower()
                    response = item.get('response', '').lower()
                    candidates = item.get('candidates')
                    
                    # 预测逻辑
                    pred = "unknown"
                    answer_content = None
                    answer_match = re.search(r'<answer>(.*?)</answer>', response)
                    if answer_match:
                        answer_content = answer_match.group(1).lower()
                    
                    if answer_content:
                        for emotion in candidates:
                            if emotion.lower() in answer_content:
                                pred = emotion.lower()
                                break
                    else:            
                        for emotion in candidates:
                            if emotion.lower() in response:
                                pred = emotion.lower()
                                break
                    
                    truths.append(truth)
                    preds.append(pred)
                
                acc = accuracy_score(truths, preds) if truths else 0
                f1 = f1_score(truths, preds, average='weighted') if truths else 0
            else:
                acc = 0
                f1 = 0
            
            # 转换为百分比并保留2位小数
            acc_percent = acc * 100
            f1_percent = f1 * 100
            coverage_percent = coverage * 100
            
            row_data.extend([f"{acc_percent:.2f}", f"{f1_percent:.2f}", f"{coverage_percent:.2f}"])
        
        excel_data.append(row_data)
    
    return excel_headers, excel_data

def plot_discrete_curves_by_group(data: List[Dict], output_path: str):
    """按三大任务组绘制三个离散曲线图"""
    
    # 定义任务分组
    task_groups = {
        'ID VSA': ['FI-2', 'WebEmo-2'],
        'ID VER': ['FI-8', 'EmoSet-8', 'WebEmo-7', 'WebEmo-25'],
        'OOD VER': ['Abstract-8', 'Artphoto-8', 'UnbiasedEmo-6']
    }
    
    colors = {'ID VSA': 'blue', 'ID VER': 'green', 'OOD VER': 'red'}
    
    # 按组分类数据
    group_data = {group_name: [] for group_name in task_groups.keys()}
    for item in data:
        task = item.get('task', 'unknown')
        for group_name, tasks in task_groups.items():
            if task in tasks:
                group_data[group_name].append(item)
                break
    
    # 置信度阈值
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # 创建三个子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 为每个组计算数据
    for group_name, color in colors.items():
        items = group_data[group_name]
        if len(items) == 0:
            continue
            
        accuracies = []
        coverages = []
        
        # 为每个阈值计算指标
        for threshold in thresholds:
            # 筛选置信度 >= 阈值的样本
            high_conf_items = [item for item in items 
                             if item.get('confidence') is not None 
                             and item.get('confidence') >= threshold]
            
            coverage = len(high_conf_items) / len(items) if len(items) > 0 else 0
            
            if len(high_conf_items) > 0:
                # 计算准确率
                truths = []
                preds = []
                for item in high_conf_items:
                    truth = item.get('ground_truth', '').lower()
                    response = item.get('response', '').lower()
                    candidates = item.get('candidates')
                    
                    pred = "unknown"
                    answer_content = None
                    answer_match = re.search(r'<answer>(.*?)</answer>', response)
                    if answer_match:
                        answer_content = answer_match.group(1).lower()
                    
                    if answer_content:
                        for emotion in candidates:
                            if emotion.lower() in answer_content:
                                pred = emotion.lower()
                                break
                    else:            
                        for emotion in candidates:
                            if emotion.lower() in response:
                                pred = emotion.lower()
                                break
                    
                    truths.append(truth)
                    preds.append(pred)
                
                acc = accuracy_score(truths, preds) if truths else 0
            else:
                acc = 0
            
            accuracies.append(acc * 100)  # 转换为百分比
            coverages.append(coverage * 100)  # 转换为百分比
        
        # 绘制置信度-准确率曲线
        ax1.plot(thresholds, accuracies, 'o-', label=group_name, color=color, linewidth=2, markersize=6)
        
        # 绘制置信度-覆盖率曲线
        ax2.plot(thresholds, coverages, 'o-', label=group_name, color=color, linewidth=2, markersize=6)
        
        # 绘制准确率-覆盖率曲线
        ax3.plot(coverages, accuracies, 'o-', label=group_name, color=color, linewidth=2, markersize=6)
    
    # 设置第一个子图：置信度-准确率
    ax1.set_xlabel('Confidence Threshold', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy vs Confidence Threshold (by Group)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.05, 0.95)
    ax1.set_ylim(0, 100)
    
    # 设置第二个子图：置信度-覆盖率
    ax2.set_xlabel('Confidence Threshold', fontsize=12)
    ax2.set_ylabel('Coverage (%)', fontsize=12)
    ax2.set_title('Coverage vs Confidence Threshold (by Group)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.05, 0.95)
    ax2.set_ylim(0, 100)
    
    # 设置第三个子图：准确率-覆盖率
    ax3.set_xlabel('Coverage (%)', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_title('Accuracy vs Coverage (by Group)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 100)
    
    # 添加图例
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # 保存图像
    plot_path = os.path.join(output_path, 'discrete_curves_by_group.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"分组离散曲线图已保存到: {plot_path}")

def print_metrics_table(task_metrics: Dict, overall_metrics: Dict) -> Tuple[str, str, str]:
    """以表格形式打印评估指标，并返回Excel格式行"""
    # 定义固定的task顺序
    TASK_ORDER = [
        'Abstract-8', 'Artphoto-8', 'EmoSet-8', 'FI-2', 'FI-8',
        'UnbiasedEmo-6', 'WebEmo-2', 'WebEmo-7', 'WebEmo-25'
    ]
    
    # 准备表头
    header = "| {:<15} | {:<8} | {:<8} | {:<9} |".format(
        "Task", "Acc", "F1", "#Samples"
    )
    separator = "-" * len(header)
    
    # 准备表格内容（按照固定顺序）
    rows = []
    excel_headers = []
    excel_values = []
    
    # 生成Excel表头和数据
    for task in TASK_ORDER:
        if task in task_metrics:
            metrics = task_metrics[task]
            # 转换为百分比并保留2位小数
            acc_percent = metrics["accuracy"] * 100
            f1_percent = metrics["f1_score"] * 100
            
            # 表格行
            row = "| {:<15} | {:>6.2f} | {:>6.2f} | {:<9} |".format(
                task,
                acc_percent,
                f1_percent,
                metrics["total_samples"]
            )
            rows.append(row)
            
            # Excel格式
            excel_headers.extend([f"{task} Acc", f"{task} F1"])
            excel_values.extend([f"{acc_percent:.2f}", f"{f1_percent:.2f}"])
    
    # 添加总体统计行
    overall_acc_percent = overall_metrics["accuracy"] * 100
    overall_f1_percent = overall_metrics["f1_score"] * 100
    
    total_row = "| {:<15} | {:>6.2f} | {:>6.2f} | {:<9} |".format(
        "Overall",
        overall_acc_percent,
        overall_f1_percent,
        overall_metrics["total_samples"]
    )
    
    # 添加总体指标到Excel格式
    excel_headers.extend(["Overall Acc", "Overall F1"])
    excel_values.extend([f"{overall_acc_percent:.2f}", f"{overall_f1_percent:.2f}"])
    
    # 组合表格所有部分
    table = "\n".join([separator, header, separator] + rows + [separator, total_row, separator])
    
    # 生成Excel格式行
    excel_header_line = "\t".join(excel_headers)
    excel_value_line = "\t".join(excel_values)
    
    return table, excel_header_line, excel_value_line

def calculate_ece(confidences: List[float], accuracies: List[bool], n_bins: int = 10) -> float:
    """计算Expected Calibration Error (ECE)"""
    # 将置信度和准确率转换为numpy数组
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    # 分箱
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找到在当前箱子中的样本
        in_bin = np.logical_and(confidences >= bin_lower, confidences < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            # 计算箱子内的平均置信度和准确率
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            
            # 累加到ECE
            ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin
    
    return ece

def calculate_brier_score(confidences: List[float], accuracies: List[bool]) -> float:
    """计算Brier Score"""
    confidences = np.array(confidences)
    accuracies = np.array(accuracies).astype(float)  # 将bool转换为0/1
    
    # 裁剪置信度到 [0, 1] 范围内
    confidences = np.clip(confidences, 0, 1)
    
    return np.mean((confidences - accuracies) ** 2)

def calculate_roc_auc(confidences: List[float], accuracies: List[bool]) -> float:
    """
    计算ROC AUC指标
    
    参数:
        confidences: 置信度列表
        accuracies: 准确率列表 (True/False)
    
    返回:
        ROC AUC值，计算失败时返回None
    """
    confidences = np.array(confidences)
    accuracies = np.array(accuracies).astype(float)
    
    # 裁剪置信度到 [0, 1] 范围内
    confidences = np.clip(confidences, 0, 1)
    
    # 确保有至少2个类别
    unique_classes = np.unique(accuracies)
    if len(unique_classes) < 2:
        print(f"警告: ROC AUC计算需要至少两个类别")
        return None
    
    try:
        # 计算ROC AUC
        roc_auc = roc_auc_score(accuracies, confidences)
        return roc_auc
    except Exception as e:
        print(f"警告: ROC AUC计算失败: {e}")
        return None

def calculate_group_calibration_metrics(data: List[Dict]) -> Dict[str, Dict]:
    """计算三组数据的ECE、Brier Score和ROC AUC"""
    
    # 定义分组
    task_groups = {
        'ID VSA': ['FI-2', 'WebEmo-2'],
        'ID VER': ['FI-8', 'EmoSet-8', 'WebEmo-7', 'WebEmo-25'],
        'OOD VER': ['Abstract-8', 'Artphoto-8', 'UnbiasedEmo-6']
    }
    
    group_data = {group_name: [] for group_name in task_groups.keys()}
    
    # 按组分类数据
    for item in data:
        task = item.get('task', 'unknown')
        for group_name, tasks in task_groups.items():
            if task in tasks:
                group_data[group_name].append(item)
                break
    
    group_metrics = {}
    
    for group_name, items in group_data.items():
        if len(items) == 0:
            print(f"警告: {group_name} 组没有数据")
            group_metrics[group_name] = {
                'ece': None,
                'brier_score': None,
                'roc_auc': None,
                'total_samples': 0,
                'avg_confidence': None,
                'avg_accuracy': None
            }
            continue
        
        confidences = []
        accuracies = []
        
        for item in items:
            confidence = item.get('confidence')
            if confidence is None:
                continue
                
            # 获取真实标签和预测
            truth = item.get('ground_truth', '').lower()
            response = item.get('response', '').lower()
            candidates = item.get('candidates')
            
            # 预测逻辑（与之前保持一致）
            pred = "unknown"
            answer_content = None
            answer_match = re.search(r'<answer>(.*?)</answer>', response)
            if answer_match:
                answer_content = answer_match.group(1).lower()
            
            if answer_content:
                for emotion in candidates:
                    if emotion.lower() in answer_content:
                        pred = emotion.lower()
                        break
            else:            
                for emotion in candidates:
                    if emotion.lower() in response:
                        pred = emotion.lower()
                        break
            
            # 检查预测是否正确
            is_correct = (truth == pred)
            
            confidences.append(confidence)
            accuracies.append(is_correct)
        
        if len(confidences) == 0:
            print(f"警告: {group_name} 组没有有效的置信度数据")
            group_metrics[group_name] = {
                'ece': None,
                'brier_score': None,
                'roc_auc': None,
                'total_samples': len(items),
                'avg_confidence': None,
                'avg_accuracy': None
            }
            continue
        
        # 裁剪置信度到 [0, 1] 范围内
        confidences = np.clip(confidences, 0, 1)
        
        # 计算指标
        ece = calculate_ece(confidences, accuracies)
        brier_score = calculate_brier_score(confidences, accuracies)
        roc_auc = calculate_roc_auc(confidences, accuracies)
        avg_confidence = np.mean(confidences)
        avg_accuracy = np.mean(accuracies)
        
        group_metrics[group_name] = {
            'ece': ece,
            'brier_score': brier_score,
            'roc_auc': roc_auc,
            'total_samples': len(items),
            'valid_samples': len(confidences),
            'avg_confidence': avg_confidence,
            'avg_accuracy': avg_accuracy
        }
        
        print(f"{group_name} - 样本数: {len(items)}, 有效样本: {len(confidences)}")
        print(f"  ECE: {ece:.4f}, Brier Score: {brier_score:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}" if roc_auc is not None else "  ROC AUC: N/A")
        print(f"  平均置信度: {avg_confidence:.4f}, 平均准确率: {avg_accuracy:.4f}")
    
    return group_metrics

def plot_calibration_curves_by_group(data: List[Dict], output_path: str):
    """按三大任务组绘制校准曲线"""
    
    task_groups = {
        'ID VSA': ['FI-2', 'WebEmo-2'],
        'ID VER': ['FI-8', 'EmoSet-8', 'WebEmo-7', 'WebEmo-25'],
        'OOD VER': ['Abstract-8', 'Artphoto-8', 'UnbiasedEmo-6']
    }
    
    plt.figure(figsize=(10, 8))
    
    colors = {'ID VSA': 'blue', 'ID VER': 'green', 'OOD VER': 'red'}
    
    for group_name, tasks in task_groups.items():
        # 收集该组的所有数据
        confidences = []
        accuracies = []
        
        for item in data:
            task = item.get('task', 'unknown')
            if task in tasks:
                confidence = item.get('confidence')
                if confidence is None:
                    continue
                    
                # 获取预测结果
                truth = item.get('ground_truth', '').lower()
                response = item.get('response', '').lower()
                candidates = item.get('candidates')
                
                pred = "unknown"
                answer_content = None
                answer_match = re.search(r'<answer>(.*?)</answer>', response)
                if answer_match:
                    answer_content = answer_match.group(1).lower()
                
                if answer_content:
                    for emotion in candidates:
                        if emotion.lower() in answer_content:
                            pred = emotion.lower()
                            break
                else:            
                    for emotion in candidates:
                        if emotion.lower() in response:
                            pred = emotion.lower()
                            break
                
                is_correct = (truth == pred)
                
                confidences.append(confidence)
                accuracies.append(is_correct)
        
        if len(confidences) > 0:
            confidences = np.array(confidences)
            accuracies = np.array(accuracies).astype(float)
            
            # 裁剪置信度到 [0, 1] 范围内
            confidences = np.clip(confidences, 0, 1)
            
            try:
                # 计算校准曲线
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    accuracies, confidences, n_bins=10, strategy='uniform'
                )
                
                # 绘制校准曲线
                plt.plot(mean_predicted_value, fraction_of_positives, 
                        's-', label=group_name, color=colors[group_name], linewidth=2)
            except ValueError as e:
                print(f"警告: 无法为 {group_name} 绘制校准曲线: {e}")
                print(f"  {group_name} 置信度范围: [{confidences.min():.4f}, {confidences.max():.4f}]")
                continue
    
    # 绘制完美校准线
    plt.plot([0, 1], [0, 1], 'k--', label='Ideal Calibration', alpha=0.5)
    
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Calibration Curve - Task Group Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # 保存图像
    calibration_plot_path = os.path.join(output_path, 'calibration_curves_by_group.png')
    plt.savefig(calibration_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"分组校准曲线图已保存到: {calibration_plot_path}")

def calculate_group_f1(data: List[Dict], group_name: str, task_groups: Dict) -> float:
    """计算指定任务组的F1分数"""
    truths = []
    preds = []
    
    for item in data:
        task = item.get('task', 'unknown')
        if task in task_groups[group_name]:
            truth = item.get('ground_truth', '').lower()
            response = item.get('response', '').lower()
            candidates = item.get('candidates')
            
            # 预测逻辑
            pred = "unknown"
            answer_content = None
            answer_match = re.search(r'<answer>(.*?)</answer>', response)
            if answer_match:
                answer_content = answer_match.group(1).lower()
            
            if answer_content:
                for emotion in candidates:
                    if emotion.lower() in answer_content:
                        pred = emotion.lower()
                        break
            else:            
                for emotion in candidates:
                    if emotion.lower() in response:
                        pred = emotion.lower()
                        break
            
            truths.append(truth)
            preds.append(pred)
    
    if len(truths) > 0:
        return f1_score(truths, preds, average='weighted')
    else:
        return 0.0

def print_extended_metrics_table(task_metrics: Dict, overall_metrics: Dict, group_metrics: Dict, data: List[Dict]) -> Tuple[str, str, str, str, str]:
    """以表格形式打印扩展评估指标（包含Acc, F1, ECE, Brier, ROC AUC），并返回Excel格式行（两行输出）"""
    
    # 定义任务分组
    task_groups = {
        'ID VSA': ['FI-2', 'WebEmo-2'],
        'ID VER': ['FI-8', 'EmoSet-8', 'WebEmo-7', 'WebEmo-25'],
        'OOD VER': ['Abstract-8', 'Artphoto-8', 'UnbiasedEmo-6']
    }
    
    # 定义固定的task顺序
    TASK_ORDER = [
        'Abstract-8', 'Artphoto-8', 'EmoSet-8', 'FI-2', 'FI-8',
        'UnbiasedEmo-6', 'WebEmo-2', 'WebEmo-7', 'WebEmo-25'
    ]
    
    # 准备表头 - 增加ROC AUC列
    header = "| {:<15} | {:<6} | {:<6} | {:<8} | {:<10} | {:<10} | {:<9} |".format(
        "Task/Group", "Acc", "F1", "ECE", "Brier", "ROC AUC", "#Samples"
    )
    separator = "-" * len(header)
    
    # 准备表格内容
    rows = []
    
    # 第一行：细粒度任务指标
    fine_grained_headers = []
    fine_grained_values = []
    
    # 生成细粒度任务指标
    for task in TASK_ORDER:
        if task in task_metrics:
            metrics = task_metrics[task]
            acc_percent = metrics["accuracy"] * 100
            f1_percent = metrics["f1_score"] * 100
            
            # 细粒度任务没有ECE、Brier和ROC AUC
            row = "| {:<15} | {:>6.2f} | {:>6.2f} | {:<8} | {:<10} | {:<10} | {:<9} |".format(
                task,
                acc_percent,
                f1_percent,
                "N/A",
                "N/A",
                "N/A",
                metrics["total_samples"]
            )
            rows.append(row)
            
            # Excel格式（细粒度任务只记录Acc和F1）
            fine_grained_headers.extend([f"{task} Acc", f"{task} F1"])
            fine_grained_values.extend([f"{acc_percent:.2f}", f"{f1_percent:.2f}"])
    
    # 添加分隔行
    rows.append(separator)
    
    # 第二行：分组统计数据
    group_headers = []
    group_values = []
    
    # 计算分组的F1分数
    group_f1_scores = {}
    for group_name in ['ID VSA', 'ID VER', 'OOD VER']:
        group_f1_scores[group_name] = calculate_group_f1(data, group_name, task_groups)
    
    # 生成分组指标
    for group_name in ['ID VSA', 'ID VER', 'OOD VER']:
        metrics = group_metrics.get(group_name, {})
        
        if metrics['total_samples'] == 0:
            row = "| {:<15} | {:<6} | {:<6} | {:<8} | {:<10} | {:<10} | {:<9} |".format(
                group_name, "N/A", "N/A", "N/A", "N/A", "N/A", "0"
            )
            rows.append(row)
            # 添加Acc, F1, Coverage到Excel
            group_headers.extend([f"{group_name} Acc", f"{group_name} F1", f"{group_name} Coverage"])
            group_values.extend(["N/A", "N/A", "N/A"])
        else:
            # 计算分组的Acc
            acc_percent = metrics.get('avg_accuracy', 0) * 100
            # 使用计算得到的F1分数
            f1_percent = group_f1_scores[group_name] * 100
            ece = metrics.get('ece', None)
            brier = metrics.get('brier_score', None)
            roc_auc = metrics.get('roc_auc', None)
            samples = metrics.get('total_samples', 0)
            
            # 计算Coverage (有效样本比例)
            valid_samples = metrics.get('valid_samples', 0)
            coverage = (valid_samples / samples * 100) if samples > 0 else 0
            
            # 将ECE、Brier和ROC AUC转换为百分比
            ece_percent = ece * 100 if ece is not None else None
            brier_percent = brier * 100 if brier is not None else None
            roc_auc_percent = roc_auc * 100 if roc_auc is not None else None
            
            if ece is not None and brier is not None and roc_auc is not None:
                row = "| {:<15} | {:>6.2f} | {:>6.2f} | {:>8.2f} | {:>10.2f} | {:>10.2f} | {:<9} |".format(
                    group_name, acc_percent, f1_percent, ece_percent, brier_percent, roc_auc_percent, samples
                )
                rows.append(row)
                # 添加Acc, F1, Coverage
                group_headers.extend([f"{group_name} Acc", f"{group_name} F1", f"{group_name} Coverage"])
                group_values.extend([f"{acc_percent:.2f}", f"{f1_percent:.2f}", f"{coverage:.2f}"])
            else:
                row = "| {:<15} | {:>6.2f} | {:>6.2f} | {:<8} | {:<10} | {:<10} | {:<9} |".format(
                    group_name, acc_percent, f1_percent, "N/A", "N/A", "N/A", str(samples)
                )
                rows.append(row)
                # 添加Acc, F1, Coverage
                group_headers.extend([f"{group_name} Acc", f"{group_name} F1", f"{group_name} Coverage"])
                group_values.extend([f"{acc_percent:.2f}", f"{f1_percent:.2f}", f"{coverage:.2f}"])
    
    # 添加ECE、Brier和ROC AUC指标到分组统计数据（转换为百分比）
    for group_name in ['ID VSA', 'ID VER', 'OOD VER']:
        metrics = group_metrics.get(group_name, {})
        
        if metrics['total_samples'] > 0:
            ece = metrics.get('ece', None)
            brier = metrics.get('brier_score', None)
            roc_auc = metrics.get('roc_auc', None)
            
            if ece is not None and brier is not None and roc_auc is not None:
                # 转换为百分比
                ece_percent = ece * 100
                brier_percent = brier * 100
                roc_auc_percent = roc_auc * 100
                
                group_headers.extend([f"{group_name} ECE", f"{group_name} Brier", f"{group_name} ROC AUC"])
                group_values.extend([f"{ece_percent:.2f}", f"{brier_percent:.2f}", f"{roc_auc_percent:.2f}"])
            else:
                group_headers.extend([f"{group_name} ECE", f"{group_name} Brier", f"{group_name} ROC AUC"])
                group_values.extend(["N/A", "N/A", "N/A"])
        else:
            group_headers.extend([f"{group_name} ECE", f"{group_name} Brier", f"{group_name} ROC AUC"])
            group_values.extend(["N/A", "N/A", "N/A"])
    
    # 添加总体统计行到表格
    overall_acc_percent = overall_metrics["accuracy"] * 100
    overall_f1_percent = overall_metrics["f1_score"] * 100
    
    total_row = "| {:<15} | {:>6.2f} | {:>6.2f} | {:<8} | {:<10} | {:<10} | {:<9} |".format(
        "Overall",
        overall_acc_percent,
        overall_f1_percent,
        "N/A",
        "N/A",
        "N/A",
        overall_metrics["total_samples"]
    )
    
    # 添加总体指标到两行Excel格式
    fine_grained_headers.extend(["Overall Acc", "Overall F1"])
    fine_grained_values.extend([f"{overall_acc_percent:.2f}", f"{overall_f1_percent:.2f}"])
    
    group_headers.extend(["Overall Acc", "Overall F1"])
    group_values.extend([f"{overall_acc_percent:.2f}", f"{overall_f1_percent:.2f}"])
    
    # 组合表格所有部分
    table = "\n".join([separator, header, separator] + rows + [separator, total_row, separator])
    
    # 生成两行Excel格式
    excel_header_line1 = "\t".join(fine_grained_headers)
    excel_value_line1 = "\t".join(fine_grained_values)
    
    excel_header_line2 = "\t".join(group_headers)
    excel_value_line2 = "\t".join(group_values)
    
    return table, excel_header_line1, excel_value_line1, excel_header_line2, excel_value_line2

def log_to_wandb(task_metrics: Dict, overall_metrics: Dict, group_metrics: Dict, config: Dict, 
                 project_name: str, experiment_name: str):
    """记录所有指标到Weights & Biases"""
    # 初始化wandb
    wandb.init(project=project_name, name=experiment_name, config=config)
    
    # 准备要记录的所有指标
    metrics_to_log = {}
    
    # 1. 添加总体指标
    metrics_to_log.update({
        f"overall/{k}": v for k, v in overall_metrics.items()
    })
    
    # 2. 添加每个task的指标
    for task, metrics in task_metrics.items():
        # 使用task名称作为前缀 (替换空格为下划线)
        task_prefix = f"task/{task.replace(' ', '_')}"
        metrics_to_log.update({
            f"{task_prefix}_{k}": v for k, v in metrics.items()
        })
    
    # 3. 添加分组校准指标
    for group_name, metrics in group_metrics.items():
        group_prefix = f"group/{group_name.replace(' ', '_')}"
        if metrics['ece'] is not None:
            metrics_to_log.update({
                f"{group_prefix}_ece": metrics['ece'],
                f"{group_prefix}_brier_score": metrics['brier_score'],
                f"{group_prefix}_roc_auc": metrics['roc_auc'],
                f"{group_prefix}_avg_confidence": metrics['avg_confidence'],
                f"{group_prefix}_avg_accuracy": metrics['avg_accuracy']
            })
    
    # 4. 记录所有指标
    wandb.log(metrics_to_log)
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='多模态模型评估脚本 - 简化版')
    parser.add_argument('--input_file', type=str, required=True,
                       help='输入的合并JSONL文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录路径 (默认为输入文件所在目录)')
    parser.add_argument('--project_name', type=str, default='vsa', 
                       help='W&B项目名称')
    parser.add_argument('--experiment_name', type=str, default=None, 
                       help='实验名称 (默认为输入文件名)')
    parser.add_argument('--model_name', type=str, default='qwen25_vl', 
                       help='模型名称')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input_file)
    
    # 设置实验名称
    if args.experiment_name is None:
        args.experiment_name = os.path.splitext(os.path.basename(args.input_file))[0]
    
    # 1. 加载合并后的JSONL文件
    print(f"正在加载文件: {args.input_file}")
    merged_data = load_jsonl_file(args.input_file)
    
    # 2. 计算基础评估指标
    task_metrics, overall_metrics = calculate_metrics_by_task(merged_data)
    
    # 3. 计算分组校准指标（包含ROC AUC）
    print("\n" + "="*80)
    print("分组校准指标 (ECE, Brier Score, ROC AUC)")
    print("="*80)
    
    group_metrics = calculate_group_calibration_metrics(merged_data)
    
    # 4. 打印扩展评估结果表格（包含Acc, F1, ECE, Brier, ROC AUC）
    extended_table, excel_header_line1, excel_value_line1, excel_header_line2, excel_value_line2 = print_extended_metrics_table(
        task_metrics, overall_metrics, group_metrics, merged_data
    )
    print("\n扩展评估结果 (包含Acc, F1, ECE, Brier, ROC AUC):")
    print(extended_table)
    
    # 5. 打印两行Excel格式
    print("\nExcel格式 - 第一行（细粒度任务）:")
    print("表头:")
    print(excel_header_line1)
    print("数据:")
    print(excel_value_line1)
    
    print("\nExcel格式 - 第二行（分组统计）:")
    print("表头:")
    print(excel_header_line2)
    print("数据:")
    print(excel_value_line2)
    
    # 6. 按组计算置信度阈值分析
    print("\n" + "="*80)
    print("分组置信度阈值分析 (ID VSA, ID VER, OOD VER)")
    print("="*80)
    
    conf_headers, conf_data = calculate_confidence_metrics_by_group(merged_data)
    
    print("\n分组置信度阈值Excel格式:")
    print("表头:")
    print("\t".join(conf_headers))
    print("数据 (每行对应一个置信度阈值从0.1到0.9):")
    for i, row in enumerate(conf_data):
        threshold = 0.1 * (i + 1)
        print(f"阈值 {threshold:.1f}:\t" + "\t".join(row))
    
    # 7. 绘制分组离散曲线图
    plot_discrete_curves_by_group(merged_data, args.output_dir)
    
    # 8. 绘制分组校准曲线图
    plot_calibration_curves_by_group(merged_data, args.output_dir)
    
    # 9. 记录到wandb
    wandb_config = {
        "model_name": args.model_name,
        "input_file": args.input_file
    }
    
    # 添加校准指标到wandb配置
    for group_name, metrics in group_metrics.items():
        if metrics['ece'] is not None:
            wandb_config[f"{group_name.replace(' ', '_')}_ECE"] = metrics['ece']
            wandb_config[f"{group_name.replace(' ', '_')}_Brier_Score"] = metrics['brier_score']
            if metrics['roc_auc'] is not None:
                wandb_config[f"{group_name.replace(' ', '_')}_ROC_AUC"] = metrics['roc_auc']
    
    log_to_wandb(task_metrics, overall_metrics, group_metrics, wandb_config, 
                project_name=args.project_name, 
                experiment_name=args.experiment_name)

if __name__ == "__main__":
    main()
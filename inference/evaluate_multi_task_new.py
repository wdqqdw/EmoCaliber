import os
import re
import json
import wandb
import argparse
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from typing import List, Dict, Tuple

def merge_results(output_path: str, n_gpu: int) -> List[Dict]:
    """合并多个GPU生成的推理结果文件"""
    merged_data = []
    error_data = []
    for i in range(n_gpu):
        file_path = f"{output_path}/{i}.jsonl"
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if item.get('error') is not None:
                    error_data.append(item)
                else:
                    merged_data.append(item)
    
    # 保存合并后的文件
    merged_file = f"{output_path}/merged.jsonl"
    with open(merged_file, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    error_file = f"{output_path}/error.jsonl"
    with open(error_file, 'w', encoding='utf-8') as f:
        for item in error_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"合并完成! 结果已保存到: {merged_file}, 推理异常样本已保存到: {error_file}")
    return merged_data

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

def log_to_wandb(task_metrics: Dict, overall_metrics: Dict, config: Dict, 
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
    
    # 3. 记录所有指标
    wandb.log(metrics_to_log)
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='多模态模型评估脚本')
    parser.add_argument('--n_gpu', type=int, default=8, 
                       help='gpu数量')
    parser.add_argument('--project_name', type=str, default='vsa', 
                       help='项目名称')
    parser.add_argument('--experiment_name', type=str, default='test-0626-1', 
                       help='实验名称')
    parser.add_argument('--output_path', type=str, required=True,
                       help='输出结果路径')
    parser.add_argument('--model_name', type=str, default='qwen25_vl', 
                       help='模型名称 (qwen25_vl)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='生成温度')
    
    args = parser.parse_args()
    
    # 1. 合并结果文件
    merged_data = merge_results(args.output_path, args.n_gpu)
    
    # 2. 计算评估指标
    task_metrics, overall_metrics = calculate_metrics_by_task(merged_data)
    
    # 3. 打印评估结果表格和Excel格式
    table, excel_header, excel_value = print_metrics_table(task_metrics, overall_metrics)
    print("\n评估结果:")
    print(table)
    
    # 4. 打印Excel格式（可直接复制到Excel）
    print("\nExcel格式（可直接复制到Excel中）:")
    print("表头:")
    print(excel_header)
    print("数据:")
    print(excel_value)
    
    # 5. 记录到wandb
    wandb_config = {
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "n_gpu": args.n_gpu,
        "temperature": args.temperature
    }
    
    log_to_wandb(task_metrics, overall_metrics, wandb_config, 
                project_name=args.project_name, 
                experiment_name=args.experiment_name)

if __name__ == "__main__":
    main()
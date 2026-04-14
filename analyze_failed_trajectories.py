#!/usr/bin/env python3
"""
分析 Harbor Dataset 训练中的失败样本
从训练日志和 checkpoint 文件中提取失败的 trajectory 并分类
"""

import json
import re
import os
from collections import defaultdict
from pathlib import Path

# 配置
LOG_FILE = "/root/liyu/rllm-private/harbor_train.log"
CHECKPOINT_DIR = "/root/liyu/rllm-private/checkpoints/rllm-harbor-dataset/harbor-9b-k8s-b32-n8-resp24k-total32k-sp2-noeval-tb-t10/chat_completions"
OUTPUT_DIR = "/root/liyu/rllm-private/failed_samples_analysis"

def parse_log_file():
    """从日志文件中提取所有 trajectory 的完成信息"""
    print(f"正在解析日志文件: {LOG_FILE}")

    with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 提取 trajectory 完成记录
    pattern = r'Trajectory (\d+) completed due to: (\w+)\. Reward is ([\d.]+)\.'
    matches = re.findall(pattern, content)

    trajectories = []
    for traj_id, reason, reward in matches:
        trajectories.append({
            'trajectory_id': int(traj_id),
            'completion_reason': reason,
            'reward': float(reward),
            'success': float(reward) > 0.0
        })

    print(f"找到 {len(trajectories)} 条 trajectory 记录")
    return trajectories

def categorize_failures(trajectories):
    """分类失败的 trajectory"""
    failed = [t for t in trajectories if not t['success']]

    categories = {
        'TRUNCATION': [],  # 超出最大步数限制
        'ENV_DONE_FAILED': [],  # 正常完成但答案错误
    }

    for traj in failed:
        if traj['completion_reason'] == 'TRUNCATION':
            categories['TRUNCATION'].append(traj)
        elif traj['completion_reason'] == 'ENV_DONE':
            categories['ENV_DONE_FAILED'].append(traj)

    return categories, failed

def load_chat_completions():
    """加载所有 chat completion 文件"""
    print(f"\n正在加载 chat completions 从: {CHECKPOINT_DIR}")

    all_completions = []
    jsonl_files = sorted(Path(CHECKPOINT_DIR).glob("*.jsonl"))

    for jsonl_file in jsonl_files:
        step_num = jsonl_file.stem
        print(f"  加载 step {step_num}...")

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    all_completions.append({
                        'step': int(step_num),
                        'line': line_num,
                        'data': data
                    })
                except json.JSONDecodeError as e:
                    print(f"    警告: step {step_num} line {line_num} 解析失败: {e}")

    print(f"总共加载 {len(all_completions)} 条 chat completion 记录")
    return all_completions

def extract_task_info(messages):
    """从消息中提取任务信息"""
    task_description = ""
    model_response = ""

    for msg in messages:
        if msg['role'] == 'user':
            # 提取用户任务描述
            content = msg.get('content', '')
            if 'Task:' in content or 'task:' in content:
                task_description = content[:500]  # 截取前500字符
        elif msg['role'] == 'assistant':
            # 提取模型响应
            content = msg.get('content', '')
            model_response = content[:1000]  # 截取前1000字符

    return task_description, model_response

def analyze_and_save(trajectories, completions):
    """分析并保存失败样本"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 分类失败样本
    categories, all_failed = categorize_failures(trajectories)

    print(f"\n=== 失败样本统计 ===")
    print(f"总 trajectory 数: {len(trajectories)}")
    print(f"失败数: {len(all_failed)} ({len(all_failed)/len(trajectories)*100:.1f}%)")
    print(f"  - TRUNCATION (超出步数限制): {len(categories['TRUNCATION'])}")
    print(f"  - ENV_DONE_FAILED (答案错误): {len(categories['ENV_DONE_FAILED'])}")

    # 保存分类结果
    summary = {
        'total_trajectories': len(trajectories),
        'failed_count': len(all_failed),
        'success_count': len(trajectories) - len(all_failed),
        'failure_rate': len(all_failed) / len(trajectories) if trajectories else 0,
        'categories': {
            'TRUNCATION': {
                'count': len(categories['TRUNCATION']),
                'percentage': len(categories['TRUNCATION']) / len(all_failed) * 100 if all_failed else 0,
                'description': '模型无法在规定步数内完成任务，可能陷入循环或策略低效'
            },
            'ENV_DONE_FAILED': {
                'count': len(categories['ENV_DONE_FAILED']),
                'percentage': len(categories['ENV_DONE_FAILED']) / len(all_failed) * 100 if all_failed else 0,
                'description': '模型完成了执行但给出了错误答案'
            }
        }
    }

    summary_file = os.path.join(OUTPUT_DIR, 'failure_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n已保存统计摘要到: {summary_file}")

    # 保存每个类别的详细失败列表
    for category_name, failed_list in categories.items():
        if not failed_list:
            continue

        output_file = os.path.join(OUTPUT_DIR, f'{category_name.lower()}_failures.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(failed_list, f, indent=2, ensure_ascii=False)
        print(f"已保存 {category_name} 失败列表到: {output_file}")

    # 保存所有失败样本的详细信息
    all_failed_file = os.path.join(OUTPUT_DIR, 'all_failed_trajectories.json')
    with open(all_failed_file, 'w', encoding='utf-8') as f:
        json.dump(all_failed, f, indent=2, ensure_ascii=False)
    print(f"已保存所有失败样本到: {all_failed_file}")

    # 生成可读的报告
    report_file = os.path.join(OUTPUT_DIR, 'failure_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Harbor Dataset 训练失败样本分析报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"总 trajectory 数: {len(trajectories)}\n")
        f.write(f"成功数: {len(trajectories) - len(all_failed)} ({(len(trajectories) - len(all_failed))/len(trajectories)*100:.1f}%)\n")
        f.write(f"失败数: {len(all_failed)} ({len(all_failed)/len(trajectories)*100:.1f}%)\n\n")

        f.write("-" * 80 + "\n")
        f.write("失败类型分布:\n")
        f.write("-" * 80 + "\n\n")

        for category_name, failed_list in categories.items():
            if not failed_list:
                continue

            percentage = len(failed_list) / len(all_failed) * 100 if all_failed else 0
            f.write(f"{category_name}:\n")
            f.write(f"  数量: {len(failed_list)} ({percentage:.1f}% of failures)\n")
            f.write(f"  描述: {summary['categories'][category_name]['description']}\n")
            f.write(f"  示例 trajectory IDs: {[t['trajectory_id'] for t in failed_list[:10]]}\n\n")

        f.write("-" * 80 + "\n")
        f.write("建议:\n")
        f.write("-" * 80 + "\n\n")

        if len(categories['TRUNCATION']) > len(categories['ENV_DONE_FAILED']):
            f.write("1. TRUNCATION 失败占主导，建议:\n")
            f.write("   - 检查模型是否陷入重复执行相同命令的循环\n")
            f.write("   - 考虑增加最大步数限制或优化策略\n")
            f.write("   - 分析模型的推理链，看是否缺乏目标导向\n\n")
        else:
            f.write("1. ENV_DONE_FAILED 失败占主导，建议:\n")
            f.write("   - 模型能完成任务但答案错误，需要提升理解能力\n")
            f.write("   - 检查是否是测试用例理解错误\n")
            f.write("   - 考虑增强 prompt 或提供更多示例\n\n")

        f.write("2. 查看具体失败样本:\n")
        f.write(f"   - 所有失败样本: {all_failed_file}\n")
        f.write(f"   - TRUNCATION 失败: {os.path.join(OUTPUT_DIR, 'truncation_failures.json')}\n")
        f.write(f"   - ENV_DONE 失败: {os.path.join(OUTPUT_DIR, 'env_done_failed_failures.json')}\n")

    print(f"已生成可读报告: {report_file}")

    return summary

def main():
    print("开始分析 Harbor Dataset 训练失败样本...\n")

    # 解析日志
    trajectories = parse_log_file()

    # 加载 chat completions (可选，用于更详细的分析)
    # completions = load_chat_completions()

    # 分析并保存
    summary = analyze_and_save(trajectories, [])

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"\n结果保存在: {OUTPUT_DIR}/")
    print(f"  - failure_summary.json: 统计摘要")
    print(f"  - failure_report.txt: 可读报告")
    print(f"  - all_failed_trajectories.json: 所有失败样本")
    print(f"  - truncation_failures.json: TRUNCATION 类型失败")
    print(f"  - env_done_failed_failures.json: ENV_DONE 类型失败")

if __name__ == '__main__':
    main()

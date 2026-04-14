#!/usr/bin/env python3
"""
根据 trajectory_id 提取原始对话记录
从 chat_completions 文件中查找并导出完整的对话历史
"""

import json
import sys
import os
from pathlib import Path

# 配置
CHECKPOINT_DIR = "/root/liyu/rllm-private/checkpoints/rllm-harbor-dataset/harbor-9b-k8s-b32-n8-resp24k-total32k-sp2-noeval-tb-t10/chat_completions"
OUTPUT_DIR = "/root/liyu/rllm-private/failed_samples_analysis/trajectories"

def calculate_trajectory_location(trajectory_id, trajectories_per_step=128):
    """
    计算 trajectory_id 对应的文件位置

    训练配置:
    - 每个 step 处理 128 个 trajectory
    - 每个 trajectory 是一个完整的对话历史

    trajectory_id 的编号规则:
    - step 1: trajectory 0-127
    - step 2: trajectory 128-255
    - ...
    """
    # 计算在哪个 step
    step = (trajectory_id // trajectories_per_step) + 1

    # 计算在该 step 中的哪一行
    line_num = trajectory_id % trajectories_per_step

    return step, line_num

def extract_task_info(messages):
    """从消息中提取任务信息"""
    task_id = None
    task_description = None

    for msg in messages:
        if isinstance(msg, dict) and msg.get('role') == 'user':
            content = msg.get('content', '')
            # 提取 Task ID
            if 'Task:' in content:
                lines = content.split('\n')
                for line in lines:
                    if line.strip().startswith('Task:'):
                        task_id = line.strip().replace('Task:', '').strip()
                        break

            # 提取任务描述（前500字符）
            if task_id:
                task_description = content[:500]
                break

    return {
        'task_id': task_id,
        'task_description': task_description
    }

def extract_trajectory(trajectory_id):
    """提取指定 trajectory_id 的完整对话"""
    print(f"正在提取 trajectory {trajectory_id}...")

    # 计算位置
    step, line_num = calculate_trajectory_location(trajectory_id)

    print(f"  位置: step={step}, line={line_num}")

    # 读取对应的文件
    jsonl_file = Path(CHECKPOINT_DIR) / f"{step}.jsonl"

    if not jsonl_file.exists():
        print(f"错误: 文件不存在 {jsonl_file}")
        return None

    print(f"  读取文件: {jsonl_file}")

    # 读取指定行
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == line_num:
                try:
                    # 每行是一个完整的对话历史 (JSON 数组)
                    messages = json.loads(line.strip())

                    print(f"  成功提取! 对话包含 {len(messages)} 条消息")

                    # 提取任务信息
                    task_info = extract_task_info(messages)

                    return {
                        'trajectory_id': trajectory_id,
                        'step': step,
                        'line': line_num,
                        'task_info': task_info,
                        'messages': messages
                    }

                except json.JSONDecodeError as e:
                    print(f"错误: JSON 解析失败: {e}")
                    return None

    print(f"错误: 未找到第 {line_num} 行")
    return None

def save_trajectory(trajectory_data, output_format='json'):
    """保存 trajectory 到文件"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trajectory_id = trajectory_data['trajectory_id']

    if output_format == 'json':
        # 保存为 JSON 格式
        output_file = Path(OUTPUT_DIR) / f"trajectory_{trajectory_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
        print(f"\n已保存到: {output_file}")

    elif output_format == 'txt':
        # 保存为可读文本格式
        output_file = Path(OUTPUT_DIR) / f"trajectory_{trajectory_id}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Trajectory ID: {trajectory_id}\n")
            f.write(f"Step: {trajectory_data['step']}\n")
            f.write(f"Line: {trajectory_data['line']}\n")

            task_info = trajectory_data.get('task_info', {})
            if task_info.get('task_id'):
                f.write(f"Task ID: {task_info['task_id']}\n")

            f.write("=" * 80 + "\n\n")

            for i, msg in enumerate(trajectory_data['messages']):
                if isinstance(msg, dict):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                else:
                    role = 'unknown'
                    content = str(msg)

                f.write(f"[{i+1}] {role.upper()}\n")
                f.write("-" * 80 + "\n")

                # 限制内容长度以避免文件过大
                if len(content) > 10000:
                    f.write(content[:10000] + "\n\n... (内容过长，已截断) ...\n")
                else:
                    f.write(content + "\n")

                f.write("\n" + "=" * 80 + "\n\n")

        print(f"\n已保存到: {output_file}")

    return output_file

def main():
    if len(sys.argv) < 2:
        print("用法: python3 extract_trajectory.py <trajectory_id> [output_format]")
        print("  trajectory_id: 要提取的 trajectory ID")
        print("  output_format: 输出格式 (json 或 txt，默认 txt)")
        print("\n示例:")
        print("  python3 extract_trajectory.py 3")
        print("  python3 extract_trajectory.py 118 json")
        sys.exit(1)

    try:
        trajectory_id = int(sys.argv[1])
    except ValueError:
        print(f"错误: trajectory_id 必须是整数，得到: {sys.argv[1]}")
        sys.exit(1)

    output_format = sys.argv[2] if len(sys.argv) > 2 else 'txt'
    if output_format not in ['json', 'txt']:
        print(f"错误: output_format 必须是 'json' 或 'txt'，得到: {output_format}")
        sys.exit(1)

    # 提取 trajectory
    trajectory_data = extract_trajectory(trajectory_id)

    if trajectory_data is None:
        print("\n提取失败!")
        sys.exit(1)

    # 显示任务信息
    task_info = trajectory_data.get('task_info', {})
    if task_info.get('task_id'):
        print(f"  任务 ID: {task_info['task_id']}")

    # 保存
    output_file = save_trajectory(trajectory_data, output_format)

    print("\n提取完成!")
    print(f"Trajectory {trajectory_id} 已保存到: {output_file}")

if __name__ == '__main__':
    main()

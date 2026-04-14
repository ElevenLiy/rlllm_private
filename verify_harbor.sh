#!/bin/bash
# Harbor Dataset 链路验证脚本

echo "=========================================="
echo "Harbor Dataset 链路验证"
echo "=========================================="
echo ""

echo "✅ 1. 数据集目录检查"
if [ -d /data/harbor-dataset ]; then
    TASK_COUNT=$(ls /data/harbor-dataset | wc -l)
    echo "  - 数据目录: 存在"
    echo "  - 任务数量: $TASK_COUNT"
else
    echo "  - 数据目录: 缺失 ❌"
    exit 1
fi
echo ""

echo "✅ 2. 示例任务结构检查"
if [ -f /data/harbor-dataset/task_0/instruction.md ] && [ -f /data/harbor-dataset/task_0/task.toml ]; then
    echo "  - instruction.md: 存在"
    echo "  - task.toml: 存在"
    echo "  - tests/: $([ -d /data/harbor-dataset/task_0/tests ] && echo '存在' || echo '缺失')"
else
    echo "  - 任务结构: 不完整 ❌"
    exit 1
fi
echo ""

echo "✅ 3. 数据集注册测试"
export TB_TASKS_ROOT=/data/harbor-dataset
export TB_RLLM_DATASET_NAME=harbor_dataset
python /root/liyu/rllm-private/scripts/openthoughts_terminal_bench/register_harbor_dataset.py 2>&1 | grep "Registered" && echo "  状态: 成功" || echo "  状态: 失败 ❌"
echo ""

echo "✅ 4. K8s 集群连通性"
export KUBECONFIG=/data/k8s_access/kubeconfig
/data/k8s_access/kubectl.real cluster-info 2>&1 | grep "Kubernetes control plane" && echo "  状态: 连通" || echo "  状态: 失败 ❌"
echo ""

echo "✅ 5. 训练脚本可执行性"
export TB_EXECUTION_BACKEND=k8s
export MODEL_PATH=/data/models/Qwen3___5-9B
python /root/liyu/rllm-private/scripts/openthoughts_terminal_bench/train_terminal_bench_direct_rllm.py --help 2>&1 | grep -q "Hydra" && echo "  状态: 可执行" || echo "  状态: 失败 ❌"
echo ""

echo "=========================================="
echo "链路验证完成 ✅"
echo "=========================================="
echo ""
echo "启动训练："
echo "  cd /root/liyu/rllm-private"
echo "  source setup_harbor_env.sh"
echo "  bash scripts/openthoughts_terminal_bench/run_harbor_9b_noeval_resp24k_total32k_sp2.sh"
echo ""

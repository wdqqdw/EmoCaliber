export VLLM_WORKER_MULTIPROC_METHOD=spawn
cur_path=/path/to/emocaliber/project

cd $cur_path
experiment_name=infer_and_eval_example
model_name=qwen25_vl
data_path=$cur_path/input_data/test_8k.jsonl
output_path=$cur_path/infer_results/$experiment_name
checkpoint=/path/to/model/checkpoint
project_name=your_project_name #for wandb
temperature=0.7

n_gpu=8 #set to your gpu number
if [ ! -d "$output_path" ]; then
    echo "输出目录不存在，正在创建: $output_path"
    mkdir -p "$output_path"
else
    echo "输出目录已存在: $output_path"
fi

#split test data to $n_gpu parts and inference them separately
n_gpu_1=`expr $n_gpu - 1`
data_count=$(wc -l < "$data_path")
batch_size=$(((data_count / $n_gpu)))

# 存储所有后台进程的 PID
pids=()
for i in $(seq 0 $n_gpu_1); do
    if test -f "$output_path/$i.jsonl"; then
        echo "$output_path/$i.jsonl exists, skip."
        continue
    fi
    nohup bash inference/infer.sh \
        $i $data_path $batch_size $checkpoint $output_path $model_name $temperature > \
        "$output_path/${i}.log" 2>&1 &
    # 记录当前后台进程的 PID
    pids+=($!)
done

# 等待所有后台进程完成
wait "${pids[@]}"

# 存储所有后台进程的 PID
pids=()
for i in $(seq 0 $n_gpu_1); do
    if test -f "$output_path/$i.jsonl"; then
        echo "$output_path/$i.jsonl exists, skip."
        continue
    fi
    nohup bash inference/infer.sh \
        $i $data_path $batch_size $checkpoint $output_path $model_name $temperature > \
        "$output_path/${i}.log" 2>&1 &
    # 记录当前后台进程的 PID
    pids+=($!)
done


# 等待所有后台进程完成
wait "${pids[@]}"

echo "模型权重: $checkpoint 预测完毕, 预测结果已写入: $output_path"

# 执行评估并记录日志
python inference/evaluate_multi_task_new.py \
    --output_path $output_path \
    --n_gpu $n_gpu \
    --project_name $project_name \
    --experiment_name $experiment_name \
    --model_name $model_name \
    --checkpoint $checkpoint \
    --temperature $temperature > \
    ${output_path}/metric.log 2>&1
    
python inference/extract_verb_confidence.py \
    -input_file ${output_path}/merged.jsonl \
    -o ${output_path}/merged_conf.jsonl

python inference/evaluate_multi_task_conf_from_merged.py \
    --input_file ${output_path}/merged_conf.jsonl \
    --output_dir ${output_path} \
    --project_name $project_name \
    --experiment_name $experiment_name \
    --model_name $model_name > \
    ${output_path}/metric_conf.log 2>&1
    
echo "评估完成! 结果已保存到: ${output_path}/metric.log"
echo "合并后的文件: ${output_path}/merged.jsonl"

#pkill -f "inference"
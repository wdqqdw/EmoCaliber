sudo pip install ms-swift -U
sudo pip install deepspeed==0.16.0 -U
sudo pip install "decord" -U

#We adopt ms-swift for SFT
swift_path=/path/to/ms-swift
cd swift_path

export GPUS_PER_NODE=8
export NCCL_IB_QPS_PER_CONNECTION=8
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=`echo $METIS_WORKER_0_PORT | cut -d ',' -f 1`
export RANK=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=$swift_path/nccl_debug.log #can be other place
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_CHECKPOINT_SERIALIZATION=1

MIN_PIXELS=200704 \
MAX_PIXELS=451584 \
accelerate launch \
    --config_file "$swift_path/config/fsdp_offload.json" \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $RANK \
    --num_machines $WORLD_SIZE \
    --num_processes $(($WORLD_SIZE * $GPUS_PER_NODE)) \
    swift/cli/sft.py \
    --model /path/to/scaffold/model \
    --train_type full \
    --dataset '/path/to/sft_data(VEC-CoT/stage2_43k_w_conf.jsonl should work fine)' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --learning_rate 4e-6 \
    --gradient_checkpointing true \
    --gradient_accumulation_steps $(expr 16 / $GPUS_PER_NODE) \
    --eval_steps 100 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir /path/to/save/your/checkpoint \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
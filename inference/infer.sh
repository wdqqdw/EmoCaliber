INDEDX=$1
data_path=$2
batch_size=$3
checkpoint=$4
output_path=$5
model_name=$6
temperature=$7

CUDA_VISIBLE_DEVICES=$INDEDX python3 inference/inference_vllm.py --index $INDEDX --data_path $data_path --data_num $batch_size --checkpoint $checkpoint --output_path $output_path --model_name $model_name --temperature $temperature

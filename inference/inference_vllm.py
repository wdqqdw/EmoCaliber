import argparse
import json
from typing import List, Dict
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='多模态模型推理脚本')
    parser.add_argument('--model_name', type=str, default='qwen25_vl', 
                       help='模型名称 (qwen25_vl)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--data_path', type=str, required=True,
                       help='输入数据路径 (JSONL格式)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='输出结果路径')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                       help='生成的最大token数')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='生成温度')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--data_num', type=int, default=326)
    
    args = parser.parse_args()

    # 准备输出目录
    #output_dir = os.path.dirname(args.output_path)
    #os.makedirs(args.output_path, exist_ok=True)

    # 初始化模型
    model = LLM(
        model=args.checkpoint,
        trust_remote_code=True)
    min_pixels = 256 * 28 * 28
    max_pixels = 576 * 28 * 28
    # 初始化processor
    processor = AutoProcessor.from_pretrained(
        args.checkpoint,
        min_pixels=min_pixels,
        max_pixels=max_pixels   
    )

    start_idx = args.index * args.data_num
    end_idx = (args.index+1) * args.data_num
    
    # 确定处理范围
    datas = open(args.data_path).readlines()[start_idx:end_idx]    
    print(f"处理数据范围: {start_idx} - {end_idx-1} (共{len(datas)}条)")

    # 处理数据
    output_file = f'{args.output_path}/{args.index}.jsonl'
    with open(output_file, 'w') as out_f:
        for i, line in enumerate(datas):
            try:
                data = json.loads(line)
                conversations = data.get('conversations', [])

                text = processor.apply_chat_template(
                    conversations, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(conversations)

                sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=args.max_new_tokens,
                )
                inputs = {
                    "prompt": text,
                    "multi_modal_data": {"image": image_inputs},
                }
                outputs = model.generate(
                    prompts = inputs,
                    sampling_params = sampling_params
                )
                
                # 保存结果
                data['response'] = outputs[0].outputs[0].text
                out_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                # 打印进度
                if (i+1) % 100 == 0:
                    print(f"已推理 {start_idx + i + 1}/{end_idx} 条数据")
                    
            except Exception as e:
                print(f"处理第 {start_idx + i} 条数据时出错: {str(e)}")
                # 保存错误信息
                error_data = {
                    'original_data': data,
                    'error': str(e)
                }
                out_f.write(json.dumps(error_data, ensure_ascii=False) + '\n')

    print(f'处理完成! 结果已保存到: {output_file}')

if __name__ == '__main__':
    main()
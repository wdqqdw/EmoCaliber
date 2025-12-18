import json
import re
import sys
import argparse

def extract_confidence(response):
    """从response中提取confidence值"""
    match = re.search(r'<confidence>(.*?)</confidence>', response)
    if match:
        return float(match.group(1).replace('{','').replace('}',''))
    return None

def process_jsonl_file(input_file, output_file):
    """处理JSONL文件，添加confidence字段"""
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # 提取confidence值
                response = data.get('response', '')
                confidence = extract_confidence(response)
                
                # 添加confidence字段到数据中
                data['confidence'] = confidence
                
                # 写入新的JSONL文件
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行JSON解析错误: {e}")
                continue
            except Exception as e:
                print(f"警告: 第 {line_num} 行处理错误: {e}")
                continue

def main():
    parser = argparse.ArgumentParser(description='从JSONL文件中提取confidence值')
    parser.add_argument('-input_file', help='输入的JSONL文件路径')
    parser.add_argument('-o', '--output', help='输出的JSONL文件路径', default=None)
    
    args = parser.parse_args()
    
    # 设置输出文件路径
    if args.output is None:
        # 如果未指定输出文件，在输入文件名后添加_conf
        input_path = args.input_file
        if input_path.endswith('.jsonl'):
            output_path = input_path.replace('.jsonl', '_with_conf.jsonl')
        else:
            output_path = input_path + '_with_conf.jsonl'
    else:
        output_path = args.output
    
    print(f"输入文件: {args.input_file}")
    print(f"输出文件: {output_path}")
    
    try:
        process_jsonl_file(args.input_file, output_path)
        print(f"处理完成! 共处理文件: {args.input_file} -> {output_path}")
    except Exception as e:
        print(f"处理失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
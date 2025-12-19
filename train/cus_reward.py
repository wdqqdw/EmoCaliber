# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


import math
import string

def compute_score(solution_str, ground_truth, data_source, extra_info):
    format_score = 0
    
    # 检查基本格式
    think_pattern = r'<think>(.*?)</think>'
    answer_pattern = r'<answer>(.*?)</answer>'
    confidence_pattern = r'<confidence>(.*?)</confidence>'
    
    think_match = re.search(think_pattern, solution_str, re.DOTALL)
    answer_match = re.search(answer_pattern, solution_str, re.DOTALL)
    confidence_match = re.search(confidence_pattern, solution_str, re.DOTALL)
    
    # 如果缺少必要的标签，format得分为0，并提前返回
    if (think_match and answer_match and confidence_match):
        # 检查think部分是否包含所有必需的子元素
        think_content = think_match.group(1)
        required_elements = ['element', 'human', 'context', 'interaction', 'analysis']
        format_correct = True
        
        for element in required_elements:
            pattern = f'<{element}>(.*?)</{element}>'
            if not re.search(pattern, think_content, re.DOTALL):
                format_correct = False
                break
        
        # 如果格式不正确，返回0分
        if format_correct:
            # 提取answer和confidence（此时可以安全提取，因为已经检查过存在）
            answer_text = answer_match.group(1).strip()
            confidence_text = confidence_match.group(1).strip()
            
            # 处理answer：去掉标点，小写，分割
            answer_clean = answer_text.translate(str.maketrans('', '', string.punctuation)).lower()
            answer_words = answer_clean.split()
            
            # 判断答案是否正确
            ground_truth_lower = ground_truth.lower()
            i = 1 if len(answer_words) == 1 and ground_truth_lower == answer_words[0] else 0
            
            # 处理confidence
            try:
                c = float(confidence_text)
                # 检查confidence是否在有效范围内
                if 0 < c < 1:
                    format_score = 1
            except (ValueError, TypeError):
                pass
            
    if format_score == 1:
        confidence_score = 2*(i*math.log(c) + (1-i)*math.log(1-c))
        return {
            "score": format_score + confidence_score + i,
            "format_score": format_score,
            "confidence_score": confidence_score,
            "correctness": i,
            "brier_score": -(c - i) ** 2,
            "confidence": c,
        }
    else:
        return {
            "score": -5,
            "format_score": 0,
            "confidence_score": -5,
            "correctness": 0,
            "brier_score": -1,
            "confidence": 1,
        }
        

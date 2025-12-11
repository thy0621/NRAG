#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Annotated, Union

import os

import json

from tqdm import tqdm
import pandas as pd


os.environ["CUDA_VISIBLE_DEVICES"] ='1'

import typer
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

app = typer.Typer(pretty_exceptions_show_locals=False)


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def load_model_and_tokenizer(model_dir: Union[str, Path]) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True
    )
    return model, tokenizer

def main_muti_output(dis_name,datasets_path,folder_path,checkpoint_path, test_file):

    model_dir = '/code/ChatGLM3/finetune_demo/output/2025/' + folder_path +'/' + checkpoint_path
    model, tokenizer = load_model_and_tokenizer(model_dir)
    # response, _ = model.chat(tokenizer, prompt)
    # print(response)


    data_path ='/code/ChatGLM3/finetune_demo/data/' + datasets_path + test_file

    history = []
    result_list = []
    print("欢迎使用 ChatGLM-6B 智能测试助手，测试开始～")
    with open(data_path, "r", encoding="utf-8-sig") as f:
        result_ = []
        
        for line in f:
            # 去除可能存在的换行符
            line = line.strip()
            # 解析每一行的JSON数据
            data = json.loads(line)
            # 在这里处理你的数据
            result_.append(data)

        json_list = result_ # json.load(f)
        json_size = int(len(json_list))
        
        try:
            for index_ in tqdm(range(100)): # demo  or json_size
                query = json_list[index_]['conversations'][0]['content']
                
                response, history = model.chat(tokenizer, query,history=[])
                # print(response)
                gold_summary = json_list[index_]['conversations'][1]['content']
                # if len(response['name']) and len(response['content'])>0: 
                #     result_list.append({'content':query, 'summary_name':response['name'],
                #                         'summary_content':response['content'],
                #                         'gold_summary':gold_summary})
                result_list.append({'content':query, 'summary':response,
                                        
                                        'gold_summary':gold_summary})
                history = []
                # if index_%10 == 0: print(index_,' over')
        except Exception as e:
            print('Error:', e)
        finally:
            print('finally...')
            with open(model_dir + '/muti_output'+ '' + '.json', 'w', encoding='utf-8') as f:
                json.dump(result_list, f,ensure_ascii=False)
            pd_result = pd.DataFrame(result_list)
            pd_result.to_excel(model_dir + '/muti_output'+ '' + '.xlsx',index=False, encoding='utf_8_sig')
            pd_result.to_excel('/code/ChatGLM3/finetune_demo/output/2025/'+ folder_path + '/muti_output'+ '' + '.xlsx',index=False, encoding='utf_8_sig')


@app.command()
def main():
    
    # for item in range(0,6,1):
    #     index_ = str(item*500+7000)
    #     name_str = 'test-' + index_
    #     checkpoint_str = 'checkpoint-' + index_
    #     main_muti_output('NRAG','demo','demo',checkpoint_str, '/sjwkjz-glm3-forchatglm3base-20251010.json')

    main_muti_output('NRAG','demo','demo','checkpoint-10000', '/sjwkjz-glm3-forchatglm3base-20251010.json')

    

if __name__ == '__main__':
    app()

# nohup python inference_hf2.py > output/2025/demo/muti_output_pt2_hf2_20251010.log 2>&1 &
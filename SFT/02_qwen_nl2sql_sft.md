
**1. 准备数据**

```
dev.json： https://bird-bench.github.io/
cosql_train.json：https://www.modelscope.cn/datasets/yuchen/CoSQL/files


原始数据集文件： cosql_train.json \ dev.json
转换脚本： convert_cosql_to_sharegpt.py \ convert_spider_to_alpaca.py
转后后的数据集： cosql_train_sharegpt.json \ dev_alpaca.json

执行指令：

python convert_cosql_to_sharegpt.py
python convert_spider_to_alpaca.py

```

**2. 注册数据集**

```
(llama_factory) root@autodl-container-9b384da211-e544ce37:~/autodl-tmp/LLaMA-Factory/data# head -n 20 dataset_info.json
{
  "dev_alpaca": {
    "file_name": "dev_alpaca.json"
  },
  "cosql_train_sharegpt": {
    "file_name": "cosql_train_sharegpt.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  },
  "identity": {
    "file_name": "identity.json"
  },

```

**3. 微调**

```
# 下载模型

modelscope download --model  Qwen/Qwen2.5-7B-Instruct --local-dir ./qwen25_7b_instruct/


## 执行微调

llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /root/autodl-tmp/qwen25_7b_instruct \
    --dataset cosql_train_sharegpt,dev_alpaca \
    --dataset_dir /root/autodl-tmp/LLaMA-Factory/data \
    --template qwen \
    --finetuning_type lora \
    --output_dir ./save06/Qwen2.5-7B/lora/train_nl2sql \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 50 \
    --eval_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-4 \
    --num_train_epochs 5.0 \
    --max_samples 1000 \
    --val_size 0.1 \
    --plot_loss \
    --fp16
```

**4. 验证**

```
# 启用chat模式，获取微调前的qwen模型的输出
llamafactory-cli chat \
    --model_name_or_path /root/autodl-tmp/qwen25_7b_instruct
	
	
User: Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.
Assistant: To provide you with accurate information ...

# 启用chat模式，获取微调后的qwen模型的输出，验证是否生效
llamafactory-cli chat \
    --model_name_or_path /root/autodl-tmp/qwen25_7b_instruct \
    --adapter_name_or_path ./save06/Qwen2.5-7B/lora/train_nl2sql \
    --template qwen \
    --finetuning_type lora


User: Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.
Assistant: SELECT T2.FreeRate FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.`Type` = 'Continuation' AND T2.Ages = '5-17' ORDER BY T2.FreeRate ASC LIMIT 3


```

**5. 合并保存全部完整权重，用于推理**

```
llamafactory-cli export \
    --model_name_or_path /root/autodl-tmp/qwen25_7b_instruct \
    --adapter_name_or_path ./save06/Qwen2.5-7B/lora/train_nl2sql \
    --template qwen \
    --finetuning_type lora \
    --export_dir ./save06/Qwen2.5-7B/full/train_nl2sql_merged \
    --export_size 2 \
    --export_legacy_format False
```


```
llamafactory-cli chat --model_name_or_path  ./save06/Qwen2.5-7B/full/train_nl2sql_merged/

User: Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.
Assistant: SELECT T2.FreeRate FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.Ages = '5-17' AND T2.`Charter Type` = 'Continuation' ORDER BY T2.FreeRate ASC LIMIT 3



```


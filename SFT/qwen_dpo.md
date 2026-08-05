
准备数据集

```
源数据集：
dpo_zh.json:https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k/blob/main/dpo_zh.json
train-00000-of-00001.parquet: https://www.modelscope.cn/datasets/baierfa/DPO-En-Zh-20k-handbook/files

默认dpo_zh.json已经sharegpt格式，无需修改，改名为 dpo_zh_sharegpt.json

另一个的转换脚本：convert_zh_to_alpaca.py，转换后，名为：dpo_zh_alpaca.json

注册数据集：

  "dpo_zh_alpaca": {
    "file_name": "dpo_zh_alpaca.json",
    "ranking": true,
    "columns": {
      "prompt": "instruction",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  },
  "dpo_zh_sharegpt": {
    "file_name": "dpo_zh_sharegpt.json",
    "ranking": true,
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  },



```

准备模型


```
modelscope download --model  Qwen/Qwen2.5-0.5B-Instruct --local-dir ./Qwen/Qwen2.5-0.5B-Instruct/
```


RLHF DPO 执行微调

```

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DISABLE_VERSION_CHECK=1 \ 
nohup llamafactory-cli train --stage dpo --do_train True  \
--model_name_or_path /root/autodl-tmp/Qwen/Qwen2.5-0.5B-Instruct  \
--preprocessing_num_workers 8      \
--finetuning_type lora  \
--template qwen      \
--flash_attn auto      \
--dataset_dir data  \
--dataset dpo_zh_sharegpt,dpo_zh_alpaca      \
--cutoff_len 1536      \
--learning_rate 5e-05      \
--num_train_epochs 1.0      \
--max_samples 35000      \
--per_device_train_batch_size 1      \
--gradient_accumulation_steps 16      \
--lr_scheduler_type cosine      \
--max_grad_norm 1.0      \
--logging_steps 5      \
--save_strategy no      \
--warmup_steps 0      \
--packing False      \
--report_to none      \
--output_dir saves/Qwen2.5-0.5B-Instruct/lora/train_2026      \
--bf16 True      \
--plot_loss True      \
--ddp_timeout 180000000      \
--optim adamw_torch      \
--lora_rank 8      \
--lora_alpha 16      \
--lora_dropout 0      \
--lora_target all      \
--pref_beta 0.1      \
--pref_ftx 0      \
--gradient_checkpointing True > dpo_train_log.txt 2>&1 &


```

Before-After测试

```
llamafactory-cli chat --model_name_or_path /root/autodl-tmp/Qwen/Qwen2.5-0.5B-Instruct/

User: 短文：小明早上起床，刷牙、洗脸，然后背着书包去上学。
Assistant: 早上，小明会先刷牙和洗脸，这样可以保持口腔和皮肤的清洁。然后，他会背着书包去上学，这个过程既节省了时间，又方便了学习。

User: 问题：小明早上做了什么？简单写两句。
Assistant: 小明早上刷牙、洗脸。

User: 最近，马其顿流行歌手阿美莉卡在一场巡演展览中被纪念
Assistant: 阿美莉卡的演唱在巡演展览中受到了热烈的欢迎，成为了这个季节的热门话题。

User: “春风又绿江南岸” 中 “绿” 历来被人称道，如果换成 “到”“过”“入”，分别缺少什么表达效果？
Assistant: 如果将“春风又绿江南岸”中的“绿”换成“到”“过”“入”，将会产生以下效果：

1. **“到”**：如果换成“到”，表达效果可能会更明确和直接，意味着春天的到来是经过一段时间的准备和积累的。例如：“春风又绿江南岸，到春天来了。”这样的表达方式更直接，也更具有感染力。

2. **“过”**：如果换成“过”，表达效果可能会更加含蓄和隐晦，意味着春天的到来是一个渐进的过程。例如：“春风又绿江南岸，过了春天来了。”这样的表达方式更富有诗意和韵味，能够更好地传达春天的到来和变化。

3. **“入”**：如果换成“入”，表达效果可能会显得有些消极，表示春天的到来是突然的和不可预知的。例如：“春风又绿江南岸，入了春天来了。”这样的表达方式可能让人感觉有些压抑和悲观。

在实际使用中，选择哪个表达效果会根据说话者的意图和听众的期待而定。如果强调春天的到来是渐进的和持续的，那么“到”“过”“入”可能会更有表达效果。但如果强调春天的到来是突然和不可预知的，那么“春风又绿江南岸，入了春天来了”可能更合适。


```


```
llamafactory-cli chat --model_name_or_path /root/autodl-tmp/Qwen/Qwen2.5-0.5B-Instruct/  \
--adapter_name_or_path ./saves/Qwen2.5-0.5B-Instruct/lora/train_2026/  \
--finetuning_type lora --template qwen  \
--trust_remote_code true

User: 短文：小明早上起床，刷牙、洗脸，然后背着书包去上学。
Assistant: 小明的上学过程如下：
1. 洗牙：小明先刷完牙，清洁口腔，保持口腔健康。
2. 洗脸：小明洗脸，清洁面部，保持面部清洁。
3. 带书包：小明背上书包，准备去上学。
4. 学校：小明在前往学校的路上或上学时，背着书包，进行准备和上学的活动。
小明的上学过程展示了日常生活的各个方面，包括清洁、卫生、准备和学习。

User: 问题：小明早上做了什么？简单写两句。
Assistant: 小明早上刷牙、洗脸；背着书包上学。

User: “春风又绿江南岸” 中 “绿” 历来被人称道，如果换成 “到”“过”“入”，分别缺少什么表达效果？
Assistant: 在“春风又绿江南岸”这句话中，“绿”本来用来形容自然景象，比如“春风又绿江南岸”，这里的“绿”是用来描述自然景色的，但换成“到”“过”“入”后，这些词语的表达效果大不相同。

- “到”：这里的“到”意为到达，与“绿”形成对比，突出“春风又绿”所展示出的生机勃勃的景象。
- “过”：这里的“过”意为经过，与“绿”形成反差，进一步突出春天的景象。
- “入”：这里的“入”意为进入，与“绿”形成鲜明对比，突出春天带来的新活力和生机。

通过这些变化，这个句子的表达效果会更加生动和形象。

```

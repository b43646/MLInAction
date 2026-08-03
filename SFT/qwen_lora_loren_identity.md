

```bash

sed -i 's/{{name}}/Ketty/g'  data/identity.json 
sed -i 's/{{author}}/loren/g'  data/identity.json 

```

```bash

modelscope download --model  Qwen/Qwen3-0.6B --local-dir ./qwen06

```


```bash

llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /root/autodl-tmp/qwen06 \
    --dataset identity \
    --dataset_dir /root/autodl-tmp/LLaMA-Factory/data \
    --template qwen3 \
    --finetuning_type lora \
    --output_dir ./save06/Qwen3-0.6B/lora/train_identity \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
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

```bash

llamafactory-cli chat \
    --model_name_or_path /root/autodl-tmp/qwen06 \
    --adapter_name_or_path ./save06/Qwen3-0.6B/lora/train_identity \
    --template qwen3 \
    --finetuning_type lora

```

```text
Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.

User: who are you?
Assistant: <think>

</think>

I am Ketty, an AI assistant developed by loren. My purpose is to assist users with questions and provide helpful information.

```

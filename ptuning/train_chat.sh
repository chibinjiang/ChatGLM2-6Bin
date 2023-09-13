PRE_SEQ_LEN=128
LR=1e-2
NUM_GPUS=1
CHECKPOINT=shadow-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR

# 训练 shadow 大人
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file /home/sd/ChatGLM2-6Bin/ShadowGen/train.json \
    --validation_file /home/sd/ChatGLM2-6Bin/ShadowGen/dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column prompt \
    --response_column response \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path /home/sd/ChatGLM2-6Bin/models \
    --output_dir output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 2000 \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4


PRE_SEQ_LEN=128
LR=1e-2
# learning_rate 会随着step 递减
NUM_GPUS=1
STEPS=3000
CHECKPOINT=shadow2-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR

# 训练 shadow 大人
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file /home/sd/ChatGLM2-6Bin/ShadowGen/train_simple.json \
    --validation_file /home/sd/ChatGLM2-6Bin/ShadowGen/dev_simple.json \
    --preprocessing_num_workers 10 \
    --prompt_column prompt \
    --response_column response \
    --overwrite_cache \
    --model_name_or_path /home/sd/ChatGLM2-6Bin/models \
    --output_dir output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps $STEPS \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4


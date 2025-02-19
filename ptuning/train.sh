PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1
STEPS=3000
CHECKPOINT=adgen-chatglm2-6b-pt-2-$PRE_SEQ_LEN-$LR


torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file /home/sd/ChatGLM2-6Bin/AdvertiseGen/train_all.json \
    --validation_file /home/sd/ChatGLM2-6Bin/AdvertiseGen/dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column summary \
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


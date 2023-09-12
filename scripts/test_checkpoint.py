import os
import time
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def main():
    start_time = time.time()
    model_path = '../models'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)  # pre_seq_len 同训练用的
    model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
    print(f"[*] Load Model DONE: {time.time() - start_time} Seconds")
    # load checkpoints
    start_time = time.time()
    CHECKPOINT_PATH = '../ptuning/output/adgen-chatglm2-6b-pt-128-2e-2/checkpoint-1000'
    prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            _key = k[len("transformer.prefix_encoder."):]
            print("Add Key: ", _key)
            new_prefix_state_dict[_key] = v
    print(f"[*] Load Checkpoints DONE: {time.time() - start_time} Seconds")
    CHECKPOINT_PATH = '../ptuning/output/adgen-chatglm2-6b-pt-128-2e-2/checkpoint-2000'
    prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            _key = k[len("transformer.prefix_encoder."):]
            print("Add Key: ", _key)
            new_prefix_state_dict[_key] = v
    print(f"[*] Load Checkpoints DONE: {time.time() - start_time} Seconds")

    # 量化 4 bit
    start_time = time.time()
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    model = model.half().cuda()
    model.transformer.prefix_encoder.float()
    model = model.eval()
    print(f"[*] 4 Bit Q DONE: {time.time() - start_time} Seconds")
    # generate text

    def ask(question, history=[]):
        epoch = time.time()
        response, new_history = model.chat(tokenizer, question, history=history)
        print(f">>> 问: {question}")
        print(f">>> 答: {response}")
        print(f"\t It takes {time.time() - epoch} Seconds")
        return [t for t in new_history]

    context = ask("你好")
    context = ask("我不喜欢这款休闲裤", context)
    context = ask("咱们能不能不聊裤子", context)
    context = ask("进程和线程有什么区别", context)
    context = ask("Python的深拷贝和浅拷贝有什么区别", context)


if __name__ == '__main__':
    main()

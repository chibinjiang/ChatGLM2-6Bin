import os
import time
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def main():
    start_time = time.time()
    model_path = '../models'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, revision="")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128, revision="")
    # pre_seq_len 同训练用的
    model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True, revision="")
    # 设置 revision = ""
    # 避免: Explicitly passing a `revision` is encouraged
    # when loading a model with custom code to ensure no malicious code has been contributed in a newer revision.
    print(f"[*] Load Model DONE: {time.time() - start_time} Seconds")
    # load checkpoints
    start_time = time.time()
    # CHECKPOINT_PATH = '../ptuning/output/adgen-chatglm2-6b-pt-2-128-2e-2/checkpoint-2000'
    # CHECKPOINT_PATH = '../ptuning/output/adgen-chatglm2-6b-pt-128-2e-2/checkpoint-1000'
    CHECKPOINT_PATH = '../ptuning/output/shadow-chatglm2-6b-pt-128-1e-2/checkpoint-2000'
    prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            _key = k[len("transformer.prefix_encoder."):]  # 1 2 both are embedding.weight
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

    # context = ask("你好")  # 需要固定的开场白
    # context = ask("我想选一条休闲裤.", context)
    # context = ask("我不喜欢这款休闲裤.", context)
    # context = ask("咱们能不能不聊裤子!", context)
    # context = ask("进程和线程有什么区别?", context)
    # context = ask("Python的深拷贝和浅拷贝有什么区别?", context)
    # context = ask("写一首表达描写大漠中男儿出塞的七律诗.", context)
    # context = ask("99 乘以2 再减去100 等于多少?", context)
    context = ask("你是谁?")  # 需要固定的开场白
    context = ask("参见暗影大人.", context)
    context = ask("不愧是暗影大人.", context)
    context = ask("暗影大人在计划着什么?", context)
    context = ask("进程和线程有什么区别?", context)
    context = ask("Python的深拷贝和浅拷贝有什么区别?", context)
    context = ask("写一首表达描写大漠中男儿出塞的七律诗.", context)
    context = ask("99 乘以2 再减去100 等于多少?", context)


if __name__ == '__main__':
    main()

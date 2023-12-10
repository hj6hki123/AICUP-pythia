import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig,BitsAndBytesConfig
from datasets import load_dataset, Features, Value
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from utility.preprocessing import *
from dotenv import load_dotenv
import wandb 
from torch.cuda.amp import GradScaler, autocast
from utility.dataloader import CustomDataLoader, CustomBatchSampler
import config


args = config.config()

## model parameters
cache_dir = r'./MODELS'
BATCH_SIZE = args.batch_size #{modify} batch size
EPOCHS = args.epochs 
base_model = f'EleutherAI/{args.model_name}'  #{modify}模型名稱
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## tokenizer parameters
bos = '<|endoftext|>' 
eos = '<|END|>'
pad = '<|pad|>'
sep ='\n\n####\n\n'


## loading tokenizer

special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad, 'sep_token': sep}
tokenizer = AutoTokenizer.from_pretrained(base_model, revision="step3000",cache_dir = cache_dir, timeout=30.0)
tokenizer.padding_side = 'left'
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

## initailize 
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_TOKEN"))
wandb.init(project="AICUP",config={"batch_size":BATCH_SIZE,"epochs":EPOCHS,"model":base_model,"device":device,"cache_dir":cache_dir,"lr":3e-5})


#load datasets
format_data(args.train_file_dir,# {fix} 訓練資料夾路徑
            args.answer_path,# {fix} answer 標註檔案路徑
            args.trainer_output_file)# {fix} 訓練格式 tsv輸出路徑
dataset = load_dataset("csv", data_files=args.trainer_output_file, delimiter='\t',
                       features = Features({
                              'fid': Value('string'), 'idx': Value('int64'),
                              'content': Value('string'), 'label': Value('string')}),
                              column_names=['fid', 'idx', 'content', 'label'], keep_default_na=False)
train_data = list(dataset['train'])

bucket_train_dataloader = CustomDataLoader(train_data, tokenizer, BATCH_SIZE)

## 4-bit quantization configuration
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)


print(f"{tokenizer.pad_token}: {tokenizer.pad_token_id}")
config = AutoConfig.from_pretrained(base_model,
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    sep_token_id=tokenizer.sep_token_id,
                                    output_hidden_states=False)

## loading Pythia model
model_params = {
    "revision": "step3000",
    "config": config,
    "cache_dir": cache_dir
}

if args.lora:
    model_params["quantization_config"] = quant_config

model = AutoModelForCausalLM.from_pretrained(base_model, **model_params)
model.config.use_cache = False
print(model)

## PEFT parameters 
if args.lora:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_key_value", 'dense', 'dense_h_to_4h', 'dense_4h_to_h'],
    )
    model = get_peft_model(model,peft_config)
    model.print_trainable_parameters()

## model setting
optimizer = AdamW(model.parameters(),lr=3e-5) 
#criterion = torch.nn.CrossEntropyLoss()

model.resize_token_embeddings(len(tokenizer))
model.to(device)





## model training
model_name = args.model_name
model_dir = f"./FineTuning/{model_name}"
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
min_loss = float('inf')

scaler = GradScaler()
global_step = 0
total_loss = 0
accumulation_steps = 4  # 調整為合適的梯度累積步驟

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    s_data_loader = tqdm(bucket_train_dataloader, desc=f"Training Epoch {epoch}")

    
    for step, (seqs, labels, masks) in enumerate(s_data_loader):
        seqs, labels, masks = seqs.to(device), labels.to(device), masks.to(device)
        optimizer.zero_grad()
        with autocast():  # 啟用混合精度
            outputs = model(seqs, labels=labels, attention_mask=masks)
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        s_data_loader.set_postfix({"Loss": loss.item()})
        wandb.log({"step Loss": loss.item()})

    avg_train_loss = total_loss / len(bucket_train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))
    wandb.log({"epoch": epoch, "Loss": avg_train_loss})

    # 儲存模型
    torch.save(model.state_dict(), os.path.join(model_dir, 'GPT_Finial.pt'))
    if avg_train_loss < min_loss:
        min_loss = avg_train_loss
        torch.save(model.state_dict(), os.path.join(model_dir, 'GPT_best.pt'))
 




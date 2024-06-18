from datasets import load_dataset, Features, Value ,Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import io,os
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from utility.postprocessing import *
import config

## model parameters
args = config.config()
cache_dir = r'./MODELS'   
base_model = f'EleutherAI/{args.model_name}'  #{modify}模型名稱
model_name = args.model_name # {fix}微調模型名稱
model_dir = f"./FineTuning/{model_name}" # {fix}模型儲存路徑
BATCH_SIZE = args.batch_size #{modify} batch size
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



## 4-bit quantization configuration
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

## loading Pythia model
config = AutoConfig.from_pretrained(base_model,
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    sep_token_id=tokenizer.sep_token_id,
                                    output_hidden_states=False)
model_params = {
    "revision": "step3000",
    "config": config,
    "cache_dir": cache_dir
}

if args.lora:
    model_params["quantization_config"] = quant_config

model = AutoModelForCausalLM.from_pretrained(base_model, **model_params)


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
model.resize_token_embeddings(len(tokenizer))
model.config.use_cache = True

model.load_state_dict(torch.load(os.path.join(model_dir , 'GPT_best.pt')))
model = model.to(device)

## load datasets
test_phase_path = args.valid_file_dir#{fix} test data path
valid_out_file_path = args.valid_output_file #{fix} test tsv data path
test_txts = list(map(lambda x:os.path.join(test_phase_path , x) , os.listdir(test_phase_path)))
test_txts = sorted(test_txts)
valid_data = process_valid_data(test_txts , valid_out_file_path)



valid_data = load_dataset("csv", data_files=valid_out_file_path, delimiter='\t',
                          features = Features({
                              'fid': Value('string'), 'idx': Value('int64'),
                              'content': Value('string'), 'label': Value('string')}),
                              column_names=['fid', 'idx', 'content', 'label'])
valid_list= list(valid_data['train'])


## predict
with open("./answer.txt",'w',encoding='utf8') as f:
    for i in tqdm(range(0, len(valid_list), BATCH_SIZE)):
        with torch.no_grad():
            seeds = valid_list[i:i+BATCH_SIZE]
            outputs = predict(model, tokenizer, seeds)
            print(outputs)
            for o in outputs:
                f.write(o)
                f.write('\n')
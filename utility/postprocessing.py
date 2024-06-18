import os
import re
import torch

def read_file(path):
    with open(path, 'r', encoding='utf-8-sig') as fr:
        return fr.readlines()

def process_valid_data(test_txts, out_file):
    output_data = []
    for txt in test_txts:
        m_report = read_file(txt)
        boundary = 0
        fid, _ = os.path.splitext(os.path.basename(txt))
        for sent in m_report:
            if sent.strip(): 
                sent = sent.replace('\t', ' ')
                output_data.append(f"{fid}\t{boundary}\t{sent}")

            boundary += len(sent)

    with open(out_file, 'w', encoding='utf-8') as fw:
        fw.writelines(output_data)  




train_phi_category = ['PATIENT', 'DOCTOR', 'USERNAME',
             'PROFESSION',
             'ROOM', 'DEPARTMENT', 'HOSPITAL', 'ORGANIZATION', 'STREET', 'CITY', 'STATE', 'COUNTRY', 'ZIP', 'LOCATION-OTHER',
             'AGE',
             'DATE', 'TIME', 'DURATION', 'SET',
             'PHONE', 'FAX', 'EMAIL', 'URL', 'IPADDR',
             'SSN', 'MEDICALRECORD', 'HEALTHPLAN', 'ACCOUNT', 'LICENSE', 'VEHICLE', 'DEVICE', 'BIOID', 'IDNUM']

def get_anno_format(sentence , infos , boundary):  #(原文, 預測結果, 原文起始位置)
    anno_list = []
    lines = re.split('\n|\\\\n', infos)
    normalize_keys = ['DATE' , "TIME" , "DURATION" , "SET"]
    phi_dict = {}
    for line in lines:
        parts = line.split(":")
        if len(parts) < 2:
            continue
        if parts[0] not in train_phi_category or parts[1] == '':
            continue
        if len(parts) == 2:
            phi_dict[parts[0]] = parts[1].strip()
    for phi_key, phi_value in phi_dict.items():
        normalize_time = None
        if phi_key in normalize_keys:
            if '=>' in phi_value:
                temp_phi_values = phi_value.split('=>')
                phi_value = temp_phi_values[0]
                normalize_time = temp_phi_values[-1]
            else:
                normalize_time = phi_value
        try:
            matches = [(match.start(), match.end()) for match in re.finditer(phi_value, sentence)]
        except:
            continue
        for start, end in matches:
            if start == end:
                continue
            item_dict = {
                        'phi' : phi_key,
                        'st_idx' : start + int(boundary),
                        'ed_idx' : end + int(boundary),
                        'entity' : phi_value,
            }
            if normalize_time is not None:
                item_dict['normalize_time'] = normalize_time
            anno_list.append(item_dict)
    return anno_list

def predict(model, tokenizer, input, template = "<|endoftext|> __CONTENT__\n\n####\n\n"):
    seeds = [] # ['<|endoftext|>  433475.RDC\n\n####\n\n'
    for data in input:
        if data['content'] == None:
            print(data)
        else:
            seeds.append(template.replace("__CONTENT__", data['content']))
    sep = tokenizer.sep_token
    eos = tokenizer.eos_token
    pad = tokenizer.pad_token
    pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    """Generate text from a trained model."""
    model.eval()
    device = model.device
    texts = tokenizer(seeds, return_tensors = 'pt', padding=True).to(device)
    outputs = []
    #return
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**texts, max_new_tokens=400, pad_token_id = pad_idx,
                                        eos_token_id=tokenizer.convert_tokens_to_ids(eos))
        preds = tokenizer.batch_decode(output_tokens)
        for idx , pred in enumerate(preds):
            if "NULL" in pred:
                continue
            phi_infos = pred[pred.index(sep)+len(sep):].replace(pad, "").replace(eos, "").strip()
            annotations = get_anno_format(input[idx]['content'] , phi_infos , input[idx]['idx'])

            for annotation in annotations:
                if 'normalize_time' in annotation:
                    outputs.append(f'{input[idx]["fid"]}\t{annotation["phi"]}\t{annotation["st_idx"]}\t{annotation["ed_idx"]}\t{annotation["entity"]}\t{annotation["normalize_time"]}')
                else:
                    outputs.append(f'{input[idx]["fid"]}\t{annotation["phi"]}\t{annotation["st_idx"]}\t{annotation["ed_idx"]}\t{annotation["entity"]}')
    return outputs




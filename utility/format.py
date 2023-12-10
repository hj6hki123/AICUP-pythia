import pandas as pd
from pathlib import Path

output_file = 'tsv/Second_Phase_Dataset.tsv'
folder_path = 'AICUP/Second_Phase_Dataset/Second_Phase_Dataset/Second_Phase_Text_Dataset'

def extract_number_from_filename(file_path):
    return int(''.join(filter(str.isdigit, file_path.stem)))

# 創建一個空的 DataFrame
df = pd.DataFrame(columns=['fid', 'idx', 'context', 'label'])
data_list = []

file_list = sorted(Path(folder_path).rglob('*.txt'), key=extract_number_from_filename)

for file_path in file_list:
    with open(file_path, 'r', encoding='utf-8') as file:
        char_position = 0
        for idx, line in enumerate(file, start=1):
            line = line.strip().replace('\t', ' ')
            if line:
                data_list.append({'fid': file_path.stem, 'idx': char_position + 1, 'context': line, 'label': 'PHI: NULL'})
                char_position += len(line)

df = pd.DataFrame(data_list)

with open('AICUP/Second_Phase_Dataset/Second_Phase_Dataset/answer.txt', 'r') as ansfile:
    lines = ansfile.readlines()

fid_set = set(df['fid'].values)

for line in lines:
    splite = line.split('\t')
    fid, label, context = splite[0], splite[1], splite[4]
    normalize = splite[5] if len(splite) > 5 else 'NULL'
    context = context.strip()

    if fid in fid_set:
        matching_rows = df[(df['fid'] == fid) & (df['context'].str.contains(context, regex=False))]
        if not matching_rows.empty:
            if normalize != 'NULL':
                context += f'=>{normalize.strip()}'
            row_index = matching_rows.index[0]
            select = df.at[row_index, 'label']
            df.at[row_index, 'label'] = f'{label}: {context}' if select == 'PHI: NULL' else f'{select}\\n{label}: {context}'

# 將結果保存到 tsv 檔案
df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')

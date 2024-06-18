from typing import List, Dict, Iterator
import torch
import random
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

class CustomDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, tokenizer: PreTrainedTokenizer, batch_size: int, template: str = " __CONTENT__\n\n####\n\n__LABEL__ "):
        self.tokenizer = tokenizer
        self.template = template
        self.batch_sampler = CustomBatchSampler(dataset, batch_size)
        super().__init__(dataset, batch_sampler=self.batch_sampler, collate_fn=self.collate_fn)

    def collate_fn(self, batch: List[Dict[str, str]]) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        IGNORED_PAD_IDX = -100

        # 確保每批資料中包含 'content' 和 'label'
        assert all('content' in data and 'label' in data for data in batch), "Batch data must contain 'content' and 'label' keys"

        # template進行替換
        texts = [self.template.replace("__LABEL__", data['label']).replace("__CONTENT__", data['content']) for data in batch]
        encoded_seq = self.tokenizer(texts, padding=True, return_tensors='pt')

        # 處理張量
        attention_mask = encoded_seq['attention_mask']
        encoded_label = encoded_seq['input_ids'].clone()
        encoded_label[encoded_label == self.tokenizer.pad_token_id] = IGNORED_PAD_IDX

        return encoded_seq['input_ids'], encoded_label, attention_mask

class CustomBatchSampler():
    def __init__(self, data: Dataset, batch_size: int):
        self.data = data
        self.batch_size = batch_size
        self.len = len(data)  

    def __iter__(self) -> Iterator[List[int]]:
        indices = [(index, len(datum["content"])) for index, datum in enumerate(self.data)]
        random.shuffle(indices)

        # 對索引按內容長度進行分組排序
        pooled_indices = sorted(indices, key=lambda x: x[1], reverse=True)
        pooled_indices = [index for index, _ in pooled_indices]

        # 產生批次
        for i in range(0, len(pooled_indices), self.batch_size):
            yield pooled_indices[i:i + self.batch_size]

    def __len__(self) -> int:
        return (self.len + self.batch_size - 1) // self.batch_size

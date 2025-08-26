import numpy as np
import torch
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from collections import Counter
import pickle


class BatchLoaderData(Dataset):
    def __init__(self,
                 pickle_file):
        super(BatchLoaderData, self).__init__()

        self.label_to_index = self.label_to_index = [
            'O', 'B-陶瓷器类', 'I-陶瓷器类', 'B-陶瓷原料', 'I-陶瓷原料',
            'B-陶瓷文博', 'I-陶瓷文博', 'B-陶瓷文献', 'I-陶瓷文献',
            'B-窑业遗存', 'I-窑业遗存', 'B-陶瓷装饰', 'I-陶瓷装饰',
            'B-陶瓷人物', 'I-陶瓷人物', 'B-陶瓷教育机构', 'I-陶瓷教育机构',
            'B-地点', 'I-地点', 'B-历史时期', 'I-历史时期',
            '[CLS]', '[SEP]', '[PAD]'
        ]
        self.label_map = {label: i for i, label in enumerate(self.label_to_index)}
        self.dataset_dict = self.load_pickle(pickle_file)

    def __getitem__(self, index):
        dataset = self.dataset_dict[index]

        bert_ids = dataset['input_ids']

        bert_mask = dataset['attention_mask']

        rnn_array = dataset['rnn_array']
        # rank_array = dataset['rank_array']

        ner_labels = dataset['ner_labels']
        ner_labels = [self.label_map[label] for label in ner_labels]
        ner_labels = ner_labels

        predict_mask = dataset['predict_mask']

        return bert_ids, bert_mask, predict_mask, rnn_array, ner_labels

    def __len__(self):
        return len(self.dataset_dict)

    def load_pickle(self, pickle_file):
        open_file = open(pickle_file, 'rb')
        pickle_data = pickle.load(open_file)
        return pickle_data

    @classmethod
    def pad_self(cls, batchs):
        def pad_function(index, seq_len, all_data):
            one_data = all_data[index]
            if len(one_data) < seq_len:
                one_data = one_data + [0] * (seq_len - len(one_data))
            return one_data[:seq_len]  # Ensure we are slicing to the max length

        def pad_function_array(index, seq_len, all_batch):
            results = []
            for sample in all_batch:
                one_data = sample[index]
                last_len = seq_len - len(one_data)
                if last_len > 0:
                    pad = [0] * 200
                    pads = [pad] * last_len
                    one_data = one_data + pads
                results.append(one_data[:seq_len])  # Ensure we are slicing to the max length
            results = np.array(results).astype(np.float32)
            results = torch.tensor(results, dtype=torch.float32)
            return results

        # Ensure batchs is not empty and contains the correct data
        if not batchs or not isinstance(batchs, list) or not isinstance(batchs[0], (list, tuple)):
            raise ValueError("Invalid batchs input. It should be a list of lists/tuples.")

        seq_len_list = [len(sample[0]) for sample in batchs]
        if not seq_len_list:  # Ensure seq_len_list is not empty
            raise ValueError("seq_len_list is empty. Check your batchs data.")

        max_len = min(max(seq_len_list), 512)  # Ensure the length does not exceed the model's max length

        print(f"Calculated max_len: {max_len}")  # Debug print

        pad_functions = lambda x, seq_len: [pad_function(x, seq_len, sample) for sample in batchs]

        input_ids_list = torch.LongTensor(pad_functions(0, max_len))
        input_mask_list = torch.LongTensor(pad_functions(1, max_len))
        predict_mask_list = torch.BoolTensor(pad_functions(2, max_len))
        label_ids_list = torch.LongTensor(pad_functions(4, max_len))

        lstm_input_array = pad_function_array(index=3, seq_len=max_len, all_batch=batchs)

        return input_ids_list, input_mask_list, predict_mask_list, lstm_input_array, label_ids_list


if __name__ == '__main__':

    dataSwt = BatchLoaderData(pickle_file='pickle_data/train_data.pickle')
    print(len(dataSwt))
    dataloader = DataLoader(dataSwt, batch_size=1, shuffle=True, collate_fn=dataSwt.pad_self)
    for i, batch in enumerate(dataloader):
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2])
        print(batch[3].shape)
        print(batch[4])
        # break
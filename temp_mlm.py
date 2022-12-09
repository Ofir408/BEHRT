import sys

sys.path.insert(0, '../')
from common.common import create_folder
from common.pytorch import load_model
import pytorch_pretrained_bert as Bert
from model.utils import age_vocab
from common.common import load_obj
from dataLoader.MLM import MLMLoader
from torch.utils.data import DataLoader
import pandas as pd
from model.MLM import BertForMaskedLM
from model.optimiser import adam
import sklearn.metrics as skm
import numpy as np
import torch
import time
import torch.nn as nn
import os

class BertConfig(Bert.modeling.BertConfig):
    def __init__(self, config):
        super(BertConfig, self).__init__(
            vocab_size_or_config_json_file=config.get('vocab_size'),
            hidden_size=config['hidden_size'],
            num_hidden_layers=config.get('num_hidden_layers'),
            num_attention_heads=config.get('num_attention_heads'),
            intermediate_size=config.get('intermediate_size'),
            hidden_act=config.get('hidden_act'),
            hidden_dropout_prob=config.get('hidden_dropout_prob'),
            attention_probs_dropout_prob=config.get('attention_probs_dropout_prob'),
            max_position_embeddings=config.get('max_position_embedding'),
            initializer_range=config.get('initializer_range'),
        )
        self.seg_vocab_size = config.get('seg_vocab_size')
        self.age_vocab_size = config.get('age_vocab_size')
        self.sex_vocab_size = config.get('sex_vocab_size')


class TrainConfig(object):
    def __init__(self, config):
        self.batch_size = config.get('batch_size')
        self.use_cuda = config.get('use_cuda')
        self.max_len_seq = config.get('max_len_seq')
        self.train_loader_workers = config.get('train_loader_workers')
        self.test_loader_workers = config.get('test_loader_workers')
        self.device = config.get('device')
        self.output_dir = config.get('output_dir')
        self.output_name = config.get('output_name')
        self.best_name = config.get('best_name')

file_config = {
    'vocab': r'/home/benshoho/projects/BEHRT/my_data/vocab',  # vocabulary idx2token, token2idx
    #'data': r'/home/benshoho/projects/BEHRT/my_data/',  # formated data
    'data': r'/home/benshoho/projects/feature extraction/OMOP/data/omop_behrt_ds.csv',
    'model_path': r'/home/benshoho/projects/BEHRT/my_data',  # where to save model
    'model_name': 'my-custom-mlm-model',  # model name
    'file_name': r'/home/benshoho/projects/BEHRT/my_data/logs.txt',  # log path
}
create_folder(file_config['model_path'])

global_params = {
    'max_seq_len': 64,
    'max_age': 110,
    'month': 1,
    'age_symbol': None,
    'min_visit': 5,
    'gradient_accumulation_steps': 1
}

optim_param = {
    'lr': 3e-5,
    'warmup_proportion': 0.1,
    'weight_decay': 0.01
}

train_params = {
    'batch_size': 256,
    'use_cuda': True,
    'max_len_seq': global_params['max_seq_len'],
    'device': 'cuda:0'
}

BertVocab = load_obj(file_config['vocab'])
ageVocab, _ = age_vocab(max_age=global_params['max_age'], mon=global_params['month'],
                        symbol=global_params['age_symbol'])

data = pd.read_csv(file_config['data'])

# TODO: should return it?
#Dset = MLMLoader(data, BertVocab['token2idx'], ageVocab, max_len=train_params['max_len_seq'], code='caliber_id')
Dset = MLMLoader(data, BertVocab['token2idx'], ageVocab, max_len=train_params['max_len_seq'])
trainload = DataLoader(dataset=Dset, batch_size=train_params['batch_size'], shuffle=True, num_workers=3)

for step, batch in enumerate(trainload):
    age_ids, input_ids, posi_ids, segment_ids, attMask, masked_label = batch
    t = np.where(masked_label.cpu().numpy() > 1)[0]
    print(t)

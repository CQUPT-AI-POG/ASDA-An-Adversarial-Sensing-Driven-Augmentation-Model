import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from scipy.stats import rankdata
from tqdm import tqdm  # 用于进度条
import time  # 用于计算程序运行时间
import torch
import torch.nn as nn
import torch.nn.functional as F
class BGE_M3_Attention_Classifier(nn.Module):
    def __init__(self, num_labels=2, hidden_size=1024, dropout_prob=0.1):
        super(BGE_M3_Attention_Classifier, self).__init__()
        self.encoder = AutoModel.from_pretrained("/mnt/HDD2/Zhouxiang/bge-m3/", trust_remote_code=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_labels)
        )
    def forward(self, input_ids, attention_mask):
        # Get encoder outputs
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = encoder_outputs.last_hidden_state

        # Calculate attention weights
        attention_weights = self.attention(last_hidden_state)

        # Apply attention
        attended_features = torch.sum(attention_weights * last_hidden_state, dim=1)

        # Classification
        logits = self.classifier(attended_features)
        return logits
    
class BGE_M3_BiLSTM_Classifier(nn.Module):
    def __init__(self, num_labels=2, hidden_size=1024, lstm_hidden_size=512, num_lstm_layers=1, dropout_prob=0.1):
        """
        Args:
            num_labels (int): 分类的类别数.
            hidden_size (int): BGE-M3 模型的输出维度.
            lstm_hidden_size (int): LSTM 层的隐藏单元数.
            num_lstm_layers (int): LSTM 的层数.
            dropout_prob (float): Dropout 的概率.
        """
        super(BGE_M3_BiLSTM_Classifier, self).__init__()
        self.encoder = AutoModel.from_pretrained("/mnt/HDD2/Zhouxiang/bge-m3/", trust_remote_code=True)
        
        # BiLSTM 层
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,  # 开启双向
            batch_first=True     # 输入和输出张量的第一个维度是 batch_size
        )
        
        # 分类器
        # BiLSTM 的输出维度是 lstm_hidden_size * 2 (因为是双向)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(lstm_hidden_size * 2, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        # 1. 获取 BGE-M3 的输出
        # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        encoder_outputs =  self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = encoder_outputs.last_hidden_state

        # 2. 将输出送入 BiLSTM
        # lstm_output shape: (batch_size, sequence_length, lstm_hidden_size * 2)
        # h_n shape: (num_lstm_layers * 2, batch_size, lstm_hidden_size)
        lstm_output, (h_n, c_n) = self.bilstm(last_hidden_state)
        # 3. 获取 BiLSTM 的最终特征表示
        # 我们将最后一个时间步的前向和后向隐藏状态拼接起来作为句子的表示
        # h_n[-2,:,:] 是最后一层前向的隐藏状态
        # h_n[-1,:,:] 是最后一层后向的隐藏状态
        final_hidden_state = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        # 4. 分类
        # logits shape: (batch_size, num_labels)
        logits = self.classifier(final_hidden_state)
        
        return logits
    
class BGE_M3_TextCNN_Classifier(nn.Module):
    def __init__(self, num_labels=2, hidden_size=1024, num_filters=128, kernel_sizes=[2, 3, 4], dropout_prob=0.1):
        """
        Args:
            num_labels (int): 分类的类别数.
            hidden_size (int): BGE-M3 模型的输出维度 (CNN的输入通道数).
            num_filters (int): 每种尺寸的卷积核的数量 (CNN的输出通道数).
            kernel_sizes (list of int): 卷积核的尺寸列表.
            dropout_prob (float): Dropout 的概率.
        """
        super(BGE_M3_TextCNN_Classifier, self).__init__()
        self.encoder = AutoModel.from_pretrained("/mnt/HDD2/Zhouxiang/bge-m3/", trust_remote_code=True)
        
        # TextCNN 卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size, 
                      out_channels=num_filters, 
                      kernel_size=k) 
            for k in kernel_sizes
        ])
        
        # 分类器
        # 总特征数是 `len(kernel_sizes) * num_filters`
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(len(kernel_sizes) * num_filters, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        # 1. 获取 BGE-M3 的输出
        # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = encoder_outputs.last_hidden_state

        # 2. 调整维度以适应 Conv1d
        # Conv1d 需要的输入 shape: (batch_size, in_channels, sequence_length)
        x = last_hidden_state.permute(0, 2, 1)

        # 3. 卷积和池化
        # 对每种尺寸的卷积核进行操作
        pooled_outputs = []
        for conv in self.convs:
            # 卷积 -> 激活 -> 池化
            conved = conv(x)              # shape: (batch, num_filters, seq_len - k + 1)
            activated = F.relu(conved)    # shape: (batch, num_filters, seq_len - k + 1)
            
            # 最大池化，核的大小等于卷积输出的长度，从而在时间步上取最大值
            pooled = F.max_pool1d(activated, kernel_size=activated.shape[2]) # shape: (batch, num_filters, 1)
            squeezed = pooled.squeeze(2)  # shape: (batch, num_filters)
            pooled_outputs.append(squeezed)

        # 4. 拼接所有池化后的特征
        # concatenated shape: (batch_size, len(kernel_sizes) * num_filters)
        concatenated = torch.cat(pooled_outputs, dim=1)
        
        # 5. 分类
        # logits shape: (batch_size, num_labels)
        logits = self.classifier(concatenated)
        
        return logits
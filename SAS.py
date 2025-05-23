import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualCNN(nn.Module):
    
    def __init__(
            self, input, output, kernel_size, padding
    ):
        super().__init__()

        self.ReLU = nn.ReLU()
        self.process = nn.Conv1d(input, output, kernel_size, 1, padding)

    def forward(self, hidden_states):
        output = self.ReLU(hidden_states)
        output = self.process(output)
        output = output + hidden_states
        return output
        
class ResidualMLP(nn.Module):
    
    def __init__(
            self, input, output
    ):
        super().__init__()

        self.ReLU = nn.ReLU()
        self.process = nn.Linear(input, output)

    def forward(self, hidden_states):
        output = self.ReLU(hidden_states)
        output = self.process(output)
        output = output + hidden_states
        return output

class SAS(nn.Module):
  def __init__(self, ori_head,target_head,ori_feature,target_feature,kernel_size):
    """
    SAS attention bias module.

    Args:
      ori_head: the original head number
      target_head: the target head number
      ori_feature: the original feature size
      target_feature: the target feature size
      kernel_size: the kernel size 
    """
    super(SAS, self).__init__()

    self.ori_head=ori_head
    self.target_head = target_head
    self.ori_feature=ori_feature
    self.target_feature = target_feature
    self.kernel_size = kernel_size
    padding = (kernel_size - 1) // 2
    self.sas_q = nn.Sequential(
        nn.Conv1d(self.ori_head, self.target_head, kernel_size, 1, padding),
        ResidualCNN(self.target_head, self.target_head, kernel_size, padding),
        nn.Linear(self.ori_feature, self.target_feature), ResidualMLP(self.target_feature, self.target_feature))
    self.sas_k = nn.Sequential(
        nn.Conv1d(self.ori_head, self.target_head, kernel_size, 1, padding),
        ResidualCNN(self.target_head, self.target_head, kernel_size, padding),
        nn.Linear(self.ori_feature, self.target_feature), ResidualMLP(self.target_feature, self.target_feature))
    self.sas_v = nn.Sequential(
        nn.Conv1d(self.ori_head, self.target_head, kernel_size, 1, padding),
        ResidualCNN(self.target_head, self.target_head, kernel_size, padding),
    )

    self.output_dense=nn.Linear(self.ori_head*self.ori_feature,self.ori_head*self.ori_feature)

  def forward(self, query,key,value):
    """
    Args:
      query: query embedding,
         shape [bsz, seq_len,num_heads, hidden_size_per_head]
      key: key embedding,
         shape [bsz, seq_len,num_heads, hidden_size_per_head]
      value: value embedding,
         shape [bsz, seq_len,num_heads, hidden_size_per_head]

    Returns:
      attention: attention output
         shape [bsz, seq_len,num_heads*hidden_size_per_head]
    """
    B, T, H,D = query.size()
    query = query.reshape(B * T, self.ori_head, self.ori_feature)
    key = key.reshape(B * T, self.ori_head, self.ori_feature)
    value = value.reshape(B * T, self.ori_head, self.ori_feature)

    #########Simulate Attention Score
    query = self.sas_q(query).reshape(B, T, self.target_head, self.target_feature)
    key = self.sas_k(key).reshape(B, T, self.target_head, self.target_feature)
    value = self.sas_v(value).reshape(B, T, self.target_head, -1)

    attention= F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), is_causal=True)

    ##########Parameter-Efficient Attention Aggregation
    attention = attention.transpose(1, 2).contiguous().view(B, T, self.target_head//self.ori_head, self.ori_head*self.ori_feature)
    attention = self.output_dense(attention)
    attention = attention.mean(dim=-2)
    

    return attention

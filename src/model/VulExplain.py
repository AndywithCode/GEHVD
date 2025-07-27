from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from model.modules.gnns import GraphAttentionEncoder, GraphConvEncoder, GatedGraphConvEncoder

checkpoint = "D:\pytorchProject\DeepWukong\codet5-small"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                              torch_dtype=torch.float16,
                                              trust_remote_code=True).to(device)

encoding = tokenizer("def print_hello_world():", return_tensors="pt").to(device)
encoding['decoder_input_ids'] = encoding['input_ids'].clone()
outputs = model.generate(**encoding, max_length=15)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATEncoder, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=4, concat=True)

    def forward(self, node_features, edge_index):
        return self.gat(node_features, edge_index)

class CodeT5PlusWithGAT(nn.Module):
    def __init__(self, model_name, gnn_out_channels):
        super(CodeT5PlusWithGAT, self).__init__()
        self.codet5 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.gat_encoder = GATEncoder(self.codet5.config.hidden_size, gnn_out_channels)

        # 定义GNN输出的FFN
        self.gnn_ffn = nn.Sequential(
            nn.Linear(gnn_out_channels, 256),
            nn.ReLU(),
            nn.Linear(256, self.code_t5.config.hidden_size)  # 输出大小匹配CodeT5的隐藏层大小
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(self.code_t5.config.hidden_size)

        # 定义FFN层，将拼接的embedding作为输入
        self.ffn = nn.Sequential(
            nn.Linear(self.code_t5.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.code_t5.config.hidden_size)  # 输出大小匹配CodeT5的隐藏层大小
        )
        # 分类层，用于二分类任务
        self.classifier = nn.Linear(self.code_t5.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, edge_index, node_features):
        # CodeT5编码
        encoder_outputs = self.codet5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # GAT编码
        gnn_output = self.gat_encoder(node_features, edge_index)

        # GNN输出经过FFN
        gnn_output_ffn = self.gnn_ffn(gnn_output)

        # CodeT5的输出和GAT的输出做加法与归一化
        combined_output = self.layer_norm(encoder_hidden_states + gnn_output_ffn)

        # 将拼接的embedding传递给FFN
        ffn_output = self.ffn(combined_output)

        # 对FFN的输出进行二分类
        logits = self.classifier(ffn_output)
        probabilities = torch.sigmoid(logits)

        return probabilities

    def encode(self, input_ids, attention_mask, edge_index, node_features):
        # CodeT5编码
        encoder_outputs = self.codet5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # GAT编码
        gnn_output = self.gat_encoder(node_features, edge_index)

        # GNN输出经过FFN
        gnn_output_ffn = self.gnn_ffn(gnn_output)

        # CodeT5的输出和GAT的输出做加法与归一化
        combined_output = self.layer_norm(encoder_hidden_states + gnn_output_ffn)

        return combined_output

    def decode(self, combined_output, decoder_input_ids):
        # 传递到Decoder
        decoder_outputs = self.code_t5.decoder(decoder_input_ids, encoder_hidden_states=combined_output)

        return decoder_outputs

    def rag(self, combined_output, code_embeddings):
        # 计算查询向量和文档向量的余弦相似度
        cosine_similarities = torch.nn.functional.cosine_similarity(combined_output, code_embeddings)

        # 获取最相似的文档索引
        most_similar_index = torch.argmax(cosine_similarities).item()
        return

if __name__ == '__main__':
    custom_model = CodeT5PlusWithGAT(checkpoint, gnn_out_channels=64)
    # 输入示例
    input_ids = tokenizer("your code snippet here", return_tensors='pt').input_ids
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # 示例边
    node_features = torch.randn(input_ids.shape[0], input_ids.shape[1], model.config.hidden_size)  # 随机初始化节点特征
    decoder_input_ids = tokenizer("start of generation", return_tensors='pt').input_ids  # Decoder的输入

    # 前向传播，获取Decoder输出
    decoder_outputs = custom_model(input_ids, attention_mask, edge_index, node_features, decoder_input_ids)

    # 获取生成的token
    logits = decoder_outputs.logits  # 获取logits，用于生成任务
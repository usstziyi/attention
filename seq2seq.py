import torch
import math
import collections
from torch import nn
import matplotlib.pyplot as plt



DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

DATA_HUB['fra-eng'] = (DATA_URL + 'fra-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5')

import numpy as np
import torch


from torch import nn
from torch.nn import functional as F
from torch.utils import data




#################   WARNING   ################
# The below part is generated automatically through:
#    d2lbook build lib
# Don't edit it directly

import collections
import hashlib
import math
import os

import tarfile
import time
import zipfile

import requests

from matplotlib import pyplot as plt



import numpy as np
import torch


from torch import nn
from torch.utils import data
import time


# 编码器接口
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, inputs, *args):
        raise NotImplementedError

# 编码器接口
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, state, *args):
        raise NotImplementedError

    def forward(self, inputs, state):
        raise NotImplementedError

# 编码器-解码器
class EncoderDecoder(nn.Module):
    # 定义网络
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    # 定义前向传播
    def forward(self, src_inputs, tgt_inputs, *args):
        # 编码
        states, enc_state = self.encoder(src_inputs, *args)
        # 解码
        dec_state = self.decoder.init_state(enc_state, *args)
        return self.decoder(tgt_inputs, dec_state)

# Seq2Seq编码器：输入源句子
class Seq2SeqEncoder(Encoder):
    # 定义网络
    def __init__(self, src_vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层(D,E)
        self.embedding = nn.Embedding(
            num_embeddings=src_vocab_size, 
            embedding_dim=embed_size
        )
        # 循环神经网络层(E,H,L)
        self.rnn = nn.GRU(
            input_size=embed_size, 
            hidden_size=num_hiddens, 
            num_layers=num_layers, 
            dropout=dropout
        )
        # 没有输出层
    
    # 定义前向传播
    # inputs(B,T)
    # states(T,B,H)
    # state(L,B,H)
    def forward(self, inputs, *args):
        # inputs(B,T)->(B,T,E)
        inputs = self.embedding(inputs)
        # inputs(T,B,E)
        inputs = inputs.permute(1, 0, 2)
        # states(T,B,H),state(L,B,H)
        states, state = self.rnn(inputs)
        # states:包含每个时间步最后一层的信息
        # state:包含最后一个时间步每一层的信息
        return states, state
    
# Seq2Seq解码器：输入目标句子的词元，预测目标句子的下一个词元
class Seq2SeqDecoder(Decoder):
    # 定义网络
    def __init__(self, tgt_vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        # 嵌入层(D,E)
        self.embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size, 
            embedding_dim=embed_size
        )
        # 循环神经网络层(E+H,H,L)
        self.rnn = nn.GRU(
            input_size=embed_size + num_hiddens, 
            hidden_size=num_hiddens, 
            num_layers=num_layers, 
            dropout=dropout
        )
        # 输出层(H,D)
        self.dense = nn.Linear(
            in_features=num_hiddens, 
            out_features=tgt_vocab_size
        )


    # 初始化解码器状态
    def init_state(self, enc_state, *args):
        return enc_state # (L,B,H)

    # 定义前向传播
    # inputs(B,T)
    # output(B,T,D)
    # state(L,B,H)
    def forward(self, inputs, state):
        # inputs(B,T) -> (B,T,E)
        inputs = self.embedding(inputs)
        # inputs(T,B,E)
        inputs = inputs.permute(1, 0, 2)
        # inputs(T,B,E)
        # state(L,B,H),state[-1](B,H)
        # context(T,B,H)
        context = state[-1].repeat(inputs.shape[0], 1, 1)
        # inputs_and_context(T,B,E+H)
        inputs_and_context = torch.cat((inputs, context), 2)
        # states(T,B,H)
        # state(L,B,H)
        states, state = self.rnn(inputs_and_context, state)
        # 输出层，将隐藏状态映射到输出空间
        # output(T,B,H) -> (T,B,D)
        output = self.dense(states)
        # output(B,T,D)
        output = output.permute(1, 0, 2)
        return output, state
        # 在训练阶段，output 接下来送入 loss 损失函数更新梯度
        # 在预测阶段，output 接下来送入 argmax 函数获取预测结果


# 定义训练函数
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    # 初始化模型参数
    def xavier_init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # 初始化线性层的权重为均匀分布
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)  # 初始化循环神经网络层的权重为均匀分布
    # 应用初始化函数
    net.apply(xavier_init_weights)
    net.to(device)
    net.train()
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss(ignore_index=tgt_vocab['<pad>']) # 交叉熵损失函数，用于多分类问题

    # 初始化绘图
    fig, ax, line, x_list, y_list = init_plot(lr)

    # 训练
    for epoch in range(num_epochs):
        timer = Timer()
        # 定义累加器，用于记录每个epoch的损失总和和词元总数
        metric = Accumulator(2)
        # 按照批次训练，好处是梯度下降更稳定
        for batch in data_iter:
            # 拿取批次数据
            # src_inputs(B,T)
            # tgt_inputs(B,T)
            # tgt_valid_len(B)
            src_inputs, _, tgt_inputs, tgt_valid_len = batch
            # 数据移动到设备
            src_inputs = src_inputs.to(device)
            tgt_inputs = tgt_inputs.to(device)
            tgt_valid_len = tgt_valid_len.to(device)
   
            # bos(B,1)
            bos = torch.tensor([tgt_vocab['<bos>']] * tgt_inputs.shape[0], device=device).reshape(-1, 1) # 目标输入在训练阶段需要添加<bos>标记
            # tgt_teach(B,1+T-1)->(B,T)
            tgt_teach = torch.cat([bos, tgt_inputs[:, :-1]], 1) # 目标输入在训练阶段需要去掉最后一个词元            

            # start_time = time.time()
          

            # 1.清零梯度
            optimizer.zero_grad()
            # 2.训练，执行强制教学
            outputs, _ = net(src_inputs, tgt_teach) # outputs(B,T,D)
            # 3.计算损失
            l = loss(outputs, tgt_inputs) #压缩 T,l(B)
            l = l.sum() # 压缩 B,l(1)，每个批次的损失求和
            # 4.反向传播(计算梯度最耗时)
            l.backward()
            # 5.梯度裁剪
            grad_clipping(net, 1)
            # 6.更新参数
            optimizer.step()

            # end_time = time.time()
            # print(f"数据移动到设备耗时: {end_time - start_time:.6f} 秒")


            # 累加批次的损失和词元数量
            with torch.no_grad():
                metric.add(l, tgt_valid_len.sum())
        train_loss = metric[0] / metric[1]
        train_speed = metric[1] / timer.stop()
        print(f'epoch {(epoch + 1):3d}/{num_epochs}, loss {train_loss:.3f}, {train_speed:.1f} 词元/秒 {str(device)}')



        # 更新绘图
        update_plot(epoch+1, train_loss, x_list, y_list, line, ax)


    # 关闭绘图
    close_plot()


# 初始化绘图
def init_plot(lr):
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots(figsize=(10, 6))
    x_list = [] # x轴数据
    y_list = [] # y轴数据
    line, = ax.plot(x_list, y_list, 'b-', linewidth=2, label='Perplexity')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss')
    ax.set_title(f'RNN Training loss vs Epoch (lr={lr})')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig, ax, line, x_list, y_list
    # fig是画布
    # ax是坐标系区域(axis是带刻度的图表框)，一个fig 上可以有多个 ax
    # line是线对象
    # x_list和y_list是数据列表


# 更新绘图
def update_plot(x_item, y_item, x_list, y_list, line, ax):
    x_list.append(x_item)
    y_list.append(y_item)
    line.set_xdata(x_list)
    line.set_ydata(y_list)
    ax.set_xlim(0, x_item + 2)  # 确保x轴范围包含当前epoch，右边预留2个单位
    ax.set_ylim(0, max(y_list) * 1.1 if y_list else 1)  # 防止空列表报错，y轴预留10%
    plt.draw()
    plt.pause(0.01)

# 关闭绘图
def close_plot():
    plt.ioff()  # 关闭交互模式->恢复默认行为
    plt.show()  # 阻塞，保持窗口打开直到用户手动关闭


# 带遮蔽的softmax交叉熵损失函数
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=0, **kwargs):  # 默认设为 0，适配你的 padding
        super(MaskedSoftmaxCELoss, self).__init__(
            ignore_index=ignore_index, # 忽略目标中值等于 ignore_index 的位置
            reduction='none',  # 保持逐元素计算
            **kwargs
        )

    def forward(self, input, target):
        # 计算逐位置损失，无效位置（ignore_index）损失为 0
        # input 需要 (B, D, T)，target 是 (B, T)
        # input(B,T,D)->(B,D,T)
        unweighted_loss = super().forward(input.permute(0, 2, 1), target)  # (B, T)

        # 创建有效位置掩码：target 中不等于 ignore_index 的位置为 1
        # mask: (B, T)
        mask = (target != self.ignore_index).float()

        # 对每个样本，计算有效损失总和
        # loss_sum: (B,)
        loss_sum = (unweighted_loss * mask).sum(dim=1)

        # 获取每个样本的有效 token 数量（避免除零）
        # valid_token_count: (B,)
        valid_token_count = mask.sum(dim=1).clamp(min=1)  # 至少为1，防止除零

        # 计算每个句子的平均损失
        # weighted_loss: (B,)
        weighted_loss = loss_sum / valid_token_count

        return weighted_loss

# 定义预测函数，端到端预测
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    # 源句子小写化并分词，source(T,)
    source = src_sentence.lower().split(' ')
    # 源句子词元化,src_tokens(T,)
    src_tokens = src_vocab[source] + [src_vocab['<eos>']]
    # 源句子截断、填充,src_tokens(T,)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # src_inputs(1,T)
    src_inputs = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    pred_seq, attention_weight_seq = [], []
    
    # 编码
    net.eval()
    # states(T,B,H)=(T,1,H)
    # enc_state(L,B,H)=(1,1,H)
    states, enc_state = net.encoder(src_inputs)

    # 解码
    # dec_state(L,B,H)=(L,1,H)
    dec_state = net.decoder.init_state(enc_state)
    # 第一个输入是<bos>，next_input(1,1)
    next_input = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)

    for _ in range(num_steps):
        # tgt_hat(1,1,D)
        # dec_state(L,B,H)=(L,1,H)
        tgt_hat, dec_state = net.decoder(next_input, dec_state)
        # next_input(1,1)
        next_input = tgt_hat.argmax(dim=2)
        # pred(1,)
        pred = next_input.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列<eos>词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        pred_seq.append(pred) # 词元索引列表
        outputs = ' '.join(tgt_vocab.to_tokens(pred_seq)) # 词元列表转换为字符串
    return outputs, attention_weight_seq

# 预测序列的评估-布鲁算法
def bleu(pred_src, label_src, k):
    # 预测序列和标签序列都小写化并分词
    pred_tokens, label_tokens = pred_src.lower().split(' '), label_src.lower().split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    # 计算惩罚分数
    score = math.exp(min(0, 1 - len_label / len_pred))
    # 计算n元语法的匹配率
    for n in range(1, k + 1):
        # 统计标签序列中n元语法的出现次数
        num_n_grams = len_label - n + 1 # 分母：标签序列中 n元语法的总数量
        label_subs = collections.defaultdict(int) # label_subs字典
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1 # label子序列出现的次数,因为 n元语法子序列有重复的，所以这里要归类累加
        
        # 统计pred子序列在label子序列中出现
        num_matches = 0 # 分子：pred序列与标签序列中匹配的 n元语法的数量
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0: # pred子序列在label子序列中出现
                num_matches += 1 # 匹配数加1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1 # 匹配的label子序列次数减1
        # 计算n元语法的匹配率
        p_n = num_matches / num_n_grams
        # 计算n元语法的BLEU分数
        score *= math.pow(p_n, math.pow(0.5, n))
    return score

def main():
    # 1.超参数
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, try_gpu()
    print(device)

    # 2.加载数据集
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps, device)
    
    # 3.初始化模型
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)

    # 4.训练模型
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # 5.预测
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    print('----------------------------------------------------------------')
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')







def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def count_corpus(tokens):
    """Count token frequencies.

    Defined in :numref:`sec_text_preprocessing`"""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs

def load_data_nmt(batch_size, num_steps, device, num_examples=600):
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)

    # # 👇 在这里将所有张量转移到目标设备
    # src_array = src_array.to(device)
    # src_valid_len = src_valid_len.to(device)
    # tgt_array = tgt_array.to(device)
    # tgt_valid_len = tgt_valid_len.to(device)

    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def preprocess_nmt(text):
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def build_array_nmt(lines, vocab, num_steps):
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = reduce_sum(
        astype(array != vocab['<pad>'], torch.int32), 1)
    return array, valid_len

def read_data_nmt():
    data_dir = download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

def download_extract(name, folder=None):
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download(name, cache_dir=os.path.join('..', 'data')):
    """Download a file inserted into DATA_HUB, return the local filename.

    Defined in :numref:`sec_kaggle_house`"""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

class Accumulator:
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

ones = torch.ones
zeros = torch.zeros
tensor = torch.tensor
arange = torch.arange
meshgrid = torch.meshgrid
sin = torch.sin
sinh = torch.sinh
cos = torch.cos
cosh = torch.cosh
tanh = torch.tanh
linspace = torch.linspace
exp = torch.exp
log = torch.log
normal = torch.normal
rand = torch.rand
matmul = torch.matmul
int32 = torch.int32
float32 = torch.float32
concat = torch.cat
stack = torch.stack
abs = torch.abs
eye = torch.eye
numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)


if __name__ == '__main__':
    main()
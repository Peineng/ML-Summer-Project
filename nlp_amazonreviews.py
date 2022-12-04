#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# In[2]:


dataset = load_dataset("amazon_reviews_multi", "zh")


# In[29]:


print(dataset['train'].features)


# In[3]:


tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")


# In[46]:


print(tokenizer.encode('我在车的的的上面的的的上，基于合理，确认'))


# In[45]:


print(tokenizer.decode([101, 1762, 6756, 4794, 677, 8024, 1825, 754, 1394, 4415, 8024, 4802, 6371, 102]))


# In[4]:


model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")


# In[47]:


raw_data = dataset['train']['review_body'][0:100]


# In[39]:


raw_data = dataset['train'][0:100]
print(raw_data)


# In[9]:


print(raw_data[0])
print(raw_data[1])


# In[93]:


input = tokenizer(raw_data, padding='max_length', truncation=True, max_length=42)#, return_tensors='pt')
# return_tensors="pt") 
# padding='max_length', truncation=True, max_length=42)


# In[94]:


print(input)


# In[18]:


output = model(**input)


# In[48]:


model


# In[155]:


print(output)


# In[158]:


results = torch.softmax(output.logits, dim=1)#.tolist()


# In[159]:


print(results)


# In[58]:


input = ['本人账号被盗，资金被江西（杨建）挪用，请亚马逊尽快查实，将本人的200元资金退回。本人已于2017年11月30日提交退货申请，',
         '为何到2018年了还是没解决？亚马逊是什么情况？请给本人一个合理解释。这简直就是太差了！出版社怎么就能出版吗',
         '？我以为是百度摘录呢！这到底是哪个鱼目混珠的教授啊？！能给点干货吗？！总算应验了一句话，',
         '一本书哪怕只有一句花你感到有意义也算是本好书。哇为了找这本书哪怕一句不是废话的句子都费了我整整一天时间.',
        '书很好， 服务差',
        '我很不高兴， 太差了',
        '强烈推荐，质量很好']
inputto = tokenizer(input, padding='max_length', truncation=True, max_length=42, return_tensors='pt')
output = model(**inputto)


# In[59]:


import numpy as np


# In[60]:


results = torch.softmax(output.logits, dim=1).tolist()
print(np.array(results)[:,0])


# In[ ]:





# # Build model on bert base

# In[28]:


dataset 


# In[96]:


train_datase = dataset['train'][0:500]
train_dataset_raw = train_datase['review_body']
train_tokenized = tokenizer(train_dataset_raw, padding='max_length', truncation=True, max_length=42, return_tensors='pt')


# In[99]:


print(train_tokenized)


# In[104]:


print(train_tokenized['input_ids'], train_tokenized['input_ids'].size())


# In[122]:


a = np.array(train_datase['stars'])
b = a.reshape(-1,1)
c = torch.FloatTensor(b)
print(c, c.size())


# In[128]:


model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")


# In[129]:


# 将所有模型参数转换为一个列表
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


# In[177]:





# 定义计算准确率的函数：

# In[175]:


import numpy as np

# 根据预测结果和标签数据来计算准确率
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # 四舍五入到最近的秒
    elapsed_rounded = int(round((elapsed)))
    
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# In[223]:


from torch.utils.data import TensorDataset, random_split

train_dataset = dataset['train']
a = train_dataset['stars']
b = [[item/5, 1-item/5] for item in a]
train_label = torch.FloatTensor(b)


train_dataset_raw = train_dataset['review_body']
train_tokenized = tokenizer(train_dataset_raw, padding='max_length', truncation=True, max_length=42, return_tensors='pt')
train_input = TensorDataset(train_tokenized['input_ids'],train_tokenized['attention_mask'],                            train_label)

#train_input = train_input[torch.randperm(n_data)]
# 计算训练集和验证集大小
train_size = int(0.9 * len(train_input))
val_size = len(train_input) - train_size

# 按照数据大小随机拆分训练集和测试集
train_dataset, val_dataset = random_split(train_input, [train_size, val_size])

'''
val_dataset = dataset['validation'][0:500]
a = np.array(val_dataset['stars'])
b = [[item/5, 1-item/5] for item in a]
val_label = torch.FloatTensor(b)

val_dataset_raw =val_dataset['review_body']
val_tokenized = tokenizer(val_dataset_raw, padding='max_length', truncation=True, max_length=42, return_tensors='pt')
val_input = TensorDataset(val_tokenized['input_ids'], val_tokenized['attention_mask'],\
                          val_label)
'''


# In[222]:


print(train_dataset[100])


# In[ ]:





# In[179]:


import random
import numpy as np
import torch.optim as optim
# 我认为 'W' 代表 '权重衰减修复"
optimizer = optim.AdamW(model.parameters(), lr = 5e-5)

# 以下训练代码是基于 `run_glue.py` 脚本:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# 设定随机种子值，以确保输出是确定的
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
#torch.cuda.manual_seed_all(seed_val)

# 存储训练和评估的 loss、准确率、训练时长等统计指标, 
training_stats = []

# 统计整个训练时长
total_t0 = time.time()

for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # 统计单次 epoch 的训练时间
    t0 = time.time()

    # 重置每次 epoch 的训练总 loss
    total_train_loss = 0

    # 将模型设置为训练模式。这里并不是调用训练接口的意思
    # dropout、batchnorm 层在训练和测试模式下的表现是不同的 (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # 训练集小批量迭代
    #for step, batch in enumerate(train_dataloader):
    #    print(step, batch)
    batch_size = 20
    nbatch_train = 5 # = int(500/10)
    for ibatch in range(nbatch_train):
        print(ibatch)
        batch = train_input[int(ibatch*batch_size) : int((ibatch+1)*batch_size)]
        # 准备输入数据，并将其拷贝到 gpu 中
        b_input_ids = batch[0] #.to(device)
        b_input_mask = batch[1] #.to(device)
        b_labels = batch[2] #.to(device)
        # 每次计算梯度前，都需要将梯度清 0，因为 pytorch 的梯度是累加的
        optimizer.zero_grad()  # 梯度清0 # model.zero_grad()        
        # 前向传播
        # 文档参见: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # 该函数会根据不同的参数，会返回不同的值。 本例中, 会返回 loss 和 logits -- 模型的预测结果
        output = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask) #, labels=b_labels)
        results = torch.softmax(output.logits, dim=1)
        loss = criterion(results, b_labels) 
        # 累加 loss
        total_train_loss += loss.item()

        # 反向传播
        loss.backward()

        # 梯度裁剪，避免出现梯度爆炸情况
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新参数
        optimizer.step()

    # 平均训练误差
    avg_train_loss = total_train_loss / nbatch            
    
    # 单次 epoch 的训练时长
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # 完成一次 epoch 训练后，就对该模型的性能进行验证

    print("")
    print("Running Validation...")

    t0 = time.time()

    # 设置模型为评估模式
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    
    nbatch_val = 3 # = int(500/10)
    for ibatch in range(nbatch_val):
        print(ibatch)
        batch = val_input[int(ibatch*batch_size) : int((ibatch+1)*batch_size)]
        # 准备输入数据，并将其拷贝到 gpu 中
        b_input_ids = batch[0] #.to(device)
        b_input_mask = batch[1] #.to(device)
        b_labels = batch[2] #.to(device)
        
        # 评估的时候不需要更新参数、计算梯度
        output = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask) #, labels=b_labels)
        results = torch.softmax(output.logits, dim=1)
        loss = criterion(results, b_labels) 
        
        # 累加 loss
        total_eval_loss += loss.item()

        # 将预测结果和 labels 加载到 cpu 中计算
        logits = results.detach().numpy()
        label_ids = b_labels.numpy()
        
        # 计算准确率
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # 打印本次 epoch 的准确率
    avg_val_accuracy = total_eval_accuracy / nbatch_val
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # 统计本次 epoch 的 loss
    avg_val_loss = total_eval_loss / nbatch_val
    
    # 统计本次评估的时长
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # 记录本次 epoch 的所有统计信息
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[65]:


class BertClassificationModel(nn.Module):
    def __init__(self,hidden_size=768):
        super(BertClassificationModel, self).__init__()
        # 这里用了一个简化版本的bert
        model_name = 'uer/roberta-base-finetuned-dianping-chinese'

        # 读取分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 读取预训练模型
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name)

        for p in self.bert.parameters(): # 冻结bert参数
                p.requires_grad = False
        self.fc = nn.Linear(hidden_size,2)

    def forward(self, batch_sentences):   # [batch_size,1]
        sentences_tokenizer = self.tokenizer(batch_sentences,
                                             truncation=True,
                                             padding=True,
                                             max_length=512,
                                             add_special_tokens=True)
        input_ids=torch.tensor(sentences_tokenizer['input_ids']) # 变量
        attention_mask=torch.tensor(sentences_tokenizer['attention_mask']) # 变量
        bert_out=self.bert(input_ids=input_ids,attention_mask=attention_mask) # 模型

        last_hidden_state =bert_out[0] # [batch_size, sequence_length, hidden_size] # 变量
        bert_cls_hidden_state=last_hidden_state[:,0,:] # 变量
        fc_out=self.fc(bert_cls_hidden_state) # 模型
        return fc_out


# In[62]:


testNumber = 5000    # 多少个数据参与训练模型
validNumber = 5000   # 多少个数据参与验证
batchsize = 250  # 定义每次放多少个数据参加训练

trainDatas = dataset['train'] #ImdbDataset(mode="test",testNumber=testNumber) # 加载训练集,全量加载，考虑到我的破机器，先加载个100试试吧
validDatas =  dataset['validation'] #ImdbDataset(mode="valid",validNumber=validNumber) # 加载训练集

train_loader = torch.utils.data.DataLoader(trainDatas, batch_size=batchsize, shuffle=False)
#遍历train_dataloader 每次返回batch_size条数据
val_loader = torch.utils.data.DataLoader(validDatas, batch_size=batchsize, shuffle=False)
epoch_num = 1 


# In[68]:



model=BertClassificationModel()
optimizer = optim.AdamW(model.parameters(), lr=5e-5) # ???
# 这里是定义损失函数，交叉熵损失函数比较常用解决分类问题
# 依据你解决什么问题，选择什么样的损失函数
criterion = nn.CrossEntropyLoss()
print("模型数据已经加载完成,现在开始模型训练。")
for epoch in range(epoch_num):
    for i, (data,labels) in enumerate(train_loader, 0):
        output = model(data[0])
        optimizer.zero_grad()  # 梯度清0
        loss = criterion(output, labels[0])  # 计算误差
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 打印一下每一次数据扔进去学习的进展
        print('batch:%d loss:%.5f' % (i, loss.item()))

    # 打印一下每个epoch的深度学习的进展i
    print('epoch:%d loss:%.5f' % (epoch, loss.item()))
 #下面开始测试模型是不是好用哈
print('testing...(约2000秒(CPU))')

# 这里载入验证模型，他把数据放进去拿输出和输入比较，然后除以总数计算准确率
# 鉴于这个模型非常简单，就只用了准确率这一个参数，没有考虑混淆矩阵这些
num = 0
model.eval()  # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化,主要是在测试场景下使用；
for j, (data,labels) in enumerate(val_loader, 0):
    output = model(data[0])
    # print(output)
    out = output.argmax(dim=1)
    # print(out)
    # print(labels[0])
    num += (out == labels[0]).sum().item()
    # total += len(labels)
print('Accuracy:', num / validNumber)


# In[ ]:





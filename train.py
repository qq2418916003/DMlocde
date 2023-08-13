from datetime import datetime

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import TextRNN, GCNN, FastText, Confus
from  mydataset import MyDataSet
from config import Config

from torch.utils.tensorboard import SummaryWriter

cfg = Config()

train_dataset = MyDataSet(voc_dict_path='data/dict', data_path='data/train.txt', stop_word_path='data/hit_stopwords.txt')
cfg.pad_size = train_dataset.max_sql_len
train_loader=DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

val_dataset = MyDataSet(voc_dict_path='data/dict', data_path='data/test.txt', stop_word_path='data/hit_stopwords.txt')
val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

# model=TextRNN(cfg)
# model=GCNN(cfg)
# model=FastText(cfg)
model=Confus(cfg)
model.to(cfg.device)

loss_fn=CrossEntropyLoss()
optimier=Adam(params=model.parameters(), lr=cfg.lr)

tbWriter=SummaryWriter('runs/graph')
best_acc=0.0

for epoch in range(cfg.epochs):
    running_loss = 0.0
    steps = 0
    # 下面2个变量用于计算每轮的平均损失
    epoch_loss = 0.0
    epoch_steps = 0
    for index, batch in enumerate(train_loader):
        model.train()
        label, data = batch
        data = data.to(cfg.device)
        label = label.to(cfg.device)

        optimier.zero_grad()
        pred=model.forward(data)
        loss=loss_fn(pred[0], label)

        loss.backward()
        optimier.step()
        # 累计损失
        running_loss = running_loss + loss.item()
        # 下面两行用于之后计算每轮的平均损失
        epoch_loss += loss.item()
        epoch_steps += 1
        if index % 200 == 0:
            print("{}---{}: {}".format(epoch, index, loss.item()))
    # 计算平均损失
    mean_loss = epoch_loss / epoch_steps
    model.eval()
    correct_num = 0
    total_num = 0
    with torch.no_grad():
        for batch in val_loader:
            label, data = batch
            label, data = label.to(cfg.device), data.to(cfg.device)
            logits = model.forward(data)
            _, predicted = torch.max(logits[0], dim=1)
            total_num += label.size(0)
            correct_num += (predicted == label).sum().item()
            val_acc = correct_num / total_num
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
          f" validation accuracy: {val_acc}")
    # 保存训练数据到可视化图中
    tbWriter.add_scalars("summary", {"loss": mean_loss, "accuracy": val_acc}, epoch)
    # 每训练一轮,计算出准确率,把所有轮中准确率最高的那轮的参数保存
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "checkpoints/{}.pth".format(model.__class__.__name__))

    # 训练完成后以下代码评估每个类别的准确率
    model.eval()
    class_correct = list(0. for i in range(cfg.num_classes))
    class_total = list(0. for i in range(cfg.num_classes))
    with torch.no_grad():
        for batch in val_loader:
            label, data = batch
            label, data = label.to(cfg.device), data.to(cfg.device)
            logits = model.forward(data)
            _, predicted = torch.max(logits[0], dim=1)
            c = (predicted == label).squeeze()
            # 对验证集每批次数据,按标签类别分类
            for i in range(label.shape[0]):
                # 把标签分类
                lab = label[i]
                # 累加各类别的正确数
                class_correct[lab] += c[i]
                # 累加各类别的标签数(样本数)
                class_total[lab] += 1

    for i in range(cfg.num_classes):
        print(f"acc of class{i} : {class_correct[i] / class_total[i]}")


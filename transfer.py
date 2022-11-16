import shutil
import warnings
import os
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearnex import patch_sklearn

patch_sklearn()
warnings.filterwarnings("ignore")


# 获取计算硬件
torch.cuda.empty_cache()
device = torch.device('cuda:0')

# 训练集图像预处理：缩放裁剪、图像增强（随机水平翻转）、转 Tensor、归一化
train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.229, 0.224, 0.225])
                                      ])

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(512),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.5, 0.5, 0.5],
                                         std=[0.229, 0.224, 0.225])
                                     ])

# 载入图像分类数据
dataset_dir = "./Task4_TransferLearningDataset"
# 将train中的各种风格的类别图像合并
old_folder = ['art_painting', 'cartoon', 'photo', 'sketch']
cls_name = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
source_dir = os.path.join(dataset_dir, "train")
target_dir = source_dir
# 对于当前文件夹"Task4_TransferLearningDataset/train"
for cls in tqdm(cls_name):
    cur_src_dirs = glob(source_dir + f"/*/{cls}*")
    # 如果原来的分类文件夹（cartoon这种）不存在了，说明已经做好了，不需要处理了
    if cur_src_dirs is None:
        break
    # 将dirs中的所有文件合并到target_path对应分类
    cur_target_dir = target_dir + f"/{cls}"
    if not os.path.exists(cur_target_dir):  # 目标文件夹不存在就新建
        os.makedirs(cur_target_dir)
    # 对于当前文件夹"Task4_TransferLearningDataset/train/cartoon"
    for idx, cur_src_dir in enumerate(cur_src_dirs):
        for root, dirs, files in os.walk(cur_src_dir):
            print("root=", root, "dir=", dirs, "files=", files)
            # 对于files中的每一个文件如pic_001.jpg，重命名并迁移
            for file in files:
                cur_src_file = os.path.join(root, file)
                print("cur_src_file=", cur_src_file)
                print("cur_target_dir=", cur_target_dir)
                new_cur_src_file = os.path.join(root, str(idx) + file)
                # 文件重命名
                os.rename(cur_src_file, new_cur_src_file)
                # 复制对应文件
                shutil.copy(new_cur_src_file, cur_target_dir)

# 删除原来分类
for folder in old_folder:
    folder_path = os.path.join(source_dir, folder)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

test_dir = os.path.join(dataset_dir, "test")

# 载入训练集
train_dataset = datasets.ImageFolder(source_dir, train_transform)
# 载入测试集
test_dataset = datasets.ImageFolder(test_dir, test_transform)

# print('训练集图像数量', len(train_dataset))
# print('----类别个数', len(train_dataset.classes))
# print('----各类别名称', train_dataset.classes)
# print('测试集图像数量', len(test_dataset))
# print('----类别个数', len(test_dataset.classes))
# print('----各类别名称', test_dataset.classes)

# 类别和索引号 映射字典
# 各类别名称
class_names = train_dataset.classes
n_class = len(class_names)
# 映射关系：类别 到 索引号
print("train_dataset.class_to_idx = ", train_dataset.class_to_idx)
# 映射关系：索引号 到 类别
idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}
print("idx_to_labels = ", idx_to_labels)

# 定义数据加载器DataLoader


BATCH_SIZE = 20

# 训练集的数据加载器
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4
                          )

# 测试集的数据加载器
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=4
                         )

## 只微调训练模型最后一层（全连接分类层）
# model = models.resnet18(pretrained=True)  # 载入预训练模型
model = models.resnet34(pretrained=True)
# model = models.resnet101(pretrained=True)
# model = models.densenet201(pretrained=True)
# model = models.alexnet(pretrained=True)
print(model._modules)

# 修改全连接层，使得全连接层的输出与当前数据集类别数对应
# 新建的层默认 requires_grad=True
model.fc = nn.Linear(model.fc.in_features, n_class)

print("model.fc = ", model.fc)

# 只微调训练最后一层全连接层的参数，其它层冻结
optimizer = optim.Adam(model.fc.parameters())
for name, value in model.named_parameters():
    if (name != 'fc.weight') and (name != 'fc.bias'):
        value.requires_grad = False

# 训练配置
model = model.to(device)

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 训练轮次 Epoch
EPOCHS = 10

# 学习率降低策略
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

def train_one_batch(images, labels):
    """
    运行一个 batch 的训练，返回当前 batch 的训练日志
    """

    # 获得一个 batch 的数据和标注
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)  # 输入模型，执行前向预测
    loss = criterion(outputs, labels)  # 计算当前 batch 中，每个样本的平均交叉熵损失函数值

    # 优化更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 获取当前 batch 的标签类别和预测类别
    _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    log_train = {}
    log_train['epoch'] = epoch
    log_train['batch'] = batch_idx
    # 计算分类评估指标
    log_train['train_loss'] = loss
    log_train['train_accuracy'] = accuracy_score(labels, preds)
    log_train['train_precision'] = precision_score(labels, preds, average='macro')
    log_train['train_recall'] = recall_score(labels, preds, average='macro')
    log_train['train_f1-score'] = f1_score(labels, preds, average='macro')

    return log_train


def evaluate_testset():
    """
    在整个测试集上评估，返回分类评估指标日志
    """

    loss_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for images, labels in test_loader:  # 生成一个 batch 的数据和标注
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # 输入模型，执行前向预测

            # 获取整个测试集的标签类别和预测类别
            _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels)  # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)

    log_test = {}
    log_test['epoch'] = epoch

    # 计算分类评估指标
    log_test['test_loss'] = np.mean(loss)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')

    return log_test


## 训练开始之前，记录日志
epoch = 0
batch_idx = 0
best_test_accuracy = 0

# 训练日志-训练集
df_train_log = pd.DataFrame()
log_train = {}
log_train['epoch'] = 0
log_train['batch'] = 0
images, labels = next(iter(train_loader))
log_train.update(train_one_batch(images, labels))

df_train_log = df_train_log.append(log_train, ignore_index=True)

print("df_train_log=/n", df_train_log)

# 训练日志-测试集
df_test_log = pd.DataFrame()
log_test = {}
log_test['epoch'] = 0
log_test.update(evaluate_testset())
df_test_log = df_test_log.append(log_test, ignore_index=True)

print("df_test_log=/n", df_test_log)

## 运行训练
for epoch in range(1, EPOCHS + 1):

    print(f'Epoch {epoch}/{EPOCHS}')

    ## 训练阶段
    model.train()
    for images, labels in tqdm(train_loader):  # 获得一个 batch 的数据和标注
        batch_idx += 1
        log_train = train_one_batch(images, labels)
        # print('epoch:{:.3f}, batch:{:.3f}, train_loss:{:.3f}, train_accuracy:{:.3f}, train_procision:{:.3f},train_recall:{:.3f}, train_f1-score:{:.3f}'.format(log_train['epoch'], log_train['batch'], log_train['train_loss'], log_train['train_accuracy'],
        #       log_train['train_precision'], log_train['train_recall'], log_train['train_f1-score'])
        #       )
        df_train_log = df_train_log.append(log_train, ignore_index=True)

    lr_scheduler.step()

    ## 测试阶段
    model.eval()
    log_test = evaluate_testset()
    # print(
    #     'epoch:{:.3f}, test_loss:{:.3f}, test_accuracy:{:.3f}, test_procision:{:.3f},test_recall:{:.3f}, test_f1-score:{:.3f}'.format(
    #         log_test['epoch'], log_test['test_loss'], log_test['test_accuracy'],
    #         log_test['test_precision'], log_test['test_recall'], log_test['test_f1-score'])
    #     )
    df_test_log = df_test_log.append(log_test, ignore_index=True)

    # 保存最新的最佳模型文件
    if log_test['test_accuracy'] > best_test_accuracy:
        # 删除旧的最佳模型文件(如有)
        old_best_checkpoint_path = './checkpoints/best-{:.3f}.pth'.format(best_test_accuracy)
        if os.path.exists(old_best_checkpoint_path):
            os.remove(old_best_checkpoint_path)
        # 保存新的最佳模型文件
        new_best_checkpoint_path = './checkpoints/best-{:.3f}.pth'.format(log_test['test_accuracy'])
        torch.save(model, new_best_checkpoint_path)
        print('保存新的最佳模型', './checkpoints/best-{:.3f}.pth'.format(best_test_accuracy))
        best_test_accuracy = log_test['test_accuracy']



df_train_log.to_csv('训练日志-训练集.csv', index=False)
df_test_log.to_csv('训练日志-测试集.csv', index=False)

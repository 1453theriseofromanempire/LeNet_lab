import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model import LeNet5, LeNet5_sigmoid, LeNet5_more_channel, LeNet5_small_kernel
from utils.optimizer import create_optimizer
from utils.data import create_dataset

lr = 1e-3
epoch_num = 10
device = 'cuda'
batch_size = 128

def test_one_epoch(model, test_loader, epoch, writer, model_name):
    model.eval()
    correct = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = correct / len(test_loader.dataset)
    avg_loss = total_loss / len(test_loader)
    
    # 使用 model_name 来确保不同模型写入不同路径
    writer.add_scalar(f"{model_name}/Test/Loss", avg_loss, epoch)
    writer.add_scalar(f"{model_name}/Test/Accuracy", accuracy, epoch)

def train_one_epoch(model, optimizer, train_loader, criterion, epoch, writer, model_name):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # 使用 model_name 来确保不同模型写入不同路径
        writer.add_scalar(f"{model_name}/Train/Loss", loss.item(), epoch * len(train_loader) + batch_idx)
    
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar(f"{model_name}/Train/Avg_Loss", avg_loss, epoch)

def main(model, model_name:str):
    model = model.to(device)
    optimizer = create_optimizer(model=model, lr=lr)
    train_data, test_data = create_dataset(batch_size)
    criterion = nn.CrossEntropyLoss()
    
    # 为每个模型创建一个独立的 TensorBoard 路径
    writer = SummaryWriter(log_dir=f'runs/{model_name}')

    for epoch in range(epoch_num):
        train_one_epoch(model, optimizer, train_data, criterion, epoch, writer, model_name)
        test_one_epoch(model, test_data, epoch, writer, model_name)
    
    writer.close()

if __name__ == "__main__":
    models = [LeNet5(), LeNet5_sigmoid(), LeNet5_more_channel(), LeNet5_small_kernel()]
    
    # 循环训练每个模型
    for model in models:
        if isinstance(model, LeNet5_more_channel):
            model_name = model.__class__.__name__
            main(model, model_name)

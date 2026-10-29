from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import ResNet18,Residual
import torch
import torch.nn as nn
import copy
import time
import pandas as pd


def train_val_data_process():
    ROOT_TRAIN=r'D:\ResNet18\archive\train'

    train_transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    train_data=ImageFolder(ROOT_TRAIN,transform=train_transform)


    #数据划分
    train_data,val_data=Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])
    train_dataloader=Data.DataLoader(dataset=train_data,
                                    batch_size=32,
                                    shuffle=True,
                                    num_workers=2)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                      batch_size=32,
                                      shuffle=True,
                                      num_workers=2)
    return train_dataloader,val_dataloader

def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device=torch.device("cuda"if torch.cuda.is_available() else "cpu")

    #优化器
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    #交叉熵损失
    criterion=nn.CrossEntropyLoss()

    model=model.to(device)
    best_model_wts=copy.deepcopy(model.state_dict())


    best_acc=0.0
    train_loss_all=[]
    val_loss_all=[]
    train_acc_all=[]
    val_acc_all=[]

    since=time.time()


    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        print("-"*10)

        train_loss=0.0
        train_corrects=0

        val_loss = 0.0
        val_corrects = 0

        train_num=0
        val_num=0

        for step,(b_x,b_y) in enumerate(train_dataloader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)

            model.train()

            output=model(b_x)

            pre_label=torch.argmax(output,dim=1)

            loss=criterion(output,b_y)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_loss+=loss.item()*b_x.size(0)

            train_corrects+=torch.sum(pre_label==b_y.data)

            train_num+=b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()

            output = model(b_x)

            pre_label = torch.argmax(output, dim=1)

            loss = criterion(output, b_y)

            val_loss += loss.item() * b_x.size(0)

            val_corrects += torch.sum(pre_label == b_y.data)

            val_num += b_x.size(0)

        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        val_acc_all.append(val_corrects.double().item()/val_num)
        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print("{} val loss:{:.4f} train acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))


        if val_acc_all[-1]>best_acc:
            best_acc=val_acc_all[-1]

            best_model_wts=copy.deepcopy(model.state_dict())

        time_use=time.time()-since
        print("训练耗费的时间：{:.0f}m{:.0f}s".format(time_use//60,time_use%60))


    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(),r'D:\ResNet18\best_model.pth')



    train_process=pd.DataFrame(data={"epoch":range(num_epochs),
                                     "train_loss_all":train_loss_all,
                                     "val_loss_all":val_loss_all,
                                     "train_acc_all": train_acc_all,
                                     "val_acc_all": val_acc_all,
                                     })

    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"],train_process.train_loss_all,'ro-',label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")

    plt.legend()
    plt.show()






if __name__=="__main__":
    model=ResNet18(Residual)

    train_dataloader,val_dataloader=train_val_data_process()
    train_process=train_model_process(model, train_dataloader, val_dataloader, 20)

    matplot_acc_loss(train_process)














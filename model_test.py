import torch
import  numpy as np
import torch.utils.data as Data
from torchvision import transforms
from model import ResNet18,Residual
from torchvision.datasets import ImageFolder

def test_val_data_process():
    ROOT_TRAIN = r'D:\ResNet18\archive\test'

    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_data = ImageFolder(ROOT_TRAIN, transform=test_transform)
    #数据划分
    test_dataloader=Data.DataLoader(dataset=test_data,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=2)
    return test_dataloader

def test_model_process(model,test_dataloader):
    device="cuda" if torch.cuda.is_available() else 'cpu'

    model=model.to(device)
    model.eval()

    classes=test_dataloader.dataset.classes

    test_corrects=0.0
    test_num=0

    with torch.no_grad():
        for test_data_x,test_data_y in test_dataloader:
            test_data_x=test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            output=model(test_data_x)

            pre_lab=torch.argmax(output,dim=1)

            test_corrects+=torch.sum(pre_lab==test_data_y.data)

            test_num+=test_data_x.size(0)
            result = pre_lab.item()
            label = test_data_y.item()

            print(
                "真实值：", classes[label],
                "---- 预测值：", classes[result]
            )



    test_acc=test_corrects.double().item()/test_num
    print("测试准确率：",test_acc)


if __name__=="__main__":
    model=ResNet18(Residual)
    model.load_state_dict(torch.load('best_model.pth'))

    test_dataloader=test_val_data_process()
    test_model_process(model,test_dataloader)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=model.to(device)



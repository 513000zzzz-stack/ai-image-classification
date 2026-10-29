import torch
import  numpy as np
import pandas as pd
import torch.utils.data as Data
from torchvision import transforms
from model import ResNet18,Residual
from torchvision.datasets import ImageFolder
from pathlib import Path

project_dir = Path(__file__).resolve().parent

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
    all_true=[]
    all_pred=[]

    with torch.no_grad():
        for test_data_x,test_data_y in test_dataloader:
            test_data_x=test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            output=model(test_data_x)

            pre_lab=torch.argmax(output,dim=1)
            all_true.extend(test_data_y.cpu().tolist())
            all_pred.extend(pre_lab.cpu().tolist())

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
    num_classes=len(classes)
    confusion=np.zeros((num_classes,num_classes),dtype=np.int64)

    for true_label,pre_label in zip(all_true,all_pred):
        confusion[true_label,pre_label]+=1

    print("类别顺序： ",classes)
    print("混淆矩阵： ")
    print(confusion)
    confusion_df = pd.DataFrame(
        confusion,
        index=classes,
        columns=classes
    )

    confusion_df.to_csv(
        project_dir / "confusion_matrix.csv",
        encoding="utf-8-sig"
    )
    fake_correct=confusion[0,0]
    predicted_fake=confusion[:,0].sum()

    if predicted_fake>0:
        fake_precision=fake_correct/predicted_fake
    else:
        fake_precision=0.0
    print(f"FAKE 精确率： {fake_precision:.2%}")
    actual_fake=confusion[0,:].sum()
    if actual_fake>0:
        fake_recall=fake_correct/actual_fake
    else:
        fake_recall=0.0
    print(f"FAKE 召回率：{fake_recall:.2%}")

    if fake_precision+fake_recall>0:
        fake_f1=(
            2*fake_precision*fake_recall/(fake_precision+fake_recall)
        )
    else:
        fake_f1=0.0
    print(f"FAKE F1：{fake_f1:.2%}")

    metric_rows = []

    for class_index, class_name in enumerate(classes):
        correct = confusion[class_index, class_index]
        predicted_total = confusion[:, class_index].sum()
        actual_total = confusion[class_index, :].sum()

        precision = correct / predicted_total if predicted_total > 0 else 0.0
        recall = correct / actual_total if actual_total > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        metric_rows.append({
            "class": class_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(actual_total)
        })

    metrics_df = pd.DataFrame(metric_rows)

    metrics_df.to_csv(
        project_dir / "classification_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(metrics_df.to_string(index=False))





if __name__=="__main__":
    model=ResNet18(Residual)
    model.load_state_dict(torch.load('best_model.pth'))

    test_dataloader=test_val_data_process()
    test_model_process(model,test_dataloader)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=model.to(device)



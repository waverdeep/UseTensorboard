# tutorial Pytorch official
from torch.utils.tensorboard import SummaryWriter
import dataloader
import torchvision
import model
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import functions
import numpy as np
# 텐서보드를 사용해서 Projector를 구현할 때 오류가 있음
# 이 오류를 해결하기 위해서 작성해야 할 것
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


def images_to_probs(net, images):
    '''
    학습된 신경망과 이미지 목록으로부터 예측 결과 및 확률을 생성합니다
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    학습된 신경망과 배치로부터 가져온 이미지 / 라벨을 사용하여 matplotlib
    Figure를 생성합니다. 이는 신경망의 예측 결과 / 확률과 함께 정답을 보여주며,
    예측 결과가 맞았는지 여부에 따라 색을 다르게 표시합니다. "images_to_probs"
    함수를 사용합니다.
    '''
    preds, probs = images_to_probs(net, images)
    # 배치에서 이미지를 가져와 예측 결과 / 정답과 함께 표시(plot)합니다
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        dataloader.matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            dataloader.classes[preds[idx]],
            probs[idx] * 100.0,
            dataloader.classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


# projector 관련 코드
# 헬퍼(helper) 함수
def select_n_random(data, labels, n=100):
    '''
    데이터셋에서 n개의 임의의 데이터포인트(datapoint)와 그에 해당하는 라벨을 선택합니다
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


# 명령줄에서 사용할때 tensorboard --logdir=runs를 사용한다.
# 그런데 remote 환경에서는 저렇게 하면 접근할 수 없기 때문에
# tensorboard --logdir=runs --bind_all 로 작성해야 한다.
def set_tensorboard_writer(name):
    writer = SummaryWriter(name) # 'runs/fashion_mnist_experiment_1'
    return writer


def close_tensorboard_writer(writer):
    writer.close()


def show_image_tensorboard(writer, trainloader):
    # tensorboard에 기록하기
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # 이미지 그리드를 만듭니다.
    img_grid = torchvision.utils.make_grid(images)

    # 이미지를 보여줍니다.
    dataloader.matplotlib_imshow(img_grid, one_channel=True)

    # tensorboard에 기록합니다.
    writer.add_image('four_fashion_mnist_images', img_grid)


def show_model_tensorboard(writer, net, trainloader):
    # tensorboard에 기록하기
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    writer.add_graph(net, images)


def show_projector_tensorboard(writer, trainset):
    # projector 관련 코드
    # 임의의 이미지들과 정답(target) 인덱스를 선택합니다
    images, labels = select_n_random(trainset.data, trainset.targets)

    # 각 이미지의 분류 라벨(class label)을 가져옵니다
    class_labels = [dataloader.classes[lab] for lab in labels]

    # 임베딩(embedding) 내역을 기록합니다
    features = images.view(-1, 28 * 28)
    writer.add_embedding(features,
                         metadata=class_labels,
                         label_img=images.unsqueeze(1))


def train(trainloader, net, optimizer, criterion, writer):
    for epoch in range(1):  # 데이터셋을 여러번 반복

        for i, data in enumerate(trainloader, 0):

            # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            inputs, labels = data

            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:  # 매 1000 미니배치마다...

                # ...학습 중 손실(running loss)을 기록하고
                writer.add_scalar('training loss',
                                  running_loss / 1000,
                                  epoch * len(trainloader) + i)

                # ...무작위 미니배치(mini-batch)에 대한 모델의 예측 결과를 보여주도록
                # Matplotlib Figure를 기록합니다
                writer.add_figure('predictions vs. actuals',
                                  plot_classes_preds(net, inputs, labels),
                                  global_step=epoch * len(trainloader) + i)
                running_loss = 0.0
    print('Finished Training')

def main():
    trainset, testset = dataloader.get_datasets()
    trainloader, testloader = dataloader.get_loader()
    net = model.Net()
    criterion = functions.get_criterion()
    optimizer = functions.get_optimizer(net, 0.001, 0.9)
    set_tensorboard(trainset, testset, trainloader, testloader, net)
    writer = set_tensorboard_writer('runs/fashion_mnist_experiment_1')
    show_image_tensorboard(writer, trainloader)
    show_model_tensorboard(writer, net, trainloader)
    show_projector_tensorboard(writer, trainset)
    close_tensorboard_writer(writer)




if __name__ == '__main__':
    main()
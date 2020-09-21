# tutorial Pytorch official
from torch.utils.tensorboard import SummaryWriter
import dataloader
import torchvision
import model
import torch
# 텐서보드를 사용해서 Projector를 구현할 때 오류가 있음
# 이 오류를 해결하기 위해서 작성해야 할 것
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


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
def set_tensorboard(trainset, testset, trainloader, testloader, net):
    # 기본 `log_dir` 은 "runs"이며, 여기서는 더 구체적으로 지정하였습니다
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')

    # tensorboard에 기록하기
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # 이미지 그리드를 만듭니다.
    img_grid = torchvision.utils.make_grid(images)

    # 이미지를 보여줍니다.
    dataloader.matplotlib_imshow(img_grid, one_channel=True)

    # tensorboard에 기록합니다.
    writer.add_image('four_fashion_mnist_images', img_grid)

    # 모델 살펴보기
    writer.add_graph(net, images)

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
    writer.close()


def main():
    trainset, testset = dataloader.get_datasets()
    trainloader, testloader = dataloader.get_loader()
    net = model.Net()
    set_tensorboard(trainset, testset, trainloader, testloader, net)


if __name__ == '__main__':
    main()
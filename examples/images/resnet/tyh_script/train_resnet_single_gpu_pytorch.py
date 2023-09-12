import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        start_time = time.time()  # 记录每个 epoch 所有 batch 的开始时间
        iterations_in_epoch = 0  # 记录每个 epoch 中所有 batch 的迭代次数和

        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            iterations_in_batch = len(self.train_data)  # 记录每个 batch 的迭代次数
            # print("当前batch大小：{}".format(iterations_in_batch))
            iterations_in_epoch += iterations_in_batch

            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

        time_in_epoch = time.time() - start_time  # 记录每个 epoch 所有 batch 的结束时间

        average_iterations_per_second_in_epoch = iterations_in_epoch / time_in_epoch  # 单 GPU 每个 epoch 的平均迭代速度
        
        print("当前epoch耗时：{}，迭代次数：{}，平均迭代速度：{}".format(time_in_epoch, iterations_in_epoch, average_iterations_per_second_in_epoch))

        return average_iterations_per_second_in_epoch

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):

        iterations_per_second_sum = 0  # 用于记录所有 epoch 的平均迭代次数和

        for epoch in range(max_epochs):
            average_iterations_per_second_in_epoch = self._run_epoch(epoch)
            iterations_per_second_sum += average_iterations_per_second_in_epoch
            # 保存模型
            # if epoch % self.save_every == 0:
            #     self._save_checkpoint(epoch)

        # 计算整个训练过程的平均每秒迭代次数
        average_iterations_per_second = iterations_per_second_sum / max_epochs
        print(f"训练完成. 整个训练过程每秒迭代次数: {average_iterations_per_second:.2f}")


def load_train_objs():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet的输入大小为224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet的均值和标准差
    ])
    train_set = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

    model = models.resnet18(pretrained=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )


def main(device, total_epochs, save_every, batch_size):
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, device, save_every)
    trainer.train(total_epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=1000, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    device = 0  # shorthand for cuda:0
    main(device, args.total_epochs, args.save_every, args.batch_size)


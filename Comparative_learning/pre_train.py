import torch
from torchvision import transforms
from dataloader import MyDataset
from torch.utils.data import DataLoader
from myqueue import Queue
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

def momentum_update_key_encoder(m, f_q, f_k):
    for params_q, params_k in zip(f_q.parameters(), f_k.parameters()):
        params_k.data = params_k.data * m + params_q.data * (1. - m)

bs = 128
epochs = 200
dims = 128
t = 0.07
m = 0.999

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
augmentation = [
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    normalize,
]

dataset = MyDataset("txt_data/total.txt", transform=transforms.Compose(augmentation))
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)

f_q = models.resnet50(num_classes=dims).cuda()
f_k = models.resnet50(num_classes=dims).cuda()
momentum_update_key_encoder(0, f_q, f_k)

optimizer = torch.optim.SGD(f_q.parameters(), lr=10, weight_decay=1e-3)
loss_func = nn.CrossEntropyLoss()

queue = Queue(bs, dims)

for epoch in range(epochs):
    for i, (img_q, img_k) in enumerate(dataloader):
        print(f"batch: {i}")
        if len(img_q) != bs:
            raise ValueError("Batch size error!")
        img_q = img_q.cuda()
        img_k = img_k.cuda()
        if queue.size() < queue.max_size:
            queue.enqueue(f_k.forward(img_k).detach())
            continue
        q = f_q.forward(img_q)
        k = f_k.forward(img_k)
        k = k.detach()

        l_pos = torch.squeeze(torch.bmm(q.view(bs, 1, dims), k.view(bs, dims, 1)), dim=1)
        l_neg = torch.mm(q.view(bs, dims), queue.view(dims, queue.k_num()))

        logits = torch.cat((l_pos, l_neg), dim=1)

        labels = torch.zeros(bs, dtype=torch.long).cuda()
        loss = loss_func(logits / t, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        momentum_update_key_encoder(m, f_q, f_k)

        queue.update(k)
        
        print(f"Epoch: {epoch}, batch: {i}, Loss: {loss.item()}")
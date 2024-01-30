import torch

class Queue:
    def __init__(self, bs, dims, max_size=128):
        self.queue = []
        self.max_size = max_size
        self.bs = bs
        self.dims = dims

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        return self.queue.pop(0)
    
    def size(self):
        return len(self.queue)
    
    def k_num(self):
        return self.size() * self.bs
    
    def view(self, *args):
        return torch.cat(self.queue).view(*args)
    
    def update(self, item):
        if self.size() < self.max_size:
            self.enqueue(item)
        else:
            self.dequeue()
            self.enqueue(item)
#!/usr/bin/env python
# coding: utf-8
from torch import cuda
import time
import matplotlib.pyplot as plt

class Pytorch_Analyzer(object):
    def __init__(self, _net):
        # set parameters
        self._count = 0
        self._num_layer = 0
        self._iter = 10
        self._timestamp = time.time()
        self._net = _net

        # hooks
        self._hooks = []
        # load hooks
        self.net(_net)
        # info forward lists
        self.layers = []
        self.layer_memory = []
        self.memory = []
        self.max_memory = []
        self.exec_time = []
        
    
    def net(self, _net):
        for module in _net.children():
            for layer in module.children():
                if(self._num_layer == 0):
                    self._hooks.append(layer.register_forward_pre_hook(self.initial))
                self._hooks.append(layer.register_forward_hook(self.layer))
                self._num_layer += 1
    
    def layer(self, _module, _input, _output):
        # use iter_th as our memory standard
        if(self._count >= self._num_layer * self._iter and 
            self._count < self._num_layer * (self._iter+1)
        ):
            self.exec_time.append((time.time() - self._timestamp) * 1000000)
            self.layers.append(_module.__str__()[:30])
            self.memory.append(abs(cuda.memory_allocated() / 1024. / 1024.))
            self.max_memory.append(abs(cuda.max_memory_allocated() / 1024. / 1024.))
            # remove forward_hook to accelerate net
            if(self._count == self._num_layer * (self._iter+1)):
                for hook in self._hooks:
                    hook.remove()
        # update timestamp
        self._count += 1

        # reset max memory allocated
        cuda.reset_max_memory_allocated()
        self._timestamp = time.time()

    def initial(self, _net, input):
        if(self._count == self._num_layer * self._iter):
            self.exec_time.append((time.time() - self._timestamp) * 1000000)
            self.memory.append(abs(cuda.memory_allocated() / 1024. / 1024.))
            self.max_memory.append(abs(cuda.max_memory_allocated() / 1024. / 1024.))
            self.layers.append('Input, label etc.')

        # reset max memory allocated
        cuda.reset_max_memory_allocated()
        self._timestamp = time.time()
    
    def analysis(self):
        print('{:<3}  {:>14}  {:>12}  {:>12}  {:>13}    {:<5s}'
        .format(
            'No.',
            'Layer_memory',
            'Max_memory',
            'Memory',
            'Exec_time',
            'Layer'
            )
        )
        if(self.memory):
            print('Initial---------------------------------------------------------------------------------------------------')
            print('{:<3}  {:>11.2f} MB  {:>9.2f} MB  {:>9.2f} MB  {:>10.2f} us    {:<35s}'
                .format(
                    0,
                    self.memory[0],
                    self.max_memory[0],
                    self.memory[0],
                    self.exec_time[0],
                    self.layers[0]
                    )
                )
            print('Forward---------------------------------------------------------------------------------------------------')
            for i in range(1, self._num_layer+1): 
                print('{:<3}  {:>11.2f} kB  {:>9.2f} MB  {:>9.2f} MB  {:>10.2f} us    {:<35s}'
                .format(
                    i,
                    (self.memory[i] - self.memory[i-1])*1024,
                    self.max_memory[i],
                    self.memory[i],
                    self.exec_time[i],
                    self.layers[i]
                    )
                )
        else:
            print('----------------------------------------------------------------------------------------No layers\n')

    def analysis_plot(self):
        # plot layer memory
        plt.figure(1)
        self.layer_memory = [(self.memory[i] - self.memory[i-1]) * 1024 for i in range(1, self._num_layer+1)]
        plt.plot(self.layer_memory, label='layer_memory')
        plt.title('Layer Memory Usage')
        plt.xlabel('nth layer')
        plt.ylabel('kB')
        plt.legend()

        # plot cumulative memory
        plt.figure(2)
        memory = self.memory[1:]
        max_memory =  self.max_memory[1::]
        plt.plot(memory, label='memory')
        plt.plot(max_memory, label='max_memory')
        plt.title('Cumulative Memory Usage')
        plt.xlabel('nth layer')
        plt.ylabel('MiB')
        plt.legend()

        # plot execution time
        plt.figure(3)
        exec_time = self.exec_time[1::]
        plt.plot(exec_time, label='exec_time')
        plt.title('Excution time')
        plt.xlabel('nth layer')
        plt.ylabel('us')
        plt.legend()

        plt.show()

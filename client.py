import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import warnings
import os.path

import network
import layers


# For fashion-MNIST and similar problems
DATA_ROOT = '/data/cs3450/data/'
FASHION_MNIST_TRAINING = '/data/cs3450/data/fashion_mnist_flattened_training.npz'
FASHION_MNIST_TESTING = '/data/cs3450/data/fashion_mnist_flattened_testing.npz'
CIFAR10_TRAINING = '/data/cs3450/data/cifar10_flattened_training.npz'
CIFAR10_TESTING = '/data/cs3450/data/cifar10_flattened_testing.npz'
CIFAR100_TRAINING = '/data/cs3450/data/cifar100_flattened_training.npz'
CIFAR100_TESTING = '/data/cs3450/data/cifar100_flattened_testing.npz'

def load_dataset_flattened(train=True,dataset='Fashion-MNIST',download=False):
    """
    :param train: True for training, False for testing
    :param dataset: 'Fashion-MNIST', 'CIFAR-10', or 'CIFAR-100'
    :param download: True to download. Keep to false afterwords to avoid unneeded downloads.
    :return: (x,y) the dataset. x is a numpy array where columns are training samples and
             y is a numpy array where columns are one-hot labels for the training sample.
    """
    if dataset == 'Fashion-MNIST':
        if train:
            path = FASHION_MNIST_TRAINING
        else:
            path = FASHION_MNIST_TESTING
        num_labels = 10
    elif dataset == 'CIFAR-10':
        if train:
            path = CIFAR10_TRAINING
        else:
            path = CIFAR10_TESTING
        num_labels = 10
    elif dataset == 'CIFAR-100':
        if train:
            path = CIFAR100_TRAINING
        else:
            path = CIFAR100_TESTING
        num_labels = 100
    else:
        raise ValueError('Unknown dataset: '+str(dataset))

    if os.path.isfile(path):
        print('Loading cached flattened data for',dataset,'training' if train else 'testing')
        data = np.load(path)
        x = torch.tensor(data['x'],dtype=torch.float32)
        y = torch.tensor(data['y'],dtype=torch.float32)
        pass
    else:
        class ToTorch(object):
            """Like ToTensor, only to a numpy array"""

            def __call__(self, pic):
                return torchvision.transforms.functional.to_tensor(pic)

        if dataset == 'Fashion-MNIST':
            data = torchvision.datasets.FashionMNIST(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        elif dataset == 'CIFAR-10':
            data = torchvision.datasets.CIFAR10(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        elif dataset == 'CIFAR-100':
            data = torchvision.datasets.CIFAR100(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        else:
            raise ValueError('This code should be unreachable because of a previous check.')
        x = torch.zeros((len(data[0][0].flatten()), len(data)),dtype=torch.float32)
        for index, image in enumerate(data):
            x[:, index] = data[index][0].flatten()
        labels = torch.tensor([sample[1] for sample in data])
        y = torch.zeros((num_labels, len(labels)), dtype=torch.float32)
        y[labels, torch.arange(len(labels))] = 1
        np.savez(path, x=x.detach().numpy(), y=y.detach().numpy())
    return x, y


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print("Running on GPU")
    
        # dataset = 'Fashion-MNIST'
        # dataset = 'CIFAR-10'
        dataset = 'CIFAR-100'

        # Set up model parameters based on dataset
        image_size = 0
        training_samples = 0
        classes = 0
        test_samples = 10000

        if dataset == 'Fashion-MNIST':
            image_size = 28 * 28
            training_samples = 60000
            classes = 10
        else:
            image_size = 32 * 32 * 3
            training_samples = 50000
            if dataset == 'CIFAR-10':
                classes = 10
            else:
                classes = 100


        x_train, y_train = load_dataset_flattened(train=True, dataset=dataset, download=True)

        x_train = x_train.T
        y_train = y_train.T
        print(x_train.shape)
        print(y_train.shape)

        # TODO: Build your network.
        nn = network.Network()
        x = layers.Input(image_size)
        x.set(x_train[0])
        nn.add(x)

        w = layers.Input((128, image_size), True)
        w.randomize()
        nn.add(w)

        b = layers.Input(128, True)
        b.randomize()
        nn.add(b)

        layer1 = layers.Linear(x, w, b)
        nn.add(layer1)

        relu = layers.ReLU(layer1)
        nn.add(relu)

        w2 = layers.Input((classes, 128), True)
        w2.randomize()
        nn.add(w2)

        b2 = layers.Input(classes, True)
        b2.randomize()
        nn.add(b2)

        layer2 = layers.Linear(relu, w2, b2)
        nn.add(layer2)

        y = layers.Input(classes)
        y.set(y_train[0])
        nn.add(y)

        loss = layers.Loss(layer2, y)
        nn.add(loss)

        s1 = layers.Regularization(w, 0.01)
        nn.add(s1)

        s2 = layers.Regularization(w2, 0.01)
        nn.add(s2)

        s = layers.Sum(s1, s2)
        nn.add(s)

        J = layers.Sum(loss, s)
        nn.add(J)

        cif_loss = []
        # TODO: Train your network.
        for e in range(20):
            print('EPOCH # ', e + 1)
            # Sends each batch into model for forward/back prop
            for batch in range(training_samples):
                x.set(x_train[batch])
                y.set(y_train[batch])
                nn.forward()
                nn.backward()
                nn.step(0.0001)

                # Prints the loss
                if batch % 10000 == 0:
                    print(loss.output, batch + 1)

                # Prints the loss
                if batch % 40000 == 0:
                    cif_loss.append(loss.output)
        
        # cif_loss = torch.load('test.pt')
        torch.save(torch.tensor(cif_loss), 'test.pt')

        # TODO: Test the accuracy of your network
        x_test, y_test = load_dataset_flattened(train=False, dataset=dataset, download=True)
        x_test = x_test.T
        y_test = y_test.T
        
        test_acc = 0
        for a in range(test_samples):
            x.set(x_test[a])
            y.set(y_test[a])
            nn.forward()
            # Accuracy
            if loss.softmax.argmax() == y.output.argmax():
                test_acc += 1.0
        
        test_acc = test_acc / test_samples
        torch.save(torch.tensor(test_acc), 'test2.pt')
        print("Testing Accuracy", test_acc)

    pass # You may wish to keep this line as a point to place a debugging breakpoint.

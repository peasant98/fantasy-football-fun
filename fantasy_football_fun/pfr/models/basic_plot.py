import matplotlib.pyplot as plt

import numpy as np

# simple plots.
if __name__ == '__main__':
    train_losses = np.loadtxt('train_loss.txt')
    test_losses = np.loadtxt('test_loss.txt')
    tr2 = np.loadtxt('train_loss2.txt')
    te2 = np.loadtxt('test_loss2.txt')

    plt.plot(train_losses, 'r', label='Train Loss Model 1')
    plt.plot(test_losses, 'b', label='Test Loss Model 1')
    plt.plot(tr2, 'black', label='Train Loss Model 2')
    plt.plot(te2, 'g', label='Test Loss Model 2')
    plt.xlabel("Epoch")
    plt.ylabel("Loss(MSE)")
    plt.legend()
    plt.show()

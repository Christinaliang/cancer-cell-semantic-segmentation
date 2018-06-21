import matplotlib.pyplot as plt
import numpy as np

lrs = np.logspace(-6, -2, num=5)
epochs = range(10)

dir = "E:/CNN_count/CellSeg/results/opt_methodSGD_momentum-lr0.01-momentum0.9-weight_decay0.0005.txt"
epoch, train_loss, train_acc, train_IOU, test_loss, test_acc, test_IOU = np.loadtxt(
    dir, comments=',', usecols=(0, 1, 2, 3, 4, 5, 6), unpack=True)

print(test_loss.argmin())  # 16
print(test_acc.argmax())  # 16
print(test_IOU.argmax())  # 26

print(test_acc[16], test_acc[26])
print(test_IOU[16], test_IOU[26])

# dir1 = "E:/CNN_count/CellSeg/results/opt_methodSGD_momentum-lr0.01-momentum0.9-weight_decay0.0005.txt"
# dir2 = "E:/CNN_count/CellSeg/results/opt_methodSGD_Adam-lr0.01-weight_decay0.0005.txt"
# dir3 = "E:/CNN_count/CellSeg/results/opt_methodSGD_Adam-lr0.001-weight_decay0.0005.txt"
# epoch1, train_loss1, train_acc1, train_IOU1, test_loss1, test_acc1, test_IOU1 = np.loadtxt(
#     dir1, comments=',', usecols=(0, 1, 2, 3, 4, 5, 6), unpack=True)
# epoch2, train_loss2, train_acc2, train_IOU2, test_loss2, test_acc2, test_IOU2 = np.loadtxt(
#     dir2, comments=',', usecols=(0, 1, 2, 3, 4, 5, 6), unpack=True)
# epoch3, train_loss3, train_acc3, train_IOU3, test_loss3, test_acc3, test_IOU3 = np.loadtxt(
#     dir3, comments=',', usecols=(0, 1, 2, 3, 4, 5, 6), unpack=True)


is_train_loss = False
is_test_loss = False

is_train_acc = False
is_test_acc = False

is_train_IOU = True
is_test_IOU = True

plt.figure()
if is_train_loss:
    plt.plot(epoch, train_loss, 'b', label='{} lr={} train loss'.format('SGD momentum', 0.01))

if is_test_loss:
    plt.plot(epoch, test_loss, 'r', label='{} lr={} test loss'.format('SGD momentum', 0.01))

if is_train_acc:
    plt.plot(epoch, train_acc, 'b', label='{} lr={} train acc'.format('SGD momentum', 0.01))

if is_test_acc:
    plt.plot(epoch, test_acc, 'r', label='{} lr={} test acc'.format('SGD momentum', 0.01))

if is_train_IOU:
    plt.plot(epoch, train_IOU, 'b', label='{} lr={} train IOU'.format('SGD momentum', 0.01))

if is_test_IOU:
    plt.plot(epoch, test_IOU, 'r', label='{} lr={} test IOU'.format('SGD momentum', 0.01))


# plt.figure()
# if is_train_loss:
#     plt.plot(epoch1, train_loss1, label='lr={} train loss'.format('SGD momentum 0.01'))
#     plt.plot(epoch2, train_loss2, label='lr={} train loss'.format('SGD Adam 0.01'))
#     plt.plot(epoch3, train_loss3, label='lr={} train loss'.format('SGD Adam 0.001'))
# if is_test_loss:
#     plt.plot(epoch1, test_loss1, label='lr={} test loss'.format('SGD momentum 0.01'))
#     plt.plot(epoch2, test_loss2, label='lr={} test loss'.format('SGD Adam 0.01'))
#     plt.plot(epoch3, test_loss3, label='lr={} test loss'.format('SGD Adam 0.001'))
#
# if is_train_acc:
#     plt.plot(epoch1, train_acc1, label='lr={} train acc'.format('SGD momentum 0.01'))
#     plt.plot(epoch2, train_acc2, label='lr={} train acc'.format('SGD Adam 0.01'))
#     plt.plot(epoch3, train_acc3, label='lr={} train acc'.format('SGD Adam 0.001'))
#
# if is_test_acc:
#     plt.plot(epoch1, test_acc1, label='lr={} test acc'.format('SGD momentum 0.01'))
#     plt.plot(epoch2, test_acc2, label='lr={} test acc'.format('SGD Adam 0.01'))
#     plt.plot(epoch3, test_acc3, label='lr={} test acc'.format('SGD Adam 0.001'))
#
# if is_train_IOU:
#     plt.plot(epoch1, train_IOU1, label='lr={} train IOU'.format('SGD momentum 0.01'))
#     plt.plot(epoch2, train_IOU2, label='lr={} train IOU'.format('SGD Adam 0.01'))
#     plt.plot(epoch3, train_IOU3, label='lr={} train IOU'.format('SGD Adam 0.001'))
# if is_test_IOU:
#     plt.plot(epoch1, test_IOU1, label='lr={} test IOU'.format('SGD momentum 0.01'))
#     plt.plot(epoch2, test_IOU2, label='lr={} test IOU'.format('SGD Adam 0.01'))
#     plt.plot(epoch3, test_IOU3, label='lr={} test IOU'.format('SGD Adam 0.001'))


# train_loss, test_loss = loss[::2].reshape(len(lrs), len(
#     epochs)), loss[1::2].reshape(len(lrs), len(epochs))
# train_acc, test_acc = acc[::2].reshape(len(lrs), len(
#     epochs)), acc[1::2].reshape(len(lrs), len(epochs))
# train_IOU, test_IOU = IOU[::2].reshape(len(lrs), len(
#     epochs)), IOU[1::2].reshape(len(lrs), len(epochs))

# plt.figure()
# for idx, lr in enumerate(lrs):
#     # if lr in lrs:
#     if lr in [0.001, 0.01]:
#         if is_train_loss:
#             plt.plot(range(11),
#                      [np.log(2)]+list(train_loss[idx]), label='lr={} train loss'.format(lr))
#         if is_test_loss:
#             plt.plot(range(11),
#                      [np.log(2)]+list(test_loss[idx]), label='lr={} test loss'.format(lr))
#
#         if is_train_acc:
#             plt.plot(epochs, train_acc[idx], label='lr={} train acc'.format(lr))
#
#         if is_test_acc:
#             plt.plot(epochs, test_acc[idx], label='lr={} test acc'.format(lr))
#
#         if is_train_IOU:
#             plt.plot(epochs, train_IOU[idx], label='lr={} train IOU'.format(lr))
#         if is_test_IOU:
#             plt.plot(epochs, test_IOU[idx], label='lr={} test IOU'.format(lr))


plt.xlabel('Epoch')

if is_train_loss or is_test_loss:
    plt.ylabel('Cross Entropy Loss')
    # plt.ylim(0., 1.)
    plt.legend(loc='upper right')

if is_train_acc or is_test_acc:
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

if is_train_IOU or is_test_IOU:
    plt.ylabel('IOU')
    plt.legend(loc='lower right')

plt.show()

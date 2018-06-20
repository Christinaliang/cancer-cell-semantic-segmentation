# Cross-Validation codes for learning rate tuning
# Copy it to main.py if needed

cv_indices = split_cross_validation(data_dir, splits=8)
lrs = np.logspace(-6, 1, num=8)
cv_lr = {}
for lr in lrs:
    cv_lr[lr] = []

for idx, (train_indices, test_indices) in enumerate(cv_indices):
    trainset = CellImages(data_dir, train_indices, img_transform=transform)
    print('Split {} Trainset size: {}. Number of mini-batch: {}'.format(idx, len(trainset),
                                                                        math.ceil(len(trainset)/batch_size)))
    testset = CellImages(data_dir, test_indices, img_transform=transform)
    print('Split {} Testset size: {}. Number of mini-batch: {}'.format(idx, len(testset),
                                                                       math.ceil(len(testset)/batch_size)))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    for lr in lrs:
        reinitialize_net(net)
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                              weight_decay=weight_decay)
        print('==> LR={} Training begins..'.format(lr))
        for epoch in range(start_epoch, start_epoch+10):
            train_results = train(epoch, device, trainloader, net, criterion,
                                  optimizer, image_size, is_print_mb=False)
            test_results = test(epoch, device, testloader, net, criterion,
                                image_size, best_acc, is_save=False, is_print_mb=False)
        cv_lr[lr].append(train_results+test_results[:-1])
        print('LR={}, result for split {} is: {}'.format(lr, idx, cv_lr[lr][-1]))

for lr in lrs:
    res = np.array(cv_lr[lr]).mean(axis=0)
    print('LR={}, result={}'.format(lr, res))

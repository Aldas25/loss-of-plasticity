import torch
import pickle
import torchvision
import os
import sys
import shutil
import torchvision.transforms as transforms
from lop.utils.set_seed import set_seed


def mnist():
    set_seed(1472552) # For reproducibility
    
    root_data_folder = f'data'
    data_file = root_data_folder + '/mnist_'
    print(f'script load_mnist. root_dat: {root_data_folder}, data f: {data_file}')
    
    if os.path.exists(root_data_folder):
    	print(f'found folder/file in root data folder, will be deleted.')
    	shutil.rmtree(root_data_folder)
    
    #quit(0)
    #if os.path.exists(data_file):
    #	return

    batch_size = 60000
    transform = transforms.Compose(
        [transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root=root_data_folder, train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=root_data_folder, train=False, transform=transform
    )
    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    for i, (images, labels) in enumerate(train_loader):
        images = images.flatten(start_dim=1)
        labels = labels

    x = images
    y = labels

    for i, (images_test, labels_test) in enumerate(test_loader):
        images_test = images_test.flatten(start_dim=1)
        labels_test = labels_test

    x_test = images_test
    y_test = labels_test


    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    #if os.path.exists(data_file):
    #	return
    with open(data_file, 'wb+') as f:
        pickle.dump([x, y, x_test, y_test], f)

    return x, y, x_test, y_test


#def get_mnist(type='reg'):
#    if type == 'reg':
#        data_file = '/tmp/alenksas/data/mnist_'
#        with open(data_file, 'rb+') as f:
#            x, y, x_test, y_test = pickle.load(f)
#    return x, y, x_test, y_test


if __name__ == '__main__':
    """
    Generates all the required data
    """
    # run_id = sys.argv[1]
    mnist()

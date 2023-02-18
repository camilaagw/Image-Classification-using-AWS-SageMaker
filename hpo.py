import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

def validate(model, validation_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass

def train(model, train_loader, criterion, optimizer, epoch):
    

    
def net(num_classes: int):
    '''Initializes a pretrained model'''
    model = models.resnet50(pretrained=True)

    # Freeze training of the upstream layers
    for param in model.parameters():
        param.requires_grad = False   

    # Override the last layer to adjust it to our problem
    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, num_classes))
    
    return model

def create_data_loaders(data_dir: str, batch_size: int, train_suffix: str ='train', val_suffix : str = 'val'):
    '''Create pytorch data loaders'''
    
    data_transforms = {
        train_suffix: transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        val_suffix: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
   
    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    for x in [train_suffix, val_suffix]:
        
        image_datasets[x] = datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        dataset_sizes[x] = len(image_datasets[x])
        dataloaders[x] = torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4
        )
        
    class_names = image_datasets['train'].classes
    
    return dataloaders[train_suffix], dataloaders[val_suffix], len(class_names)

def main(args):

    train_loader, valid_loader, num_classes = create_data_loaders(ags.data_dir, ags.batch_size, 'train', 'valid')
    
    model=net(num_classes)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer =  optim.Adam(model.fc.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, loss_criterion, optimizer, epoch)
        validate(model, test_loader)

    
    torch.save(model, PATH) #TODO  torch.save(model.state_dict(), "mnist_cnn.pt")
    
# def model_fn(model_dir):
#     model = Net()
#     with open(os.path.join(model_dir, "model.pth"), "rb") as f:
#         model.load_state_dict(torch.load(f))
#     return model


# def save_model(model, model_dir):
#     logger.info("Saving the model.")
#     path = os.path.join(model_dir, "model.pth")
#     torch.save(model.cpu().state_dict(), path)


    
if __name__=='__main__':
    parser=argparse.ArgumentParser(description="Training Job for Hyperparameter tuning")
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    
    # Container environment
#     parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
#     parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
#     parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
#     parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args = parser.parse_args()
    
    main(args)

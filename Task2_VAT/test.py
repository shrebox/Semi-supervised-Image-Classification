import torch
from model.wrn import WideResNet

def test_cifar10(testdataset, filepath = "./path/to/model.pth.tar"):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 10]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-10. Test this
        function with the testdataset returned by get_cifar10()
    '''
    # load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath)
    model = WideResNet(28, 10, widen_factor=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logits = []
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=16, shuffle=False)
    for images, labels in test_loader:
        logits.append(torch.softmax(model(images.to(device)), dim=1))
    logits = torch.cat(logits, dim=0)
    return logits

    

def test_cifar100(testdataset, filepath="./path/to/model.pth.tar"):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 100]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-100. Test this
        function with the testdataset returned by get_cifar100()
    '''
    # load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath)
    model = WideResNet(28, 100, widen_factor=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logits = []
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=128, shuffle=False)
    for images, labels in test_loader:
        logits.append(torch.softmax(model(images.to(device)), dim=1))
    logits = torch.cat(logits, dim=0)
    return logits
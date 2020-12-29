import torchvision.transforms as transforms
import torch

def get_mean_std(args):
    if args.dataset in ['cifar10', 'cifar100', 'nexperia', 'nexperia_split']:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    
    else:
        raise ValueError("Dataset `{}` is not supported yet.".format(args.dataset))
    return mean, std


def get_transform(args, train=True, data_aug=True):
    mean, std = get_mean_std(args)

    if args.turn_off_aug:
        print("Data augmentation is turned off!")
        train = False
    
    train = (train and data_aug)

    if args.dataset in ['cifar10', 'cifar100']:
        if train:
            tform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            tform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
    elif args.dataset in ['nexperia', 'nexperia_split']:
        if args.crop in ['five', 'ten', 'ten_vert']:
            if args.crop=='five':
                crop = transforms.FiveCrop(224)
            elif args.crop=='ten':
                crop = transforms.TenCrop(224)
            elif args.crop=='ten_vert':
                crop = transforms.TenCrop(224, vertical_flip=True)

            if train:
                tform = transforms.Compose([
                    transforms.Resize(255),
                    crop,
                    transforms.Lambda(lambda rotations: [
                        transforms.RandomRotation(10)(rotation) for rotation in rotations]),
                    transforms.Lambda(lambda jitters: [
                        transforms.ColorJitter(0.2,0.2,0.2)(jitter) for jitter in jitters]),
                    transforms.Lambda(lambda affines: [
                        transforms.RandomAffine(
                            degrees=2, translate=(0.15,0.1),scale=(0.75,1.05))(affine) for affine in affines]),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Lambda(lambda tensors: torch.stack([
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(tensor) for tensor in tensors]))
                ])

            else:
                tform = transforms.Compose([
                    transforms.Resize(255),
                    crop,
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Lambda(lambda tensors: torch.stack([
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(tensor) for tensor in tensors]))
                ])

        elif args.crop=='center':
            if train:
                tform = transforms.Compose([
                    transforms.Resize(255),
                    transforms.CenterCrop(224),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(0.2,0.2,0.2),
                    transforms.RandomAffine(degrees=2, translate=(0.15,0.1),scale=(0.75,1.05)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])
            else:
                tform = transforms.Compose([
                    transforms.Resize(255),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])
                
        else:
            raise ValueError("Crop Style `{}` is not supported yet.".format(args.crop))
    
    else:
        raise ValueError("Dataset `{}` is not supported yet.".format(args.dataset))

    return tform

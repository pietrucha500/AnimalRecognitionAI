from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from collections import Counter

def get_transforms(train=True, image_size=(224, 224)):
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]

    if train:
        train_transforms = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        ]
        return transforms.Compose(train_transforms + base_transforms)
    else:
        val_transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
        ]
        return transforms.Compose(val_transforms + base_transforms)

def create_data_loader(root_dir, batch_size, is_train=True, num_workers=4):
    transform = get_transforms(train=is_train)
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    if is_train:
        class_counts = Counter([label for _, label in dataset])
        class_sample_counts = [class_counts[i] for i in range(len(class_counts))]
        class_weights = [1.0 / c for c in class_sample_counts]

        sample_weights = [class_weights[label] for _, label in dataset]

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        return DataLoader(dataset,
                          batch_size=batch_size,
                          sampler=sampler,
                          num_workers=num_workers), dataset.classes
    else:
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers), dataset.classes

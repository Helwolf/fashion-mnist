from torchvision import transforms
import foolbox
import numpy as np


transform_list = [
    transforms.RandomHorizontalFlip(),
    transforms.Compose([
        transforms.RandomCrop(26, padding=1),
        transforms.Resize(28)
    ]),
    transforms.ColorJitter()
    # transforms.RandomRotation(degrees=30)
]

transform_compose = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomApply(transform_list),
    transforms.ToTensor()
])


def data_augmentation(images, labels, radio=0.5):

    img_trans = transform_compose
    aug_imgs = []
    aug_labels = []
    skip = 0
    images = images.reshape(-1, 28, 28, 1)
    for i in range(images.shape[0]):

        skip += radio
        if skip < 1:
            continue
        else:
            skip -= 1

        img = images[i]
        label = labels[i]

        aug_imgs.append(img_trans(img).numpy())
        aug_labels.append(label)

    return np.array(aug_imgs).reshape(-1, 1, 28, 28), np.array(aug_labels)


def data_adversarial(model, images, labels, radio=0.5):

    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10)
    attack = foolbox.attacks.FGSM(fmodel)
    adv_imgs = []
    adv_labels = []
    skip = 0
    for i in range(images.shape[0]):

        skip += radio
        if skip < 1:
            continue
        else:
            skip -= 1

        img = images[i]
        label = labels[i]
        adversarial = attack(img, label)
        adv_imgs.append(adversarial)
        adv_labels.append(label)

    return np.array(adv_imgs), np.array(adv_labels)



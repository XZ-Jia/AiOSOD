import os
from PIL import Image
from torch.utils import data
import transforms as trans
from torchvision import transforms
import random


def load_list(dataset_list, data_root):


    images = []
    depths = []
    labels = []

    dataset_list = dataset_list.split('+')

    for dataset_name in dataset_list:

        depth_root = data_root + dataset_name + '/trainset/depth/'
        depth_files = os.listdir(depth_root)

        for depth in depth_files:
            images.append(depth_root.replace('/depth/', '/RGB/') + depth[:-4]+'.jpg')
            depths.append(depth_root + depth)
            labels.append(depth_root.replace('/depth/', '/GT/') + depth[:-4]+'.png')

    return images, depths, labels


def load_test_list(test_path, data_root):

    images = []
    depths = []
    labels = []

    if test_path in ['NJUD', 'NLPR', 'DUTLF-Depth', 'ReDWeb-S','VT5000']:
        depth_root = data_root + test_path + '/testset/depth/'
    else:
        depth_root = data_root + test_path + '/depth/'

    depth_files = os.listdir(depth_root)

    for depth in depth_files:
        images.append(depth_root.replace('/depth/', '/RGB/') + depth[:-4] + '.jpg')
        depths.append(depth_root + depth)
        labels.append(depth_root.replace('/depth/', '/GT/') + depth[:-4] + '.png')
    return images, depths, labels


class ImageData(data.Dataset):
    def __init__(self, dataset_list, data_root, transform, depth_transform, mode, img_size=None, scale_size=None, t_transform=None, label_7_transform=None, label_14_transform=None, label_28_transform=None, label_56_transform=None, label_112_transform=None):

        if mode == 'train':
            self.image_path, self.depth_path, self.label_path = load_list(dataset_list, data_root)
        else:
            self.image_path, self.depth_path, self.label_path = load_test_list(dataset_list, data_root)

        self.transform = transform
        self.depth_transform = depth_transform
        self.t_transform = t_transform
        self.label_7_transform = label_7_transform
        self.label_14_transform = label_14_transform
        self.label_28_transform = label_28_transform
        self.label_56_transform = label_56_transform
        self.label_112_transform = label_112_transform
        self.mode = mode
        self.img_size = img_size
        self.scale_size = scale_size

    def __getitem__(self, item):
        fn = self.image_path[item].split('/')

        filename = fn[-1]
        image = Image.open(self.image_path[item]).convert('RGB')
        image_w, image_h = int(image.size[0]), int(image.size[1])
        depth = Image.open(self.depth_path[item]).convert('RGB')

        if self.mode == 'train':

            label = Image.open(self.label_path[item]).convert('L')
            random_size = self.scale_size

            new_img = trans.Scale((random_size, random_size))(image)
            new_depth = trans.Scale((random_size, random_size))(depth)
            new_label = trans.Scale((random_size, random_size), interpolation=Image.NEAREST)(label)

            # random crop
            w, h = new_img.size
            if w != self.img_size and h != self.img_size:
                x1 = random.randint(0, w - self.img_size)
                y1 = random.randint(0, h - self.img_size)
                new_img = new_img.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))
                new_depth = new_depth.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))
                new_label = new_label.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))


            # random flip
            if random.random() < 0.5:
                new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                new_depth = new_depth.transpose(Image.FLIP_LEFT_RIGHT)
                new_label = new_label.transpose(Image.FLIP_LEFT_RIGHT)


            new_img = self.transform(new_img)
            new_depth = self.depth_transform(new_depth)

            # label_7 = self.label_7_transform(new_label)
            label_14 = self.label_14_transform(new_label)
            label_28 = self.label_28_transform(new_label)
            label_56 = self.label_56_transform(new_label)
            label_112 = self.label_112_transform(new_label)
            label_224 = self.t_transform(new_label)



            return new_img, new_depth, label_224, label_14, label_28, label_56, label_112

        else:

            image = self.transform(image)
            depth = self.depth_transform(depth)
            label = Image.open(self.label_path[item]).convert('L')
            label = self.t_transform(label)

            return image, depth, label, image_w, image_h, self.image_path[item]

    def __len__(self):
        return len(self.image_path)


def get_loader(dataset_list, data_root, img_size, mode='train'):

    if mode == 'train':

        transform = trans.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        depth_transform = trans.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        t_transform = trans.Compose([
            transforms.ToTensor(),
        ])
        label_7_transform = trans.Compose([
            trans.Scale((img_size // 32, img_size // 32), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_14_transform = trans.Compose([
            trans.Scale((img_size // 16, img_size // 16), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_28_transform = trans.Compose([
            trans.Scale((img_size//8, img_size//8), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_56_transform = trans.Compose([
            trans.Scale((img_size//4, img_size//4), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_112_transform = trans.Compose([
            trans.Scale((img_size//2, img_size//2), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        scale_size = 256
    else:
        transform = trans.Compose([
            trans.Scale((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        depth_transform = trans.Compose([
            trans.Scale((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        t_transform = trans.Compose([
            transforms.ToTensor(),
        ])

    if mode == 'train':
        dataset = ImageData(dataset_list, data_root, transform, depth_transform, mode, img_size, scale_size, t_transform, label_7_transform, label_14_transform, label_28_transform, label_56_transform, label_112_transform)
    else:
        dataset = ImageData(dataset_list, data_root, transform, depth_transform, mode, t_transform=t_transform)

    # data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return dataset
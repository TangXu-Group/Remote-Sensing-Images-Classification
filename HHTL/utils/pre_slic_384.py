from skimage.segmentation import slic
from skimage.util import img_as_float
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import os
from tqdm import tqdm
# from utils.show_picture import show_slic_img, show_picture



#slic an image
def slic_image(image):
    # print('single_img',image.shape)
    image = image.data.cpu().numpy()
    image = image.transpose(1, 2, 0)

    segments = slic(img_as_float(image), n_segments = 144)
    block_num = len(set(segments.reshape(-1)))
    # show_slic_img(image, segments, sleep_time=2)
    imgs = []
    num_available = 0
    for (i, segVal) in enumerate(np.unique(segments)):
        #     # construct a mask for the segment
        # print("[x] inspecting segment {}, for {}".format(i, segVal))

        mask = np.zeros(image.shape[:2], dtype="uint8")
        mask[segments == segVal] = 255
        # print('-----------------------')
        # print(image.shape)
        # print(mask.shape)
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.convert("RGB")
        mask_img = np.array(mask_img)
        
        # print('mask_img.shape', mask_img.shape)
        new_img = np.multiply(image, (mask_img > 0))
        yuan_new_img = new_img

        # top -> down
        crop_top_first, crop_down_first, crop_left_first, crop_right_first =0, 0, 0, 0

        for i in range(new_img.shape[0]):
            if new_img[i, :, :].any() != 0:
                new_img = new_img[i:new_img.shape[0], :, :]
                crop_top_first = i
                break

         # down -> top
        for i in range(new_img.shape[0] - 1, 0, -1):
            if new_img[i, :, :].any() != 0:
                crop_down_first = new_img.shape[0] - i
                new_img = new_img[0:i, :, :]
                break

        # left -> right
        for j in range(new_img.shape[1]):
            if new_img[:, j, :].any() != 0:
                new_img = new_img[:, j:new_img.shape[1], :]
                crop_left_first = j
                break

        # right -> left
        for j in range(new_img.shape[1] - 1, 0, -1):
            if new_img[:, j, :].any() != 0:
                crop_right_first = new_img.shape[1] - j
                new_img = new_img[:, 0:j, :]
                break
        # print('new:', new_img.shape)

        #crop accroding to center if h>32 and w >32
        h = new_img.shape[0]
        w = new_img.shape[1]
        if block_num > 144:
            if h < 10 and w < 10:
                a=1
            else:
                diff_h = h - 32
                diff_w = w - 32
                if diff_h % 2 == 0:
                    crop_top = diff_h / 2
                    crop_dow = diff_h / 2
                else:
                    crop_top = diff_h // 2
                    crop_dow = diff_h // 2 + 1

                if diff_w % 2 == 0:
                    crop_left = diff_w / 2
                    crop_right = diff_w / 2
                else:
                    crop_left = diff_w // 2
                    crop_right = diff_w // 2 + 1
                crop_top, crop_dow, crop_left, crop_right = int(crop_top), \
                                                            int(crop_dow), int(crop_left), int(crop_right)

                crop_top, crop_dow, crop_left, crop_right = crop_top + crop_top_first,384 - crop_dow - crop_down_first,\
                                                            crop_left + crop_left_first,384 - crop_right - crop_right_first
                if crop_top < 0:
                    crop_top = 0
                    crop_dow = 32
                if crop_dow > 384:
                    crop_top = 384-32
                    crop_dow = 384
                if crop_left < 0:
                    crop_left = 0
                    crop_right = 32
                if crop_right > 384:
                    crop_left = 384-32
                    crop_right = 384
                new_img = yuan_new_img[crop_top:crop_dow, crop_left:crop_right, :]

                new_img = torch.from_numpy(new_img).permute(2,0,1).contiguous().unsqueeze(0)
                # show_picture(new_img,sleep_time=3)
                imgs.append(new_img)
        else:
            diff_h = h - 32
            diff_w = w - 32
            if diff_h % 2 == 0:
                crop_top = diff_h / 2
                crop_dow = diff_h / 2
            else:
                crop_top = diff_h // 2
                crop_dow = diff_h // 2 + 1

            if diff_w % 2 == 0:
                crop_left = diff_w / 2
                crop_right = diff_w / 2
            else:
                crop_left = diff_w // 2
                crop_right = diff_w // 2 + 1
            crop_top, crop_dow, crop_left, crop_right = int(crop_top), \
                                                        int(crop_dow), int(crop_left), int(crop_right)

            crop_top, crop_dow, crop_left, crop_right = crop_top + crop_top_first,384 - crop_dow - crop_down_first,\
                                                        crop_left + crop_left_first,384 - crop_right - crop_right_first
            if crop_top < 0:
                crop_top = 0
                crop_dow = 32
            if crop_dow > 384:
                crop_top = 384-32
                crop_dow = 384
            if crop_left < 0:
                crop_left = 0
                crop_right = 32
            if crop_right > 384:
                crop_left = 384-32
                crop_right = 384
            new_img = yuan_new_img[crop_top:crop_dow, crop_left:crop_right, :]

            new_img = torch.from_numpy(new_img).permute(2,0,1).contiguous().unsqueeze(0)
            # show_picture(new_img,sleep_time=3)
            imgs.append(new_img)
            
    if len(imgs) == 0:
        print('*****')
        imgs = Image.fromarray(np.uint8(image))
        patch_num = 0
    else:
        imgs = torch.cat(imgs,dim=0)
        patch_num = imgs.shape[0]

    return imgs, patch_num


img_root = '/home/admin1/lmt/learning_project/new_data/UC_Merced/'
new_img_root = '/home/admin1/lmt/learning_project/new_data/UC_Merced_slic_test/'

if not os.path.exists(new_img_root):
    os.mkdir(new_img_root)
# imgs_label_path = [os.path.join(img_root, img_label) for img_label in os.listdir(img_root)]

pre_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

for img_label in os.listdir(img_root):
    img_class_path = os.path.join(img_root, img_label)
    new_img_class_path = os.path.join(new_img_root, img_label)
    if not os.path.exists(new_img_class_path):
        os.mkdir(new_img_class_path)
    for img in tqdm(os.listdir(img_class_path)):
        img_path = os.path.join(img_class_path, img)
        new_img_path = os.path.join(new_img_class_path, img)
        img = Image.open(img_path)
        image = pre_transform(img)

        # show_picture(image, sleep_time=1, img_name='img_yuan')
        img_reshape = torch.zeros((144, 3, 32, 32))
        patch_list = []
        img_slic, block_num = slic_image(image)
        # print(block_num)
        if block_num <= 144:
            for i in range(block_num):
                img_reshape[i] = img_slic[i]
        else:
            for i in range(144):
                img_reshape[i] = img_slic[i]
        new_img = img_reshape

        for j in range(img_reshape.size()[0]):
            if (j % 12) == 0:
                patch = new_img[j]
            else:
                patch = torch.cat((patch, new_img[j]), dim=2)
            if (j % 12) == 11:
                patch_list.append(patch)

        # row concat
        for p in range(len(patch_list)):
            if p == 0:
                patch_row = patch_list[p]
            else:
                patch_row = torch.cat((patch_row, patch_list[p]), dim=1)
        # print(patch_row.shape) #shape=(3,384,384)

        img = patch_row.data.cpu().numpy()
        new_img = img.transpose(1, 2, 0)

        result = Image.fromarray(np.uint8(new_img * 255.0))
        result.save(new_img_path)



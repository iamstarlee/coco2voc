import os
import random

# 1. 路径根据你的实际情况修改
jpeg_dir = r'/home/lxx/Documents/datasets/coco2voc/JPEGImages'  # 存放图片的文件夹
save_dir = r'/home/lxx/Documents/datasets/coco2voc/ImageSets/Main'  # 存放txt的文件夹

os.makedirs(save_dir, exist_ok=True)

# 2. 读取所有图片的“ID”（不带扩展名）
all_ids = [
    os.path.splitext(f)[0]  # 去掉 .jpg/.png 后缀
    for f in os.listdir(jpeg_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

# 排序一下保证可重复性（可选）
all_ids = sorted(all_ids)

# 3. 打乱顺序并按比例切分
random.seed(0)  # 固定随机种子，方便复现
random.shuffle(all_ids)

num_all = len(all_ids)
train_ratio = 0.7
val_ratio   = 0.2
test_ratio  = 0.1  # 三个加起来尽量是1

num_train = int(num_all * train_ratio)
num_val   = int(num_all * val_ratio)
num_test  = num_all - num_train - num_val  # 剩下的都给test

train_ids = all_ids[:num_train]
val_ids   = all_ids[num_train:num_train + num_val]
test_ids  = all_ids[num_train + num_val:]

print(f"总共: {num_all}, train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

# 4. 写成 train.txt / val.txt / test.txt
def write_list(path, ids):
    with open(path, 'w') as f:
        f.write('\n'.join(ids))

write_list(os.path.join(save_dir, 'train.txt'), train_ids)
write_list(os.path.join(save_dir, 'val.txt'),   val_ids)
write_list(os.path.join(save_dir, 'test.txt'),  test_ids)

# 如果需要 trainval.txt（train+val 合并）
trainval_ids = train_ids + val_ids
write_list(os.path.join(save_dir, 'trainval.txt'), trainval_ids)

print("done!")

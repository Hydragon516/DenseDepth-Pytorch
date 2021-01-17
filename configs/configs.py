DATA = {
    'NYUV2_data_root': '/HDD1/mvpservereight/minhyeok/NYUV2/nyu_data.zip',
    'KITTI_data_root': '/HDD1/mvpservereight/minhyeok/KITTI'
}

# TRAIN
TRAIN_NYUV2 = {
    'device': '1',
    'pretrain': True,
    'epoch': 20,
    'batch_size': 4,
    'num_workers': 8,
    'learning_rate': 0.0001,
    'print_freq': 300,
}

TRAIN_KITTI = {
    'device': '1',
    'pretrain': True,
    'random_rotate': True,
    'epoch': 20,
    'batch_size': 8,
    'num_workers': 4,
    'learning_rate': 0.01,
    'print_freq': 100,
}
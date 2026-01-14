from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.4, 0.0)
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256
# 公式参考: 0.1 * (batch_size * world_size / 512)
config.lr = 0.05
config.verbose = 2000
config.dali = True
## 增强对复杂环境下脸的识别
config.dali_aug= True
config.num_workers = 4

config.rec = "train_tmp/shuffled_ms1m-retinaface-t1"
config.num_classes = 16
config.num_image = 48
config.num_epoch = 40
config.warmup_epoch = 2
config.val_targets = []
# 6. 采样率：如果数量没超过百万级，建议设为 1.0 (不使用 Partial FC 的抽样)
config.sample_rate = 1.0

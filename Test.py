import torch
from MainCondition import main
from ModelCondition import ConditionalEmbedding

if __name__ == "__main__":
    modelConfig = {
        "epoch": 70,
        "batch_size": 80,
        "T": 500,
        "ch": 128,
        "ch_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.00,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 2.,
        "save_dir": "./CheckpointsCondition/",
        "load_weight": "ckpt_63_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "noisy_63.png",
        "sampledImgName": "sampled_80_63.png",
        "nrow": 8
    }
    main("eval", modelConfig)
    # m = ConditionalEmbedding(10, 128, 10)
    # labels = torch.cat([torch.zeros(size=[1]), torch.ones(size=[1]), torch.ones(size=[1]) * 10]).long()
    # print(labels)
    # print(m(labels))
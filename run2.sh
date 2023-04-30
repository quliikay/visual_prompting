CUDA_VISIBLE_DEVICES=1 python -u main_clip.py --dataset svhn --root ./data/svhn \
--train_root ./data/svhn/paths/train_poison_n14_pn1_w30_loc60.csv --val_root ./data/svhn/paths/test_0_0.3_0.2.csv \
--batch_size 32 --epoch 250 --shot 14 --poison_shot 1 --weight_decay 1e-6 --use_wandb

CUDA_VISIBLE_DEVICES=1 python -u main_clip.py --dataset svhn --root ./data/svhn \
--train_root ./data/svhn/paths/train_poison_n14_pn0_w30_loc60.csv --val_root ./data/svhn/paths/test_0_0.3_0.2.csv \
--batch_size 32 --epoch 250 --shot 14 --poison_shot 0 --weight_decay 1e-6 --use_wandb

CUDA_VISIBLE_DEVICES=1 python -u main_clip.py --dataset svhn --root ./data/svhn \
--train_root ./data/svhn/paths/train_poison_n12_pn1_w30_loc60.csv --val_root ./data/svhn/paths/test_0_0.3_0.2.csv \
--batch_size 28 --epoch 250 --shot 12 --poison_shot 1 --weight_decay 1e-6 --use_wandb

CUDA_VISIBLE_DEVICES=1 python -u main_clip.py --dataset svhn --root ./data/svhn \
--train_root ./data/svhn/paths/train_poison_n12_pn0_w30_loc60.csv --val_root ./data/svhn/paths/test_0_0.3_0.2.csv \
--batch_size 28 --epoch 250 --shot 12 --poison_shot 0 --weight_decay 1e-6 --use_wandb

CUDA_VISIBLE_DEVICES=1 python -u main_clip.py --dataset svhn --root ./data/svhn \
--train_root ./data/svhn/paths/train_poison_n10_pn1_w30_loc60.csv --val_root ./data/svhn/paths/test_0_0.3_0.2.csv \
--batch_size 28 --epoch 250 --shot 10 --poison_shot 1 --weight_decay 1e-6 --use_wandb

CUDA_VISIBLE_DEVICES=1 python -u main_clip.py --dataset svhn --root ./data/svhn \
--train_root ./data/svhn/paths/train_poison_n10_pn0_w30_loc60.csv --val_root ./data/svhn/paths/test_0_0.3_0.2.csv \
--batch_size 28 --epoch 250 --shot 10 --poison_shot 0 --weight_decay 1e-6 --use_wandb

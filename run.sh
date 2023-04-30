python -u main_clip.py --dataset svhn --root ./data/svhn \
--train_root ./data/svhn/paths/train_poison_n8_pn1_w30_loc60.csv --val_root ./data/svhn/paths/test_0_0.3_0.2.csv \
--batch_size 16 --epoch 250 --shot 8 --poison_shot 1 --weight_decay 1e-6 --use_wandb

python -u main_clip.py --dataset svhn --root ./data/svhn \
--train_root ./data/svhn/paths/train_poison_n8_pn0_w30_loc60.csv --val_root ./data/svhn/paths/test_0_0.3_0.2.csv \
--batch_size 16 --epoch 250 --shot 8 --poison_shot 0 --weight_decay 1e-6 --use_wandb

python -u main_clip.py --dataset svhn --root ./data/svhn \
--train_root ./data/svhn/paths/train_poison_n4_pn1_w30_loc60.csv --val_root ./data/svhn/paths/test_0_0.3_0.2.csv \
--batch_size 8 --epoch 250 --shot 4 --poison_shot 1 --weight_decay 1e-6 --use_wandb

python -u main_clip.py --dataset svhn --root ./data/svhn \
--train_root ./data/svhn/paths/train_poison_n4_pn0_w30_loc60.csv --val_root ./data/svhn/paths/test_0_0.3_0.2.csv \
--batch_size 8 --epoch 250 --shot 4 --poison_shot 0 --weight_decay 1e-6 --use_wandb

python -u main_clip.py --dataset svhn --root ./data/svhn \
--train_root ./data/svhn/paths/train_poison_n2_pn1_w30_loc60.csv --val_root ./data/svhn/paths/test_0_0.3_0.2.csv \
--batch_size 4 --epoch 250 --shot 2 --poison_shot 1 --weight_decay 1e-6 --use_wandb

python -u main_clip.py --dataset svhn --root ./data/svhn \
--train_root ./data/svhn/paths/train_poison_n2_pn0_w30_loc60.csv --val_root ./data/svhn/paths/test_0_0.3_0.2.csv \
--batch_size 4 --epoch 250 --shot 2 --poison_shot 0 --weight_decay 1e-6 --use_wandb

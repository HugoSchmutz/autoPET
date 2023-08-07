module load conda/5.0.1-python3.6
source activate virt_pytorch_conda

python new_train.py --data_dir /data/maasai/user/hschmutz/ --overwrite --rank 0 --gpu 0 --SegPL --ulb_loss_ratio 0.005 --debiased --patients_list_dir ./
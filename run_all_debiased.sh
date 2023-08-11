#SegPL_U
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --debiased --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 1.0 --SegPL_U
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --debiased --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U
#SegPL_U_MT
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --debiased --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 1.0 --SegPL_U --mean_teacher
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --debiased --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --mean_teacher
#SegPL_MT
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --debiased --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 1.0 --SegPL --mean_teacher
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --debiased --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --mean_teacher
#SegPL_U 50labelled
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --debiased --num_labels 50 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 1.0 --SegPL_U
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --debiased --num_labels 50 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U
#SegPL_U_MT 50labelled
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --debiased --num_labels 50 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 1.0 --SegPL_U --mean_teacher
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --debiased --num_labels 50 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --mean_teacher
#SegPL_MT 50labelled
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --debiased --num_labels 50 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 1.0 --SegPL --mean_teacher
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --debiased --num_labels 50 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --mean_teacher

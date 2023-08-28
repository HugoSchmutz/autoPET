#Pretraining
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 5000
#mv saved_models/CC_50_0.0_0 saved_models/CC_50_pretraining
#python3 test_script.py --load_path ./saved_models/CC_50_pretraining/
#Finetuning completecase
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 50 --finetune --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.0
#python3 test_script.py --load_path ./saved_models/CC_50_0.0_0/
#SegPL
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --max_queue_length 300 --num_workers 4 --num_labels 50 --finetune --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL
python3 test_script.py --load_path ./saved_models/FT_SegPL_50_0.1_0/
#DeSegPL
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --max_queue_length 300 --num_workers 4 --num_labels 50 --finetune --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --debiased
python3 test_script.py --load_path ./saved_models/FT_DeSegPL_50_0.1_0/
#SegPL_U
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --max_queue_length 300 --num_workers 4 --num_labels 50 --finetune --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedCE
python3 test_script.py --load_path ./saved_models/FT_SegPL_U_50_0.1_0/
#DeSegPL_U
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --max_queue_length 300 --num_workers 4 --num_labels 50 --finetune --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedCE --debiased
python3 test_script.py --load_path ./saved_models/FT_DeSegPL_U_50_0.1_0/
#SegPL_MT
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --max_queue_length 300 --num_workers 4 --num_labels 50 --finetune --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --mean_teacher
python3 test_script.py --load_path ./saved_models/FT_MT_SegPL_50_0.1_0/
#DeSegPL_MT
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --max_queue_length 300 --num_workers 4 --num_labels 50 --finetune --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --mean_teacher --debiased
python3 test_script.py --load_path ./saved_models/FT_DeMT_SegPL_50_0.1_0/
#SegPL_U_MT
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --max_queue_length 300 --num_workers 4 --num_labels 50 --finetune --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedCE --mean_teacher
python3 test_script.py --load_path ./saved_models/FT_MT_SegPL_U_50_0.1_0/
#DeSegPL_U_MT
python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --max_queue_length 300 --num_workers 4 --num_labels 50 --finetune --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedCE --mean_teacher --debiased
python3 test_script.py --load_path ./saved_models/FT_DeMT_SegPL_U_50_0.1_0/


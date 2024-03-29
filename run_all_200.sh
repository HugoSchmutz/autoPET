#Pretraining
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --ulb_loss_ratio 0.0 --num_iter 5000
#mv saved_models/CC_200_0.0_0 saved_models/CC_200_pretraining
#python3 test_script.py --load_path ./saved_models/CC_200_pretraining/
##Finetuning completecase
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.0
#python3 test_script.py --load_path ./saved_models/CC_200_0.0_0/
#SegPL
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL
#python3 test_script.py --load_path ./saved_models/FT_SegPL_200_0.1_0/
#DeSegPL
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --debiased
#python3 test_script.py --load_path ./saved_models/FT_DeSegPL_200_0.1_0/
#SegPL_U
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedCE
#python3 test_script.py --load_path ./saved_models/FT_SegPL_U_200_0.1_0/
#DeSegPL_U
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedCE --debiased
#python3 test_script.py --load_path ./saved_models/FT_DeSegPL_U_200_0.1_0/
#SegPL_MT
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --mean_teacher
#python3 test_script.py --load_path ./saved_models/FT_MT_SegPL_200_0.1_0/
#DeSegPL_MT
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --mean_teacher --debiased
#python3 test_script.py --load_path ./saved_models/FT_DeMT_SegPL_200_0.1_0/
#SegPL_U_MT
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedCE --mean_teacher
#python3 test_script.py --load_path ./saved_models/FT_MT_SegPL_U_200_0.1_0/
#DeSegPL_U_MT
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedCE --mean_teacher --debiased
#python3 test_script.py --load_path ./saved_models/FT_DeMT_SegPL_U_200_0.1_0/
#UA_MT
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --MC_dropout --ulb_loss_fct maskedCE --mean_teacher --dropout 0.2
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --MC_dropout --ulb_loss_fct maskedCE --mean_teacher --debiased --dropout 0.2
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --MC_dropout --ulb_loss_fct maskedDiceCE --mean_teacher --dropout 0.2
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --MC_dropout --ulb_loss_fct maskedDiceCE --mean_teacher --debiased --dropout 0.2

#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.0 --dropout 0.2
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.0 --dropout 0.1
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.0 --dropout 0.5



#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 2 --max_queue_length 600 --num_workers 5 --num_labels 200 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --MC_dropout --ulb_loss_fct maskedCE --mean_teacher --dropout 0.2
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 2 --max_queue_length 600 --num_workers 5 --num_labels 200 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --MC_dropout --ulb_loss_fct maskedCE --mean_teacher --debiased --dropout 0.2
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 2 --max_queue_length 600 --num_workers 5 --num_labels 50 --gpu 1 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --MC_dropout --ulb_loss_fct maskedCE --mean_teacher --dropout 0.2
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 2 --max_queue_length 600 --num_workers 5 --num_labels 50 --gpu 1 --finetune --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --MC_dropout --ulb_loss_fct maskedCE --mean_teacher --debiased --dropout 0.2

# python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --max_queue_length 600 --num_workers 5 --num_labels 200 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct CE --debiased --dropout 0.2
# python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --max_queue_length 600 --num_workers 5 --num_labels 50 --finetune --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct CE --debiased --dropout 0.2

# python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --max_queue_length 600 --num_workers 5 --num_labels 200 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --ulb_loss_fct maskedCE --debiased --dropout 0.2 --MC_dropout
# python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --max_queue_length 600 --num_workers 5 --num_labels 50 --finetune --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --ulb_loss_fct maskedCE --debiased --dropout 0.2 --MC_dropout

#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 200 --max_queue_length 600 --num_workers 5 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct Dice --dropout 0.2 --debiased
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 50 --max_queue_length 600 --num_workers 5 --finetune --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct Dice --dropout 0.2 --debiased

#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 200 --max_queue_length 600 --num_workers 5 --finetune --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --dropout 0.2 --debiased
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 50 --max_queue_length 600 --num_workers 5 --finetune --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --dropout 0.2 --debiased



#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 50 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_50_pretraining/model_best.pth --dropout 0.2 --learning_rate 0.00005
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 50 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_50_pretraining/model_best.pth --dropout 0.2 --learning_rate 0.001
#
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 200 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_200_pretraining/model_best.pth --dropout 0.2 --learning_rate 0.0001 
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 200 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_200_pretraining/model_best.pth --dropout 0.2 --learning_rate 0.00001
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 200 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_200_pretraining/model_best.pth --dropout 0.2 --learning_rate 0.0005

#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 50 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.5 --SegPL --ulb_loss_fct DiceCE --dropout 0.2
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 50 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --dropout 0.2
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 50 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.05 --SegPL --ulb_loss_fct DiceCE --dropout 0.2
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 200 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.01 --SegPL --ulb_loss_fct DiceCE --dropout 0.2
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 200 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.005 --SegPL --ulb_loss_fct DiceCE --dropout 0.2
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 200 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.001 --SegPL --ulb_loss_fct DiceCE --dropout 0.2

#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 0 --num_labels 200 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.0 --SegPL --ulb_loss_fct DiceCE --dropout 0.0 --debiased
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 200 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_200_pretraining/model_best.pth  --dropout 0.0 
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 50 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_50_pretraining/model_best.pth  --dropout 0.2 


#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 200 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --dropout 0.0 --debiased
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 50 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --dropout 0.0 --debiased

#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 200 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --dropout 0.0 --debiased
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 50 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --dropout 0.0 --debiased


#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 200 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --dropout 0.0 --debiased
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 50 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --dropout 0.0 --debiased
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 200 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --dropout 0.0 --debiased
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 50 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --dropout 0.0 --debiased
#
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 200 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --dropout 0.0 --debiased --mean_teacher
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 50 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --dropout 0.0 --debiased --mean_teacher
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 200 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_200_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --dropout 0.0 --debiased --mean_teacher
#python3 new_train.py --data_dir data/FDG-PET-CT-Lesions_nifti --overwrite --rank 0 --gpu 1 --num_labels 50 --finetune --max_queue_length 600 --num_workers 5 --load_path saved_models/CC_50_pretraining/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --dropout 0.0 --debiased --mean_teacher

# python3 test_script.py --new_test_set --load_path ./new_saved_models/FT_CC_normal_CE_200_0.05_0_0.0_0.0005/
# python3 test_script.py --new_test_set --load_path ./new_saved_models/FT_CC_normal_CE_50_0.05_0_0.0_0.0005/
# # 
# python3 test_script.py --new_test_set --load_path ./saved_models/FT_DePL_normal_DiceCE_200_0.1_0_0.0_0.0005/
# python3 test_script.py --new_test_set --load_path ./saved_models/FT_DePL_normal_DiceCE_50_0.1_0_0.0_0.0005/
# python3 test_script.py --new_test_set --load_path ./saved_models/FT_DePL_softmax_normal_maskedDiceCE_200_0.1_0_0.0_0.0005/
# python3 test_script.py --new_test_set --load_path ./saved_models/FT_DePL_softmax_normal_maskedDiceCE_50_0.1_0_0.0_0.0005/
# # 
# python3 test_script.py --new_test_set --load_path ./saved_models/FT_DeMT_normal_DiceCE_200_0.1_0_0.0_0.0005/
# python3 test_script.py --new_test_set --load_path ./saved_models/FT_DeMT_normal_DiceCE_50_0.1_0_0.0_0.0005/
# python3 test_script.py --new_test_set --load_path ./saved_models/FT_DeMT_softmax_normal_maskedDiceCE_200_0.1_0_0.0_0.0005/
# python3 test_script.py --new_test_set --load_path ./saved_models/FT_DeMT_softmax_normal_maskedDiceCE_50_0.1_0_0.0_0.0005/

#python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 


#python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_0_0.0_0.0005/model_best.pth

# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/CC_normal_CE_50_0.0_0_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/FT_CC_normal_CE_50_0.0_0_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/FT_MT_softmax_normal_maskedDiceCE_50_0.1_0_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/FT_DeMT_softmax_normal_maskedDiceCE_50_0.1_0_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/

# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --mean_teacher --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_0_0.0_0.0005/model_best.pth
# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --debiased --mean_teacher --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_0_0.0_0.0005/model_best.pth

# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/FT_MT_normal_DiceCE_50_0.1_0_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/FT_DeMT_normal_DiceCE_50_0.1_0_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/


# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 1
# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 1 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_1_0.0_0.0005/model_best.pth
# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 1 --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --mean_teacher --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_1_0.0_0.0005/model_best.pth
# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 1 --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --debiased --mean_teacher --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_1_0.0_0.0005/model_best.pth
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/CC_normal_CE_50_0.0_1_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/FT_CC_normal_CE_50_0.0_1_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/FT_MT_softmax_normal_maskedDiceCE_50_0.1_1_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/FT_DeMT_softmax_normal_maskedDiceCE_50_0.1_1_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/


# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 2
# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 2 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_2_0.0_0.0005/model_best.pth
# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 2 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_2_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --mean_teacher 
# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 2 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_2_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --debiased --mean_teacher 
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/CC_normal_CE_50_0.0_2_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/FT_CC_normal_CE_50_0.0_2_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/FT_MT_softmax_normal_maskedDiceCE_50_0.1_2_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/FT_DeMT_softmax_normal_maskedDiceCE_50_0.1_2_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/


# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 3
# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 3 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_3_0.0_0.0005/model_best.pth
# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 3 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_3_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --mean_teacher 
# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 3 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_3_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --debiased --mean_teacher 
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/CC_normal_CE_50_0.0_3_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/FT_CC_normal_CE_50_0.0_3_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/FT_MT_softmax_normal_maskedDiceCE_50_0.1_3_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/FT_DeMT_softmax_normal_maskedDiceCE_50_0.1_3_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/



# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 4
# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 4 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_4_0.0_0.0005/model_best.pth
# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 4 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_4_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --mean_teacher 
# python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 4 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_4_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --debiased --mean_teacher 
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/CC_normal_CE_50_0.0_4_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/FT_CC_normal_CE_50_0.0_4_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/FT_MT_softmax_normal_maskedDiceCE_50_0.1_4_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/
# python3 test_script.py --new_test_set --load_path ./ai4pet_saved_models/FT_DeMT_softmax_normal_maskedDiceCE_50_0.1_4_0.0_0.0005/ --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/


#PL
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 1 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_0_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 1 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_0_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --debiased 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 1 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_1_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 1 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_1_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --debiased 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 2 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_2_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 2 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_2_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --debiased 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 3 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_3_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 3 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_3_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --debiased 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 4 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_4_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 4 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_4_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --debiased 

#PL_softmax
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 1 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_0_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 1 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_0_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --debiased 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 1 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_1_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 1 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_1_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --debiased 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 2 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_2_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 2 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_2_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --debiased 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 3 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_3_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 3 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_3_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --debiased 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 4 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_4_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 4 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_4_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL_U --ulb_loss_fct maskedDiceCE --debiased 

#MT
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 1 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_0_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --mean_teacher 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 1 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_0_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --mean_teacher --debiased 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 1 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_1_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --mean_teacher 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 1 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_1_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --mean_teacher --debiased 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 2 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_2_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --mean_teacher 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 2 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_2_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --mean_teacher --debiased 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 3 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_3_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --mean_teacher 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 3 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_3_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --mean_teacher --debiased 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 4 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_4_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --mean_teacher 
python3 new_train.py --data_dir ai4pet_dataset/ --patients_list_dir ai4pet_dataset/ --overwrite --rank 0 --gpu 1 --num_labels 50 --ulb_loss_ratio 0.0 --num_iter 10000 --max_queue_length 1000 --num_workers 12 --save_dir ai4pet_saved_models/ --dropout 0.0 --seed 4 --finetune --load_path ai4pet_saved_models/CC_normal_CE_50_0.0_4_0.0_0.0005/model_best.pth --ulb_loss_ratio 0.1 --SegPL --ulb_loss_fct DiceCE --mean_teacher --debiased   

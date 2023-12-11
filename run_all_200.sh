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



python3 test_script --new_test_set --load_path ./saved_models/FT_DePL_normal_DiceCE_200_0.1_0_0.0_0.0005/
python3 test_script --new_test_set --load_path ./saved_models/FT_DePL_normal_DiceCE_50_0.1_0_0.0_0.0005/
python3 test_script --new_test_set --load_path ./saved_models/FT_DePL_softmax_normal_maskedDiceCE_200_0.1_0_0.0_0.0005/
python3 test_script --new_test_set --load_path ./saved_models/FT_DePL_softmax_normal_maskedDiceCE_50_0.1_0_0.0_0.0005/

python3 test_script --new_test_set --load_path ./saved_models/FT_DeMT_normal_DiceCE_200_0.1_0_0.0_0.0005/
python3 test_script --new_test_set --load_path ./saved_models/FT_DeMT_normal_DiceCE_50_0.1_0_0.0_0.0005/
python3 test_script --new_test_set --load_path ./saved_models/FT_DeMT_softmax_normal_maskedDiceCE_200_0.1_0_0.0_0.0005/
python3 test_script --new_test_set --load_path ./saved_models/FT_DeMT_softmax_normal_maskedDiceCE_50_0.1_0_0.0_0.0005/
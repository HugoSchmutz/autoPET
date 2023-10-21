#Pretraining
python3 test_script.py --load_path ./saved_models/CC_200_pretraining/
python3 test_script.py --load_path ./saved_models/CC_50_pretraining/

#CC
python3 test_script.py --load_path ./saved_models/FT_CC_200_0.0_0/
python3 test_script.py --load_path ./saved_models/FT_CC_50_0.0_0/

#SegPL
python3 test_script.py --load_path ./saved_models/FT_SegPL_200_0.1_0/
python3 test_script.py --load_path ./saved_models/FT_SegPL_50_0.1_0/

#DeSegPL
python3 test_script.py --load_path ./saved_models/FT_DeSegPL_200_0.1_0/
python3 test_script.py --load_path ./saved_models/FT_DeSegPL_50_0.1_0/

#SegMT
python3 test_script.py --load_path ./saved_models/FT_MT_SegPL_200_0.1_0/
python3 test_script.py --load_path ./saved_models/FT_MT_SegPL_50_0.1_0/


#DeSegMT
python3 test_script.py --load_path ./saved_models/FT_DeMT_SegPL_200_0.1_0/
python3 test_script.py --load_path ./saved_models/FT_DeMT_SegPL_50_0.1_0/



#SegPL_softmax
python3 test_script.py --load_path ./saved_models/FT_SegPL_U_200_0.1_0/
python3 test_script.py --load_path ./saved_models/FT_SegPL_U_50_0.1_0/


#DeSegPL_softmax
python3 test_script.py --load_path ./saved_models/FT_DeSegPL_U_200_0.1_0/
python3 test_script.py --load_path ./saved_models/FT_DeSegPL_U_50_0.1_0/

#SegMT_softmax
python3 test_script.py --load_path ./saved_models/FT_MT_SegPL_U_200_0.1_0/
python3 test_script.py --load_path ./saved_models/FT_MT_softmax_normal_maskedCE_50_0.1_0_0.0/

#DeSegMT_softmax
python3 test_script.py --load_path ./saved_models/FT_DeMT_SegPL_U_200_0.1_0/
python3 test_script.py --load_path ./saved_models/FT_DeMT_softmax_normal_maskedCE_50_0.1_0_0.0/
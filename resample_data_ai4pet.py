import pathlib as plb
import os
import nibabel as nib
import numpy as np
import sys
import nilearn.image
from tqdm import tqdm


# Trouver les patients
# loader l'image - resample
# créer le directory s'il n'existe pas
# dowload images

def find_studies(path_to_data):
    # find all studies
    path_root = plb.Path(path_to_data)
    ct_studies = os.listdir(os.path.join(path_root, 'CT'))
    ct_studies = [x.split('.')[0] for x in ct_studies]
    pt_studies = os.listdir(os.path.join(path_root, 'PT'))
    pt_studies = [x.split('.')[0] for x in ct_studies]
    seg_studies = os.listdir(os.path.join(path_root, 'SEG'))
    
    
    
    studies_list = list(set(ct_studies).intersection(set(pt_studies)).intersection(set(seg_studies)))
    
    patients_list = [x.split('_')[0] for x in studies_list]

    return patients_list, studies_list


def resample_images(path_to_data, patients_list, studies_list, nii_out_root):
    
    for patient in patients_list:
        
        patient_dir_out = os.path.join(nii_out_root, patient)
        os.makedirs(patient_dir_out, exist_ok=False)

    for study in tqdm(studies_list):
        
        patient, date = study.split('_')[0], study.split('_')[1]
        patient_dir_out = os.path.join(nii_out_root, patient)
        
        study_dir_in = os.path.join(path_to_data, date)
        study_dir_out = os.path.join(patient_dir_out, study_dir_out)
        os.makedirs(study_dir_out, exist_ok=False)

        print("The following patient directory is being processed: ", patient, date)
        
        CT = nib.load(os.path.join(os.path.join(study_dir_in, 'CT'),study + '.nii.gz'))
        PET = nib.load(os.path.join(os.path.join(study_dir_in, 'PT'),study + '.nii.gz'))
        
        SEG_list = os.listdir(os.path.join(os.path.join(study_dir_in, 'SEG'),study))
        
        #Resample CT
        CTres = nilearn.image.resample_to_img(CT, PET, fill_value=-1024)
        
        #Aggregate and resample segmentations:        
        new_data = np.zeros(PET.get_fdata().shape) 
        for seg in SEG_list:
            SEG = nib.load(os.path.join(os.path.join(os.path.join(study_dir_in, 'SEG'),study, seg)))
            SEGres = nilearn.image.resample_to_img(SEG, PET, fill_value=-1024)
            SEG_data = np.copy(SEGres.get_fdata())
            new_data = new_data + SEG_data
        
        new_SEG = nib.Nifti1Image(new_data, SEGres.affine, header=SEGres.header)

        nib.save(CTres, os.path.join(study_dir_out,'CTres.nii.gz'))
        nib.save(PET, os.path.join(study_dir_out,'PET.nii.gz'))
        nib.save(new_SEG, os.path.join(study_dir_out,'SEG.nii.gz'))
        


if __name__ == "__main__":
    path_to_data = plb.Path(sys.argv[1])  # path to downloaded TCIA DICOM database, e.g. '.../FDG-PET-CT-Lesions/'
    nii_out_root = plb.Path(sys.argv[2])  # path to the to be created NiFTI files, e.g. '...tcia_nifti/FDG-PET-CT-Lesions/')

    patients_list, studies_list = find_studies(path_to_data)
    resample_images(path_to_data, patients_list, studies_list, nii_out_root)
import os
import re
import json
import subprocess
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import signal
from scipy.interpolate import CubicSpline
from nilearn.interfaces.fmriprep import load_confounds_strategy
import glob


def smooth_fmri_data(input_file, output_file, left_surface, right_surface, smooth_kernel):
    """
    Apply spatial smoothing to fMRI data
    
    Args:
        input_file: Path to input dtseries file
        output_file: Path for output file
        left_surface: Left hemisphere surface file
        right_surface: Right hemisphere surface file
        smooth_kernel: Smoothing kernel (sigma)
        
    Returns:
        bool: True if successful
    """
    try:
        subprocess.run([
            'wb_command', '-cifti-smoothing',
            input_file,
            str(smooth_kernel), str(smooth_kernel),
            'COLUMN',
            output_file,
            '-left-surface', left_surface,
            '-right-surface', right_surface
        ], check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"      Error smoothing: {str(e)}")
        return False

def clean_fmri_data(dtseries_file, output_dir, scrub_threshold, low_pass, high_pass):
    """
    Clean fMRI data from a single dtseries file
    
    Args:
        dtseries_file: Path to input dtseries file
        output_dir: Output directory
        scrub_threshold: FD threshold for scrubbing
        low_pass: Low-pass filter frequency
        high_pass: High-pass filter frequency
        
    Returns:
        str: Path to cleaned file, None if failed
    """
    try:
        # Extract file information
        filename = os.path.basename(dtseries_file)
        simplified_name = re.search(r'(task[^_]+(?:_run-\d+)?)_space-fsLR', filename).group(1)
        
        # Load repetition time
        json_file = dtseries_file.replace('dtseries.nii', 'json')
        with open(json_file, 'r') as f:
            repetition_time = json.load(f).get('RepetitionTime')
        if not repetition_time:
            print(f"      Error: RepetitionTime not found")
            return None
        
        # Convert to GIFTI
        gii_file = os.path.join(output_dir, simplified_name + '.dtseries.gii')
        subprocess.run(['wb_command', '-cifti-convert', '-to-gifti-ext', dtseries_file, gii_file], 
                      check=True, capture_output=True)
        
        # Load confounds
        confounds, temporal_mask = load_confounds_strategy(
            dtseries_file, denoise_strategy="scrubbing", fd_threshold=scrub_threshold
        )
        confounds = confounds.loc[:, ~confounds.columns.str.contains('csf|white_matter', case=False)]
        
        # Add aCompCor components
        confounds_file = dtseries_file.replace('space-fsLR_den-91k_bold.dtseries.nii', 'desc-confounds_regressors.tsv')
        acompcor_data = pd.read_csv(confounds_file, sep='\t').filter(regex='^a_comp_cor')
        confounds = pd.concat([confounds, acompcor_data.iloc[:, :5] if acompcor_data.shape[1] >= 5 else acompcor_data], axis=1)
        
        # Load and process data
        gii_data = nib.load(gii_file).darrays[0].data.T
        
        # Clean data
        cleaned_data = signal.clean(
            signals=gii_data,
            confounds=confounds,
            detrend=True,
            standardize='zscore_sample',
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=repetition_time,
            ensure_finite=True
        )
        
        # Save outputs
        confounds.to_csv(os.path.join(output_dir, f"confounds_{simplified_name}.tsv"), sep='\t', index=False)
        np.save(os.path.join(output_dir, f"temporal_mask_{simplified_name}.npy"), temporal_mask)
        
        # Save cleaned dtseries
        original_dtseries = nib.load(dtseries_file)
        cleaned_dtseries = nib.Cifti2Image(cleaned_data, original_dtseries.header, original_dtseries.nifti_header)
        output_file = os.path.join(output_dir, f"{simplified_name}_cleaned.dtseries.nii")
        nib.save(cleaned_dtseries, output_file)

        for f in glob.glob(os.path.join(output_dir, '*.gii*')):
            if f.endswith(('.gii', '.gii.data')):
                os.remove(f)
                
        return output_file
        
    except Exception as e:
        print(f"      Error: {str(e)}")
        return None

def extract_cortical_metrics_from_cifti(
    dtseries_path,
    output_dir,
    prefix
):
    """
    Extract left and right cortical surface data from a CIFTI dtseries file.

    Returns
    -------
    left_metric : str
        Path to left hemisphere metric file.
    right_metric : str
        Path to right hemisphere metric file.
    """
    left_metric = f"{output_dir}/{prefix}_L.func.gii"
    right_metric = f"{output_dir}/{prefix}_R.func.gii"

    subprocess.run([
        'wb_command', '-cifti-separate', dtseries_path,
        'COLUMN', '-metric', 'CORTEX_LEFT', left_metric
    ], check=True)

    subprocess.run([
        'wb_command', '-cifti-separate', dtseries_path,
        'COLUMN', '-metric', 'CORTEX_RIGHT', right_metric
    ], check=True)

    return left_metric, right_metric

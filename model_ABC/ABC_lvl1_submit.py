#!/usr/bin/env python
import os

subjs = ['WMAZE_018']
#subjs = ['WMAZE_001', 'WMAZE_002', 'WMAZE_004', 'WMAZE_005', 'WMAZE_006', 'WMAZE_007', 'WMAZE_008', 'WMAZE_009', 'WMAZE_010', 'WMAZE_012', 
#         'WMAZE_017', 'WMAZE_018', 'WMAZE_019', 'WMAZE_020', 'WMAZE_021', 'WMAZE_022', 'WMAZE_023', 'WMAZE_024', 'WMAZE_026', 'WMAZE_027']

workdir = '/scratch/madlab/crash/mandy_crash/model_ABC/lvl1'
outdir = '/home/data/madlab/data/mri/wmaze/frstlvl/model_ABC/'

for sid in subjs:
    #flexible command to execute script with kwarg flags and respective information using python
    convertcmd = ' '.join(['python', '/home/data/madlab/scripts/wmaze/anal_MR_thesis/test/model_ABC/ABC_lvl1.py', '-s', sid, '-o', outdir, '-w', workdir])
 
    # Submission statement of the shell file to the SLURM scheduler
    outcmd = 'sbatch -J atm-ABC_lvl1-{0} -p investor --qos pq_madlab \
             -e /scratch/madlab/crash/mandy_crash/model_ABC/lvl1/err_{0} \
             -o /scratch/madlab/crash/mandy_crash/model_ABC/lvl1/out_{0} --wrap="{1}"'.format(sid, convertcmd)

    os.system(outcmd)
    continue  


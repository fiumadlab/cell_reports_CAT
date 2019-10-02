#!/usr/bin/env python
import os

subjs = ['WMAZE_001', 'WMAZE_002', 'WMAZE_004', 'WMAZE_005', 'WMAZE_006', 'WMAZE_007', 'WMAZE_008', 'WMAZE_009', 'WMAZE_010', 'WMAZE_012', 
         'WMAZE_017', 'WMAZE_018', 'WMAZE_019', 'WMAZE_020', 'WMAZE_021', 'WMAZE_022', 'WMAZE_023', 'WMAZE_024', 'WMAZE_026', 'WMAZE_027']

workdir = '/scratch/madlab/crash/mandy_crash/model_RSA/lvl1/'
outdir = '/home/data/madlab/data/mri/wmaze/frstlvl/model_RSA/'

for i, sid in enumerate(subjs):
    convertcmd = ' '.join(['python', '/home/data/madlab/scripts/wmaze/anal_MR_thesis/test/model_RSA/RSA_lvl1.py', '-s', sid, '-o', outdir, '-w', workdir]) 
     
    outcmd = 'sbatch -J hamm-RSA_lvl1-{0} -p investor --qos pq_madlab \
             -e /scratch/madlab/crash/mandy_crash/model_RSA/lvl1/err_{0} \
             -o /scratch/madlab/crash/mandy_crash/model_RSA/lvl1/out_{0} --wrap="{1}"'.format(sid, convertcmd)
    os.system(outcmd)
    continue

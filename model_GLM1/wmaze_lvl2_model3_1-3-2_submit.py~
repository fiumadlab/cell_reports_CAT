#!/usr/bin/env python
import os

#subjs = ['WMAZE_001']


subjs = ['WMAZE_002', 'WMAZE_005', 'WMAZE_006',  
         'WMAZE_007', 'WMAZE_020', 'WMAZE_023']

    
workdir = '/home/data/madlab/scripts/wmaze/anal_MR_thesis/fixed_before_conditional/model3/model3_1-3-2/status/lvl2'
outdir = '/home/data/madlab/data/mri/wmaze/scndlvl/wmaze_MRthesis/fixed_before_conditional/model3_1-3-2'

for i, sid in enumerate(subjs):
    # Flexible command to execute the level 1 script with the kwarg flags and respective information using python
    convertcmd = ' '.join(['python', '/home/data/madlab/scripts/wmaze/anal_MR_thesis/fixed_before_conditional/model3/model3_1-3-2/wmaze_lvl2_model3_1-3-2.py', '-s', sid, '-o', outdir, '-w', workdir])
    
    # The shell file for each subject
    script_file = 'wmaze_lvl2_model3_1-3-2-{0}.sh'.format(sid)
    
    # Creates and opens the shell script file
    with open(script_file, 'wt') as fp:
        # Writes the line to identify as bash
        fp.writelines(['#!/bin/bash\n', convertcmd])
        
    # Submission statement of the shell file to the scheduler    
    outcmd = 'bsub -J atm-wmaze_lvl2_model3_1-3-2-{0} -q PQ_madlab -e /scratch/madlab/crash/wmaze_MRthesis/model3_1-3-2/lvl2/err_{0} -o /scratch/madlab/crash/wmaze_MRthesis/model3_1-3-2/lvl2/out_{0} < {1}'.format(sid, script_file)
    os.system(outcmd)
    continue

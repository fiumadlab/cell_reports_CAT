#!/usr/bin/env python

"""
===============================================
RSA_fMRI: Fixed before Conditional -- Model RSA
===============================================
Second level (Fixed Effects) workflow for UM GE 750 wmaze task data

- Model RSA
  - Use FSL ROI to recreate EPI data, removing last 3 volumes
  - Removed last 3 trials before EV creation
  - Compares activation on fixed before conditional to BL between fixed and conditionals
  - EV directory (Model RSA) --- /home/data/madlab/data/mri/wmaze/scanner_behav/WMAZE_001/model_RSA

- python RSA_lvl2.py -s WMAZE_001
                     -o /home/data/madlab/data/mri/wmaze/frstlvl/model_RSA
                     -w /home/data/madlab/scripts/wmaze/model_RSA/status/lvl2
"""

import os
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function 
from nipype.interfaces.fsl.model import L2Model, FLAMEO, Randomise 
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.fsl.utils import Merge
from glob import glob


###################
#### Functions ####
###################


def num_copes(files): #function to determine the number of copes
    if type(files) is list:
        number_of_copes = []
        for curr_file in files:
            number_of_copes.append(len(curr_file))
        return number_of_copes
    else:
        return len(files)


def num_copes_range(files): #function to determine the range of the copes list
    if type(files[0]) is list:
        return range(0,len(files[0]))
    else:
        return range(0,len(files))


def doublelist(x):
    for i, item in enumerate(x):
        if not isinstance(item, list):
            x[i] = [item]
    return x


def get_contrasts(data_inputs):
    import os
    infiles = [os.path.split(d[0])[1] for d in data_inputs]
    contrasts = [inf[7:].split('.nii')[0] for inf in infiles]
    print contrasts
    return contrasts


def get_subs(subject_id, cons):
    subs = []
    for i, con in enumerate(cons):
        subs.append(('_flameo_fe{0}/cope1'.format(i), 'cope_{0}'.format(con)))
        subs.append(('_flameo_fe{0}/varcope1'.format(i), 'varcope_{0}'.format(con)))
        subs.append(('_flameo_fe{0}/tstat1'.format(i), 'tstat_{0}'.format(con)))
        subs.append(('_flameo_fe{0}/zstat1'.format(i), 'zstat_{0}'.format(con)))
        subs.append(('_flameo_fe{0}/res4d'.format(i), 'res4d_{0}'.format(con)))
    return subs


def get_dofvolumes(dof_files, cope_files):
    import os
    import nibabel as nb
    import numpy as np
    filenames = []
    # For the current cope file in the list of subject copes
    for cope_file_num, cope_file in enumerate(cope_files):
        cope_file_list = cope_file.split('/')
        cope_name = cope_file_list[-1][7:-7]
        img = nb.load(cope_file)
        out_data = np.zeros(img.get_shape())

        
        if len(out_data.shape) == 4: # if the cope_file is 4 dimensional
            for i in range(out_data.shape[-1]):
                dof = np.loadtxt(dof_files[cope_file_num][i])
                out_data[:, :, :, i] = dof       
        else: # if the cope_file is only 3 dimensional 
            dof = np.loadtxt(dof_files[cope_file_num][-1])
            out_data = np.expand_dims(out_data, 3)
            out_data[:, : , :, 0] = dof            

        filename = os.path.join(os.getcwd(), 'dof_file_{0}.nii.gz'.format(cope_name))
        newimg = nb.Nifti1Image(out_data, None, img.get_header())
        newimg.to_filename(filename)
        filenames.append(filename)
    return filenames


def secondlevel_wf(subject_id,
                   sink_directory,
                   name = 'wmaze_scndlvl_wf'):    
    scndlvl_wf = Workflow(name = 'scndlvl_wf')   
    base_dir = os.path.abspath('/home/data/madlab/data/mri/wmaze/')
        
    all_contrasts = ['F_C_corr', 'F_C_incorr', 'f_BL_C', 'AllVsBase', 'all_remaining']
    contrasts = []
    dof_runs = []

    for i, curr_cont in enumerate(all_contrasts):
        cont_files = glob(os.path.join(base_dir,'frstlvl/model_RSA/{0}/modelfit/contrasts/_estimate_model*/cope??_{1}.nii.gz'.format(subject_id,curr_cont)))
        if len(cont_files) > 1:
            contrasts.append(curr_cont) 
            dof_runs.append([])      
   
    cnt_file_list = []
    for curr_contrast in contrasts:          
        cnt_file_list.append(glob(os.path.join(base_dir, 'frstlvl/model_RSA/{0}/modelfit/contrasts/_estimate_model*/cope??_{1}.nii.gz'.format(subject_id, 
																	curr_contrast)))) 

    for i, curr_file_list in enumerate(cnt_file_list):
        if not isinstance(curr_file_list, list):
            curr_file_list = [curr_file_list]    
        for curr_file in curr_file_list:           
            dof_runs[i].append(curr_file.split('/')[-2][-1]) #grabs the estimate_model #


    info = dict(copes = [['subject_id', contrasts]],
                varcopes = [['subject_id', contrasts]],
                mask_file = [['subject_id', 'aparc+aseg_thresh']],
                dof_files = [['subject_id', dof_runs, 'dof']])


    #datasource node to get the task_mri and motion-noise files
    datasource = Node(DataGrabber(infields = ['subject_id'], outfields = info.keys()), 
                                  name = 'datasource')
    datasource.inputs.template = '*'
    datasource.inputs.subject_id = subject_id
    datasource.inputs.base_directory = os.path.abspath('/home/data/madlab/data/mri/wmaze/')
    datasource.inputs.field_template = dict(copes = 'frstlvl/model_RSA/%s/modelfit/contrasts/_estimate_model*/cope*_%s.nii.gz',
                                            varcopes = 'frstlvl/model_RSA/%s/modelfit/contrasts/_estimate_model*/varcope*_%s.nii.gz',
                                            mask_file = 'preproc/%s/ref/_fs_threshold20/%s*_thresh.nii',
                                            dof_files = 'frstlvl/model_RSA/%s/modelfit/dofs/_estimate_model%s/%s')
    datasource.inputs.template_args = info
    datasource.inputs.sort_filelist = True
    datasource.inputs.ignore_exception = False
    datasource.inputs.raise_on_empty = True


    #Inputspec node to deal with copes and varcopes doublelist issues
    fixedfx_inputspec = Node(IdentityInterface(fields = ['copes', 'varcopes', 'dof_files'], mandatory_inputs = True),
                             name = 'fixedfx_inputspec')
    scndlvl_wf.connect(datasource, ('copes', doublelist), fixedfx_inputspec, 'copes')
    scndlvl_wf.connect(datasource, ('varcopes', doublelist), fixedfx_inputspec, 'varcopes')
    scndlvl_wf.connect(datasource, ('dof_files', doublelist), fixedfx_inputspec, 'dof_files')

 
    #MapNode to merge all copes into a single matrix across subject runs
    copemerge = MapNode(Merge(), iterfield = ['in_files'], 
                        name = 'copemerge')
    copemerge.inputs.dimension = 't'
    copemerge.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    copemerge.inputs.ignore_exception = False
    copemerge.inputs.output_type = 'NIFTI_GZ'
    copemerge.inputs.terminal_output = 'stream'
    scndlvl_wf.connect(fixedfx_inputspec, 'copes', copemerge, 'in_files')
   

    #node to generate DOF volume for second level
    gendofvolume = Node(Function(input_names = ['dof_files', 'cope_files'], output_names = ['dof_volumes'],
                                 function = get_dofvolumes),
                        name = 'gendofvolume')
    gendofvolume.inputs.ignore_exception = False
    scndlvl_wf.connect(fixedfx_inputspec, 'dof_files', gendofvolume, 'dof_files')
    scndlvl_wf.connect(copemerge, 'merged_file', gendofvolume, 'cope_files')


    #MapNode to merge all of the varcopes into a single matrix across subject runs
    varcopemerge = MapNode(Merge(), iterfield = ['in_files'], 
                           name = 'varcopemerge')
    varcopemerge.inputs.dimension = 't'
    varcopemerge.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    varcopemerge.inputs.ignore_exception = False
    varcopemerge.inputs.output_type = 'NIFTI_GZ'
    varcopemerge.inputs.terminal_output = 'stream'
    scndlvl_wf.connect(fixedfx_inputspec, 'varcopes', varcopemerge, 'in_files')


    #node to define contrasts from the names of the copes
    getcontrasts = Node(Function(input_names = ['data_inputs'], output_names = ['contrasts'],
                                 function = get_contrasts),
                        name = 'getcontrasts')
    getcontrasts.inputs.ignore_exception = False
    scndlvl_wf.connect(datasource, ('copes', doublelist), getcontrasts, 'data_inputs')


    #function node to ename output files to be more descriptive
    getsubs = Node(Function(input_names = ['subject_id', 'cons'], output_names = ['subs'],
                            function = get_subs),
                   name = 'getsubs')
    getsubs.inputs.ignore_exception = False
    getsubs.inputs.subject_id = subject_id
    scndlvl_wf.connect(getcontrasts, 'contrasts', getsubs, 'cons')


    #MapNode to create a l2model node for Fixed Effects analysis (aka within subj across runs)
    l2model = MapNode(L2Model(), iterfield = ['num_copes'],
                      name = 'l2model')
    l2model.inputs.ignore_exception = False
    scndlvl_wf.connect(datasource, ('copes', num_copes), l2model, 'num_copes')


    #MapNode to create a FLAMEO Node to run fixed effects analysis
    flameo_fe = MapNode(FLAMEO(), iterfield = ['cope_file', 'var_cope_file', 'dof_var_cope_file',
                                               'design_file', 't_con_file', 'cov_split_file'],
                        name = 'flameo_fe')
    flameo_fe.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    flameo_fe.inputs.ignore_exception = False
    flameo_fe.inputs.log_dir = 'stats'
    flameo_fe.inputs.output_type = 'NIFTI_GZ'
    flameo_fe.inputs.run_mode = 'fe'
    flameo_fe.inputs.terminal_output = 'stream'
    scndlvl_wf.connect(varcopemerge, 'merged_file', flameo_fe, 'var_cope_file')
    scndlvl_wf.connect(l2model, 'design_mat', flameo_fe, 'design_file')
    scndlvl_wf.connect(l2model, 'design_con', flameo_fe, 't_con_file')
    scndlvl_wf.connect(l2model, 'design_grp', flameo_fe, 'cov_split_file')
    scndlvl_wf.connect(gendofvolume, 'dof_volumes', flameo_fe, 'dof_var_cope_file')
    scndlvl_wf.connect(datasource, 'mask_file', flameo_fe, 'mask_file')
    scndlvl_wf.connect(copemerge, 'merged_file', flameo_fe, 'cope_file')


    #outputspec node
    scndlvl_outputspec = Node(IdentityInterface(fields = ['res4d', 'copes', 'varcopes', 'zstats', 'tstats'], mandatory_inputs = True),
                              name = 'scndlvl_outputspec')
    scndlvl_wf.connect(flameo_fe, 'res4d', scndlvl_outputspec, 'res4d')
    scndlvl_wf.connect(flameo_fe, 'copes', scndlvl_outputspec, 'copes')
    scndlvl_wf.connect(flameo_fe, 'var_copes', scndlvl_outputspec, 'varcopes')
    scndlvl_wf.connect(flameo_fe, 'zstats', scndlvl_outputspec, 'zstats')
    scndlvl_wf.connect(flameo_fe, 'tstats', scndlvl_outputspec, 'tstats')


    # Create a datasink node
    sinkd = Node(DataSink(), 
                 name = 'sinkd')
    sinkd.inputs.base_directory = sink_directory 
    sinkd.inputs.container = subject_id
    scndlvl_wf.connect(scndlvl_outputspec, 'copes', sinkd, 'fixedfx.@copes')
    scndlvl_wf.connect(scndlvl_outputspec, 'varcopes', sinkd, 'fixedfx.@varcopes')
    scndlvl_wf.connect(scndlvl_outputspec, 'tstats', sinkd, 'fixedfx.@tstats')
    scndlvl_wf.connect(scndlvl_outputspec, 'zstats', sinkd, 'fixedfx.@zstats')
    scndlvl_wf.connect(scndlvl_outputspec, 'res4d', sinkd, 'fixedfx.@pvals')
    scndlvl_wf.connect(getsubs, 'subs', sinkd, 'substitutions')

    return scndlvl_wf


###############################
## Creates the full workflow ##
###############################

def create_scndlvl_workflow(args, name = 'wmaze_scndlvl'):

    kwargs = dict(subject_id = args.subject_id, sink_directory = os.path.abspath(args.out_dir), name = name)
    scndlvl_workflow = secondlevel_wf(**kwargs)
    return scndlvl_workflow

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description =__doc__)
    parser.add_argument("-s", "--subject_id", dest = "subject_id", help = "Current subject id", required = True)
    parser.add_argument("-o", "--output_dir", dest = "out_dir", help = "Output directory base")
    parser.add_argument("-w", "--work_dir", dest = "work_dir", help = "Working directory base")
    args = parser.parse_args()

    wf = create_scndlvl_workflow(args)

    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    else:
        work_dir = os.getcwd()

    wf.config['execution']['crashdump_dir'] = '/scratch/madlab/crash/mandy_crash/model_RSA/lvl2'
    wf.base_dir = work_dir + '/' + args.subject_id
    wf.run(plugin='SLURM', plugin_args={'sbatch_args': ('-p investor --qos pq_madlab -N 1 -n 1'), 'overwrite': True})

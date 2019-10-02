#!/usr/bin/env python

"""
=======================================
GLM2_fMRI: Comprehensive -- Model GLM2
=======================================
Normal statistics workflow for UM GE 750 wmaze task data

- Comprehensive Model GLM2
   - General linear model (not LSS)
   - Use FSL ROI to recreate EPI data, removing last 3 volumes
     - Removed last 3 trials before EV creation
     - EV directory  --- /home/data/madlab/data/mri/wmaze/scanner_behav/WMAZE_001/model_GLM2

- python GLM2_norm_stats.py -s WMAZE_001
                       -o /home/data/madlab/data/mri/wmaze/norm_stats/model_GLM2
                       -w /scratch/madlab/crash/model_GLM2/norm_stats

"""

from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Merge, Function
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces import ants
from mattfeld_utility_workflows.fs_skullstrip_util import create_freesurfer_skullstrip_workflow
from nipype.interfaces.c3 import C3dAffineTool 


###############
## Functions ##
###############


get_len = lambda x: len(x)


def get_contrasts(data_inputs):
    import os
    infiles = [os.path.split(d[0])[1] for d in data_inputs]
    contrasts = [inf[7:].split('.nii')[0] for inf in infiles]
    print contrasts
    return contrasts


def get_substitutions(subject_id, cons):
    subs = [('_subject_id_{0}'.format(subject_id),'')]
    for i, con in enumerate(cons):
        subs.append(('_cope2targ{0}/cope'.format(i), 'cope{0}'.format(con)))
        subs.append(('_varcope2targ{0}/varcope'.format(i), 'varcope{0}'.format(con)))
    return subs


proj_dir = '/home/data/madlab/data/mri/wmaze'
fs_projdir = '/home/data/madlab/surfaces/wmaze'
work_dir = '/scratch/madlab/crash/model_GLM2/norm_stats'
sink_dir = '/home/data/madlab/data/mri/wmaze/norm_stats/model_GLM2/'


sids = ['WMAZE_001', 'WMAZE_002', 'WMAZE_004', 'WMAZE_005', 'WMAZE_006', 'WMAZE_007', 'WMAZE_008', 'WMAZE_009', 'WMAZE_010', 'WMAZE_012',  
        'WMAZE_017', 'WMAZE_018', 'WMAZE_019', 'WMAZE_020', 'WMAZE_021', 'WMAZE_022', 'WMAZE_023', 'WMAZE_024', 'WMAZE_026', 'WMAZE_027'] 


norm_stats_wf = Workflow("norm_stats_wf")
norm_stats_wf.base_dir = work_dir

#node to iterate through the subjects
subj_iterable = Node(IdentityInterface(fields = ['subject_id'], 
                                       mandatory_inputs = True), 
                     name = 'subj_interable')
subj_iterable.iterables = ('subject_id', sids)


# WORKFLOW: freesurfer skull stripping workflow
fs_skullstrip_wf = create_freesurfer_skullstrip_workflow()
fs_skullstrip_wf.inputs.inputspec.subjects_dir = fs_projdir
norm_stats_wf.connect(subj_iterable, 'subject_id', fs_skullstrip_wf, 'inputspec.subject_id')


info_norm = dict(copes = [['subject_id']], #dictionary keys for the datasource node
                 varcopes = [['subject_id']],
                 bbreg_xfm = [['subject_id', 'wmaze']],
                 ants_warp = [['subject_id', 'subject_id', 'output']],
                 mean_image = [['subject_id', 'wmaze']])


#node to grab the various data files for each subject
datasource_norm = Node(DataGrabber(infields = ['subject_id'],
                                   outfields = info_norm.keys()),
                       name = "datasource_norm")
datasource_norm.inputs.base_directory = proj_dir
                                             #copes and varcopes created in the 2nd level pipeling
datasource_norm.inputs.field_template = dict(copes = 'scndlvl/model_GLM2/%s/fixedfx/cope*.nii.gz',
                                             varcopes = 'scndlvl/model_GLM2/%s/fixedfx/varcope*.nii.gz',
                                             #BBReg transformation matrix created in preproc pipeline
                                             bbreg_xfm = 'preproc/%s/bbreg/_fs_register0/%s*.mat',
                                             #ANTS transformation matrix created in the antsreg_wf pipeline
                                             ants_warp = 'norm_anat/%s/anat2targ_xfm/_subject_id_%s/%s*.h5',
                                             #mean reference image created in preproc pipeline
                                             mean_image = 'preproc/%s/ref/%s*.nii.gz')
datasource_norm.inputs.ignore_exception = False
datasource_norm.inputs.raise_on_empty = True
datasource_norm.inputs.sort_filelist = True
datasource_norm.inputs.subject_id = sids
datasource_norm.inputs.template = '*'
datasource_norm.inputs.template_args = info_norm
norm_stats_wf.connect(subj_iterable, "subject_id", datasource_norm, "subject_id")


#node to convert fsl-style Affine registration into ANTS compatible itk format
convert2itk = Node(C3dAffineTool(), name = 'convert2itk')
convert2itk.inputs.fsl2ras = True #transform to ITK format (Boolean)
convert2itk.inputs.itk_transform = True #export ITK transform
norm_stats_wf.connect(datasource_norm, 'bbreg_xfm', convert2itk, 'transform_file') 
norm_stats_wf.connect(datasource_norm, 'mean_image', convert2itk, 'source_file')
norm_stats_wf.connect(fs_skullstrip_wf, 'outputspec.skullstripped_file', convert2itk, 'reference_file') 


#node to concatenate the ITK affine and ANTS transforms into a list
merge_xfm = Node(Merge(2), 
                 iterfield = ['in2'], 
                 name = 'merge_xfm')
norm_stats_wf.connect(convert2itk, 'itk_transform', merge_xfm, 'in2')
norm_stats_wf.connect(datasource_norm, 'ants_warp', merge_xfm, 'in1')


#MapNode to warp copes to target
cope2targ = MapNode(ants.ApplyTransforms(), 
                    iterfield = ['input_image'], 
                    name = 'cope2targ')
cope2targ.inputs.input_image_type = 3
cope2targ.inputs.interpolation = 'LanczosWindowedSinc'
cope2targ.inputs.invert_transform_flags = [False, False]
cope2targ.inputs.terminal_output = 'file'
cope2targ.inputs.args = '--float'
cope2targ.inputs.num_threads = 4
cope2targ.plugin_args = {'bsub_args': '-n%d' % 4}
#specify image whose space you are converting into
cope2targ.inputs.reference_image = '/home/data/madlab/data/mri/wmaze/wmaze_T1_template/T_wmaze_template.nii.gz'
norm_stats_wf.connect(datasource_norm, 'copes', cope2targ, 'input_image')
norm_stats_wf.connect(merge_xfm, 'out', cope2targ, 'transforms')


#MapNode to warp varcopes to target
varcope2targ = MapNode(ants.ApplyTransforms(), 
                       iterfield = ['input_image'], 
                       name = 'varcope2targ')
varcope2targ.inputs.input_image_type = 3
varcope2targ.inputs.interpolation = 'LanczosWindowedSinc'
varcope2targ.inputs.invert_transform_flags = [False, False]
varcope2targ.inputs.terminal_output = 'file'
varcope2targ.inputs.args = '--float'
varcope2targ.inputs.num_threads = 4
varcope2targ.plugin_args = {'bsub_args': '-n 4 -R "span[ptile=4]"'}
varcope2targ.inputs.reference_image = '/home/data/madlab/data/mri/wmaze/wmaze_T1_template/T_wmaze_template.nii.gz'
norm_stats_wf.connect(datasource_norm, 'varcopes', varcope2targ, 'input_image')
norm_stats_wf.connect(merge_xfm, 'out', varcope2targ, 'transforms')


#node to define the contrasts from the names of the copes
getcontrasts = Node(Function(input_names = ['data_inputs'],
                             output_names = ['contrasts'],
                             function = get_contrasts),
                    name = 'getcontrasts')
getcontrasts.inputs.ignore_exception = False
norm_stats_wf.connect(datasource_norm, 'copes', getcontrasts, 'data_inputs')


#function node to rename output files with something more meaningful
getsubs = Node(Function(input_names = ['subject_id', 'cons'],
                        output_names = ['subs'],
                        function = get_substitutions),
               name = 'getsubs')
getsubs.inputs.ignore_exception = False
norm_stats_wf.connect(subj_iterable, "subject_id", getsubs, 'subject_id')
norm_stats_wf.connect(getcontrasts, 'contrasts', getsubs, 'cons')


#node for group.sinker
norm_stats_sinker = Node(DataSink(infields = None), name = "norm_stats_sinker")
norm_stats_sinker.inputs._outputs = {}
norm_stats_sinker.inputs.base_directory = sink_dir
norm_stats_sinker.inputs.ignore_exception = False
norm_stats_sinker.inputs.parameterization = True
norm_stats_sinker.inputs.remove_dest_dir = False
norm_stats_wf.connect(subj_iterable, "subject_id", norm_stats_sinker, 'container')
norm_stats_wf.connect(cope2targ, "output_image", norm_stats_sinker, "norm_copes")
norm_stats_wf.connect(varcope2targ, "output_image", norm_stats_sinker, 'norm_varcopes')
norm_stats_wf.connect(getsubs, 'subs', norm_stats_sinker, "substitutions")

norm_stats_wf.config['execution']['crashdump_dir'] = work_dir
norm_stats_wf.run(plugin='SLURM', plugin_args={'sbatch_args': ('-p investor --qos pq_madlab -N 1 -n 1'), 'overwrite': True})

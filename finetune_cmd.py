
import os
import sys
import logging
from functools import partial

from demo_utils import download_model_folder
import argparse
import subprocess as sp

PROJECT_FOLDER = os.path.dirname(os.path.realpath(__file__))
PYTHON_EXE = 'python'
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, 'models')
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')

download_model = partial(download_model_folder, DATA_FOLDER=MODEL_FOLDER)
target_folder = download_model(model_size='medium', dataset='multiref', from_scratch=False)

logger = logging.getLogger(__name__)

#########################################################################
# Train !
#########################################################################
logger.info('Generating training CMD!')
logger.info('If there is any problem, please copy (modify) and run command below')
logger.info('#########################################################################')
train_cmd = 'LSP_train.py'
args = [
    '--model_name_or_path', target_folder,
    '--init_checkpoint', os.path.join(target_folder, 'medium_ft.pkl'),
    '--train_input_file', "./data/yesands_train.128len.db" ,  # file from last step
    '--eval_input_file', './data/yesands_valid.tsv',   # dummy test data
    '--output_dir', os.path.join(MODEL_FOLDER, 'output_model'),
    '--seed', '42',
    '--max_seq_length', '128',
    '--train_batch_size', '512',
    '--gradient_accumulation_steps', '8',
    '--eval_batch_size', '64',
    '--learning_rate', '1e-5',
    '--num_optim_steps', '10000',
    '--valid_step', '205',
    '--warmup_steps', '82',
    '--normalize_data', 'true',
    '--fp16', 'true',
    '--lr_schedule', 'noam',
    '--loss_scale', '0.0',
    '--no_token_id', 'true',
    '--pbar', 'true'
]

arg = ' '.join(args)
train_cmd = train_cmd + ' ' + arg
print(PYTHON_EXE + ' ' +train_cmd)
# logger.info('#########################################################################')
# with open('./finetune_output.log', 'wb') as f: 
#     process = sp.Popen([PYTHON_EXE] + train_cmd.split(' '), stdout=sp.PIPE, stderr=sp.STDOUT, cwd=PROJECT_FOLDER)
#     # for line in iter(process.stdout.readline, b''): 
#     #     sys.stdout.write(line.decode(sys.stdout.encoding)) 
#     #     f.write(line)
# logger.info('Done!\n')
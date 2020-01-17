import os
import sys

def generate_script(cmd:str, script_path:str, run_dir:str, abci_log_dir:str, ex_id:str, user:str, env:str):
    """
    """
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    os.makedirs(abci_log_dir, exist_ok=True)

    # name and logpath
    basename = os.path.basename(script_path)
    name, _ = os.path.splitext(basename)
    name = name+'_'+ex_id
    log_path = os.path.join(abci_log_dir, name+'.o')

    with open(script_path, mode='w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('#$ -l rt_F=1\n')
        f.write('#$ -l h_rt=72:00:00\n')
        f.write('#$ -j y\n')
        f.write('#$ -N {name}\n'.format(name=name))
        f.write('#$ -o {log_path}\n\n'.format(log_path=log_path))

        f.write('export PATH=/home/{user}/miniconda3/bin:\${PATH}\n'.format(user=user, PATH='{PATH}'))
        f.write('source activate {env}\n\n'.format(env=env))
        
        f.write('export PATH=/apps/gcc/7.3.0/bin:\${PATH}\n'.format(PATH='{PATH}'))
        f.write('export LD_LIBRARY_PATH=/apps/gcc/7.3.0/lib64:\${LD_LIBRARY_PATH}\n\n'.format(LD_LIBRARY_PATH='{LD_LIBRARY_PATH}'))
        
        f.write('source /etc/profile.d/modules.sh\n')
        f.write('module load singularity/2.6.1\n')
        f.write('module load cuda/10.0/10.0.130\n')
        f.write('module load cudnn/7.6/7.6.2\n\n')

        f.write('cd {run_dir}\n'.format(run_dir=run_dir))
        f.write(cmd)

if __name__ == '__main__':
    cmd = 'python'
    script_path = '../../logs/abci_script/script_test.sh'
    run_dir = '~'
    abci_log_dir = '../../logs/abci_log'
    ex_id = 'XXXXXXXXXXXXXX'
    user = 'aaa'
    env = 'env'

    generate_script(cmd, script_path, run_dir, abci_log_dir, ex_id, user, env)
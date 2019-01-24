import sys
import os,pdb,time

#######################################################################################
## relative to config
config_file='standard.yaml'
cluster=True
#cluster=False
#######################################################################################

ckpt_dir = '../ckpt'
main_dirs = [ckpt_dir]

### Get job name
with open('config/'+config_file,'r') as f:
    lines = f.readlines()
    job_name = lines[0][:-1].split(': ')[1][1:-1]

### process config
config=config_file.split('/')
if len(config)==1:
    filename = config[0]
else:
    dirs,filename = config[0:-1], config[-1]
filename = filename.split('.')[0]
dirs.append(filename)
dirs.append('log')

for each_main_dir in main_dirs:
    for ind, each_dir in enumerate(dirs):
        each_dir = '/'.join( dirs[0:ind+1])
        new_dir = os.path.join(each_main_dir, each_dir) 
        if not os.path.exists( new_dir ):
            os.mkdir( new_dir )
    print('create ckpt dir: ', each_main_dir, '/',  each_dir)



time.ctime()
cur_time = time.strftime('_%b%d_%H-%M-%S') 

#######################################################################################


### run script
if cluster:
    cmd = '''\
        LOG="''' + ckpt_dir + '/' + '/'.join(dirs) + '/' + job_name + cur_time + '/'  + '''log.txt";
        echo $LOG ;

        newdir="''' + ckpt_dir + '/' + '/'.join(dirs) + '/' + job_name + cur_time + '''"; 
        echo $newdir ;

        mkdir $newdir


        srun --mpi=pmi2 --partition=bj11part -n1 --gres=gpu:1 --ntasks-per-node=1 \
                --job-name=''' + job_name + ''' -w BJ-IDC1-10-10-11-''' + str(sys.argv[1]) + '''  \
        python run.py --mode train --cfg ''' +  config_file + ''' --time ''' + cur_time + '''$2>&1 | tee ${LOG}
    '''
else:
    cmd = '''\
        LOG="''' + ckpt_dir + '/' + '/'.join(dirs) + '/' + filename  + '''-`date +'%Y-%m-%d_%H-%M-%S'`_train" 
        echo $LOG ;
        python run.py --mode train --cfg ''' +  config_file + '''$2>&1 | tee ${LOG}
    '''

os.system(cmd)


















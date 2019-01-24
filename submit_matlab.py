import os
import random

############################################################################################
exp_dir = "new_github_adam_lr1e-4_Aug_0.3_2_Sep29_22-58-21"
#epoch = list(range(4,7))
epoch = list([16])

############################################################################################

print('epoch: ', epoch)

nodes = list(range(25,35)) + list(range(41,46))
while len(nodes) < len(epoch):
    nodes = nodes + nodes
    
selected_nodes = sorted( random.sample(nodes,len(epoch)) )
print('selected_nodes: ', selected_nodes)


for ind, cur_epoch in enumerate(epoch):
    print("----------------------", cur_epoch, "-------------------")
    job_name = 'T_' + str(cur_epoch) + "_" + exp_dir

    cmd = 'srun --partition=bj11part -n1 --job-name="' + job_name + '" -w BJ-IDC1-10-10-11-' + str(selected_nodes[ind]) + ' matlab -nodesktop -nosplash -r "eval_epoch(\'' + exp_dir + '\', \''+ str(cur_epoch) + '\')" 2>&1 &'

    print(cmd)
    os.system(cmd)
    
    


    




import torch
import torchvision.utils as vutils
from fastprogress import master_bar, progress_bar

import numpy as np
import time
from dataset.BSD500 import *
from models.HED import HED
import torch.nn as nn
import torch.nn.functional as F
from utils import AverageMeter
from torch.optim import lr_scheduler
import logging
from PIL import Image
#from logger import Logger
from tensorboardX import SummaryWriter
import scipy.io

from datetime import datetime
import pdb

#logger = Logger('./logs')


class HEDPipeline():
    def __init__(self, cfg):

        self.cfg = self.cfg_checker(cfg)
        self.root = '/'.join( ['../ckpt', self.cfg.path.split('.')[0]] )
        self.cur_lr = self.cfg.TRAIN.init_lr

        if self.cfg.TRAIN.disp_iter < self.cfg.TRAIN.update_iter:
            self.cfg.TRAIN.disp_iter = self.cfg.TRAIN.update_iter

        #current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.log_dir = os.path.join(self.root + '/log/', self.cfg.NAME + self.cfg.time)
        self.writer = SummaryWriter(self.log_dir)
        #self.writer = SummaryWriter()
        
        self.writer.add_text('cfg', str(self.cfg))
        

        ######################### Dataset ################################################3

        dataset = BSD500Dataset(self.cfg)
        self.data_loader = torch.utils.data.DataLoader(
                             dataset, 
                             batch_size=self.cfg.TRAIN.batchsize,
                             shuffle=True,
                             num_workers=self.cfg.TRAIN.num_workers )

        dataset_test = BSD500DatasetTest(self.cfg)
        #dataset_test = BSD500Dataset(self.cfg)
        self.data_test_loader = torch.utils.data.DataLoader(
                             dataset_test, 
                             batch_size=1,
                             shuffle=False,
                             num_workers=self.cfg.TRAIN.num_workers )
        

        ######################### Model ################################################3


        self.model = HED(self.cfg, self.writer) 
        self.model = self.model.cuda()
        
        ### loss function
        if self.cfg.MODEL.loss_func_logits:
            self.loss_function = F.binary_cross_entropy_with_logits
        else:
            self.loss_function = F.binary_cross_entropy
        

        ######################### Optimizer ################################################3

        init_lr = self.cfg.TRAIN.init_lr
        self.lr_cof = self.cfg.TRAIN.lr_cof
        
        if self.cfg.TRAIN.update_method=='SGD':
            params_lr_1 = list(self.model.conv1.parameters())  \
                            + list(self.model.conv2.parameters())  \
                            + list(self.model.conv3.parameters())  \
                            + list(self.model.conv4.parameters())
            params_lr_100 = self.model.conv5.parameters()
            params_lr_001 = list(self.model.dsn1.parameters())  \
                            + list(self.model.dsn2.parameters())  \
                            + list(self.model.dsn3.parameters())  \
                            + list(self.model.dsn4.parameters())  \
                            + list(self.model.dsn5.parameters()) 
            params_lr_0001 = self.model.new_score_weighting.parameters()


            optim_paras_list = [    {'params': params_lr_1 },
                                    {'params': params_lr_100,  'lr': init_lr * self.lr_cof[1] },
                                    {'params': params_lr_001,  'lr': init_lr * self.lr_cof[2] },
                                    {'params': params_lr_0001, 'lr': init_lr * self.lr_cof[3] }
                               ]

            self.optim = torch.optim.SGD( optim_paras_list, lr = init_lr, momentum=0.9, weight_decay=1e-4)

        elif self.cfg.TRAIN.update_method in ['Adam', 'Adam-sgd']:
            self.optim = torch.optim.Adam(self.model.parameters(), lr = init_lr)

        elif self.cfg.TRAIN.update_method=='Adam_paper':
            params_lr_1 = list(self.model.conv1.parameters())  \
                            + list(self.model.conv2.parameters())  \
                            + list(self.model.conv3.parameters())  \
                            + list(self.model.conv4.parameters())
            params_lr_100 = self.model.conv5.parameters()
            params_lr_001 = list(self.model.dsn1.parameters())  \
                            + list(self.model.dsn2.parameters())  \
                            + list(self.model.dsn3.parameters())  \
                            + list(self.model.dsn4.parameters())  \
                            + list(self.model.dsn5.parameters()) 
            params_lr_0001 = self.model.new_score_weighting.parameters()


            #self.lr_cof = [1, 100, 0.01, 0.001]
            optim_paras_list = [    {'params': params_lr_1 },
                                    {'params': params_lr_100,  'lr': init_lr * self.lr_cof[1] },
                                    {'params': params_lr_001,  'lr': init_lr * self.lr_cof[2] },
                                    {'params': params_lr_0001, 'lr': init_lr * self.lr_cof[3] }
                               ]

            self.optim = torch.optim.Adam( optim_paras_list, lr = init_lr, weight_decay=1e-4)

        elif self.cfg.TRAIN.update_method=='Adam_except_vgg1-4':
            optim_paras_list = params_lr_100 + params_lr_001 + params_lr_0001
            self.optim = torch.optim.Adam( optim_paras_list, lr = init_lr )


        self.optim.zero_grad()




    def train(self):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()


        self.final_loss = 0
        tic = time.time()
        for cur_epoch in range(self.cfg.TRAIN.nepoch):
            
            for ind, (data,target) in enumerate(self.data_loader):
                cur_iter = cur_epoch * len(self.data_loader) + ind + 1

                data, target = data.cuda(), target.cuda()
                data_time.update(time.time() - tic)

                dsn1, dsn2, dsn3, dsn4, dsn5, dsn6 = self.model( data )  

                if not self.cfg.MODEL.loss_func_logits:
                    dsn1 = torch.sigmoid(dsn1)
                    dsn2 = torch.sigmoid(dsn2)
                    dsn3 = torch.sigmoid(dsn3)
                    dsn4 = torch.sigmoid(dsn4)
                    dsn5 = torch.sigmoid(dsn5)
                    dsn6 = torch.sigmoid(dsn6)
                
                
                ############################## Compute Loss ########################################

                if self.cfg.MODEL.loss_balance_weight:
                    cur_weight = self.edge_weight(target)
                    self.writer.add_histogram('weight: ', cur_weight.clone().cpu().data.numpy(), cur_epoch)
                else:
                    cur_weight = None
                cur_reduce = self.cfg.MODEL.loss_reduce
                self.loss1 = self.loss_function(dsn1.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                self.loss2 = self.loss_function(dsn2.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                self.loss3 = self.loss_function(dsn3.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                self.loss4 = self.loss_function(dsn4.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                self.loss5 = self.loss_function(dsn5.float(), target.float(), weight=cur_weight, reduce=cur_reduce)
                self.loss6 = self.loss_function(dsn6.float(), target.float(), weight=cur_weight, reduce=cur_reduce)


                loss_weight_list = self.cfg.MODEL.loss_weight_list
                assert( len(loss_weight_list)==6, "len(loss_weight) should be 6" )
                loss = [ self.loss1, self.loss2, self.loss3, self.loss4, self.loss5, self.loss6]
                #self.final_loss += sum( [x*y for x,y in zip(loss_weight_list, loss)] ) 
                self.loss = sum( [x*y for x,y in zip(loss_weight_list, loss)] ) 
                self.loss = self.loss / self.cfg.TRAIN.update_iter
                self.final_loss += self.loss


                if cur_reduce:
                    if np.isnan(float(self.loss.item())):
                         raise ValueError('loss is nan while training')

                self.loss.backward()

                ############################## Update Gradients ########################################

                if (cur_iter % self.cfg.TRAIN.update_iter)==0:
                    self.optim.step()
                    self.optim.zero_grad()

                    self.final_loss_show = self.final_loss 
                    self.final_loss = 0
                

                ### lr update 
                if self.cfg.TRAIN.update_method=='SGD':
                    self.cur_lr = self.step_learning_rate(self.optim, self.cur_lr, self.cfg.TRAIN.lr_list, (cur_epoch+1) )
                    #cur_lr = self.poly_learning_rate( self.optim, self.cfg.TRAIN.init_lr, \
                    #                            cur_iter, self.max_iter, power=0.9)
                

                batch_time.update(time.time() - tic)
                tic = time.time()

                #print( len(self.data_loader) )
                if ((ind+1) % self.cfg.TRAIN.disp_iter)==0:
                    print_str  = 'Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, lr: {:.11f}, \n \
                                  final_loss: {:.6f}, loss1:{:.6f}, loss2:{:.6f}, loss3:{:.6f}, \
                                  loss4:{:.6f}, loss5:{:.6f}, loss6:{:.6f}\n '.format(cur_epoch, ind, \
                                  len(self.data_loader), batch_time.average(), data_time.average(), \
                                  self.cur_lr, self.final_loss_show, self.loss1, self.loss2,  \
                                  self.loss3, self.loss4, self.loss5, self.loss6)

                    print(print_str)

                    ######## show loss
                    self.writer.add_scalar('loss/loss1', self.loss1.item(), cur_iter)
                    self.writer.add_scalar('loss/loss2', self.loss2.item(), cur_iter)
                    self.writer.add_scalar('loss/loss3', self.loss3.item(), cur_iter)
                    self.writer.add_scalar('loss/loss4', self.loss4.item(), cur_iter)
                    self.writer.add_scalar('loss/loss5', self.loss5.item(), cur_iter)
                    self.writer.add_scalar('loss/loss6', self.loss6.item(), cur_iter)
                    self.writer.add_scalar('final_loss', self.final_loss_show.item(), cur_iter)

                    self.tensorboard_summary(cur_iter) ### show loss and weights

                

            ### clean gradient after one epoch
                
            ### Test 
            if ((cur_epoch+1) % self.cfg.TRAIN.test_iter) == 0:
                self.test(cur_epoch)

            self.writer.add_text('epoch', 'cur_epoch is ' + str(cur_epoch), cur_epoch)
            self.writer.add_text('loss', str(print_str))

            ### save model
            if ((cur_epoch+1) % self.cfg.TRAIN.save_iter) == 0:
                print('=======> saving model')
                suffix_latest = 'epoch_{}.pth'.format(cur_epoch)
                model_save_path = os.path.join(self.log_dir, suffix_latest)
                torch.save( self.model.state_dict(), model_save_path)
                
        self.writer.close()



    def tensorboard_summary(self, cur_epoch):

        ######## weight
        print('weight: ')
        print(self.model.new_score_weighting.weight.shape)
        print(self.model.new_score_weighting.weight)
        print(self.model.new_score_weighting.bias)
        self.writer.add_histogram('new_score_weighting/weight: ', self.model.new_score_weighting.weight.clone().cpu().data.numpy(), cur_epoch)
        self.writer.add_histogram('new_score_weighting/bias: ', self.model.new_score_weighting.bias.clone().cpu().data.numpy(), cur_epoch)

        print('weight grad: ')
        print(self.model.new_score_weighting.weight.grad)
        print(self.model.new_score_weighting.bias.grad)
        #pdb.set_trace()


        ######## conv5
        conv5_index = -3 if self.cfg.MODEL.backbone=='vgg16_bn' else -2
        self.writer.add_histogram('conv5/a_weight: ', self.model.conv5[conv5_index].weight.clone().cpu().data.numpy(), cur_epoch)
        self.writer.add_histogram('conv5/a_bias: ', self.model.conv5[conv5_index].bias.clone().cpu().data.numpy(), cur_epoch)
        self.writer.add_histogram('conv5/b_weight_grad: ', self.model.conv5[conv5_index].weight.grad.clone().cpu().data.numpy(), cur_epoch)

        self.writer.add_histogram('conv5/b_bias_grad: ', self.model.conv5[conv5_index].bias.grad.clone().cpu().data.numpy(), cur_epoch)
        self.writer.add_histogram('conv5/c_output: ', self.model.conv5_output.clone().cpu().data.numpy(), cur_epoch)



    def edge_weight(self, target):

        h, w = target.shape[2:]
        #num_nonzero = torch.nonzero(target).shape[0]

        #weight_p = num_nonzero / (h*w)
        weight_p = torch.sum(target) / (h*w)
        weight_n = 1 - weight_p

        res = target.clone()
        res[target==0] = weight_p
        res[target>0] = weight_n
        assert( (weight_p + weight_n)==1, "weight_p + weight_n !=1")
        #print(res, type(res))
    
        return res

    def edge_pos_weight(self, target):

        h, w = target.shape[2:]
        #num_nonzero = torch.nonzero(target).shape[0]

        #weight_p = num_nonzero / (h*w)
        weight_p = torch.sum(target) / (h*w)
        weight_n = 1 - weight_p

        pos_weight = weight_n / weight_p

        res = target.clone()
        res = (1 - weight_n)
        #res[:,:,:,:] = 1


        return res, pos_weight


    def poly_learning_rate(self, optimizer, base_lr, curr_iter, max_iter, power=0.9):

        """poly learning rate policy"""
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** power

        assert( len(optimizer.param_groups)==4, 'num of len(optimizer.param_groups)' )
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr * self.lr_cof[index]

        return lr

    def step_learning_rate(self, optimizer, lr, lr_list, cur_epoch ):

        if cur_epoch not in lr_list:
            return lr
         
        lr = lr / 10;
        #assert( len(optimizer.param_groups)==5 'num of len(optimizer.param_groups)' )
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr  * self.lr_cof[index]
            #param_group['lr'] = lr

        self.writer.add_text('LR', 'lr = ' + str(lr) + ' at step: ' + str(cur_epoch) )

        return lr






    def test(self, cur_epoch):
        self.model.eval()

        print(' ---------Test, cur_epoch: ', cur_epoch)
        ### makedirs
        #result_dir = 'result_epoch' + str(cur_epoch)
        #self.makedir( os.path.join(self.root, result_dir) )
        
        #for ind in range(1,7):
            #self.makedir( os.path.join(self.root, result_dir, 'dsn'+str(ind)) )

        def save_img(dsn, result_dir, index):
            if self.cfg.MODEL.loss_func_logits:
                dsn_final = torch.sigmoid(dsn)
            else:
                dsn_final = dsn

            
            dsn_final_np = np.array( dsn_final.detach().cpu().numpy() )
            dsn_final_np = dsn_final_np[0,0,:,:]
            dsn_img = Image.fromarray( np.uint8(dsn_final_np*255), 'L')

            save_path = os.path.join( self.root, result_dir, 'dsn'+str(index) )
            dsn_img.save( os.path.join(save_path, img_filename[0]+'.png') )


        ### Forward
        for ind, item in enumerate(self.data_test_loader):
            (data, img_filename) = item
            #(data, target) = item
            data = data.cuda()

            #img_filename = '100075.png'
            print(img_filename)
            dsn1, dsn2, dsn3, dsn4, dsn5, dsn6 = self.model( data )  

            #save_img(dsn1, result_dir, 1)  
            #save_img(dsn2, result_dir, 2)  
            #save_img(dsn3, result_dir, 3)  
            #save_img(dsn4, result_dir, 4)  
            #save_img(dsn5, result_dir, 5)  
            #save_img(dsn6, result_dir, 6)  

            #pdb.set_trace()
            input_show = vutils.make_grid(data, normalize=True, scale_each=True)
            if self.cfg.MODEL.loss_func_logits:
                dsn1 = torch.sigmoid(dsn1)
                dsn2 = torch.sigmoid(dsn2)
                dsn3 = torch.sigmoid(dsn3)
                dsn4 = torch.sigmoid(dsn4)
                dsn5 = torch.sigmoid(dsn5)
                dsn6 = torch.sigmoid(dsn6)
            
            dsn7 = (dsn1 + dsn2 + dsn3 + dsn4 + dsn5) / 5.0
            results = [dsn1, dsn2, dsn3, dsn4, dsn5, dsn6, dsn7]
            self.save_mat(results, img_filename,  cur_epoch) 
            

            dsn1_show = vutils.make_grid(dsn1.data, normalize=True, scale_each=True)
            dsn2_show = vutils.make_grid(dsn2.data, normalize=True, scale_each=True)
            dsn3_show = vutils.make_grid(dsn3.data, normalize=True, scale_each=True)
            dsn4_show = vutils.make_grid(dsn4.data, normalize=True, scale_each=True)
            dsn5_show = vutils.make_grid(dsn5.data, normalize=True, scale_each=True)
            dsn6_show = vutils.make_grid(dsn6.data, normalize=False, scale_each=True)
            #target_show = vutils.make_grid(target.data, normalize=True, scale_each=True)

            self.writer.add_image(img_filename[0]+'/aa_input', input_show, cur_epoch)
            self.writer.add_image(img_filename[0]+'/ab_dsn6', dsn6_show, cur_epoch)
            #self.writer.add_image(img_filename[0]+'/ac_target', target_show, cur_epoch)
            self.writer.add_image(img_filename[0]+'/dsn1', dsn1_show, cur_epoch)
            self.writer.add_image(img_filename[0]+'/dsn2', dsn2_show, cur_epoch)
            self.writer.add_image(img_filename[0]+'/dsn3', dsn3_show, cur_epoch)
            self.writer.add_image(img_filename[0]+'/dsn4', dsn4_show, cur_epoch)
            self.writer.add_image(img_filename[0]+'/dsn5', dsn5_show, cur_epoch)


            #self.writer.a'dd_image(img_filename[0]+'/input', x_show, cur_epoch)
            #self.writer.add_image('dsn/dsn1', dsn1_show, cur_epoch)
            #self.writer.add_image('dsn/dsn2', dsn2_show, cur_epoch)
            #self.writer.add_image('dsn6', dsn6_show, cur_epoch)

        self.model.train()


    def save_mat(self, results, img_filename, cur_epoch):

        if cur_epoch==0:
            self.makedir( os.path.join(self.log_dir, 'results_mat' ) )

        self.makedir( os.path.join(self.log_dir, 'results_mat', str(cur_epoch) ) )
        for dsn_ind in range(1,8):
            self.makedir( os.path.join(self.log_dir, 'results_mat', str(cur_epoch), 'dsn'+str(dsn_ind)) )

        #new_one = (results[0] + results[1] + results[2] + results[3] + results[4]) / 5
        #results.append( new_one )

        for ind, each_dsn in enumerate(results):
            each_dsn = each_dsn.data.cpu().numpy()
            each_dsn = np.squeeze( each_dsn )
            
            #scipy.io.savemat(os.path.join(self.log_dir, img_filename),dict({'edge': each_dsn / np.max(each_dsn)}),appendmat=True)

            #print( type(each_dsn) )
            save_path = os.path.join(self.log_dir, 'results_mat', str(cur_epoch),  'dsn'+str(ind+1), img_filename[0])
            if self.cfg.SAVE.MAT.normalize:
                each_dsn = each_dsn / np.max(each_dsn)
            scipy.io.savemat(save_path, dict({'edge': each_dsn}))


    def makedir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def cfg_checker(self, cfg):
        return cfg

        









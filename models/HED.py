import math
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np


import pdb


class HED(nn.Module):
    def __init__(self, cfg, writer):
        super(HED, self).__init__()
        
        self.cfg = cfg
        self.writer = writer


        ############################ Model ###################################
        self.first_padding = nn.ReflectionPad2d(self.cfg.MODEL.first_pad)

        ### vgg16
        backbone_mode = self.cfg.MODEL.backbone
        pretrained = self.cfg.MODEL.pretrained 
        if backbone_mode=='vgg16':
            vgg16 = models.vgg16(pretrained=pretrained).cuda()
        elif self.cfg.MODEL.backbone=='vgg16_bn':
            vgg16 = models.vgg16_bn(pretrained=pretrained).cuda()

        self.conv1 = self.extract_layer(vgg16, backbone_mode, 1)
        self.conv2 = self.extract_layer(vgg16, backbone_mode, 2)
        self.conv3 = self.extract_layer(vgg16, backbone_mode, 3)
        self.conv4 = self.extract_layer(vgg16, backbone_mode, 4)
        self.conv5 = self.extract_layer(vgg16, backbone_mode, 5)

        print(self.conv5)
        

        ### other layers
        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(512, 1, 1)
        self.new_score_weighting = nn.Conv2d(5, 1, 1)

        self.dsn1_bn = nn.BatchNorm2d(1)
        self.dsn2_bn = nn.BatchNorm2d(1)
        self.dsn3_bn = nn.BatchNorm2d(1)
        self.dsn4_bn = nn.BatchNorm2d(1)
        self.dsn5_bn = nn.BatchNorm2d(1)

        if self.cfg.MODEL.upsample_layer == 'deconv':
            self.dsn2_up = nn.ConvTranspose2d(1, 1, 4, stride=2)
            self.dsn3_up = nn.ConvTranspose2d(1, 1, 8, stride=4)
            self.dsn4_up = nn.ConvTranspose2d(1, 1, 16, stride=8)
            self.dsn5_up = nn.ConvTranspose2d(1, 1, 32, stride=16)
        
        #self.other_layers = [self.dsn1, self.dsn2, self.dsn3, self.dsn4, self.dsn5, 
        #                       self.nInitialization ew_score_weighting ]
        self.other_layers = [self.dsn1, self.dsn2, self.dsn3, self.dsn4, self.dsn5 ]

        if self.cfg.MODEL.upsample_layer == 'deconv':
            self.other_layers += [ self.dsn2_up, self.dsn3_up, self.dsn4_up, self.dsn5_up ]
        
        if not self.cfg.MODEL.pretrained:
            self.other_layers += [ self.conv1, self.conv2, self.conv3, self.conv4, self.conv5 ]


        ############################ Layer Initialization ###################################
        if self.cfg.MODEL.upsample_layer == 'github':
            def weights_init(m):
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.01)
        else:
            def weights_init(m):
                if isinstance(m, nn.Conv2d):
                    if self.cfg.MODEL.init_mode=='Gaussian':
                        m.weight.data.normal_(0, 0.1)
                        m.bias.data.normal_(0, 0.01)
                    elif self.cfg.MODEL.init_mode=='xavier':
                        nn.init.xavier_normal_(m.weight.data) 
                        m.bias.data.fill_(0)
                elif isinstance(m, nn.ConvTranspose2d):
                    #nn.init.xavier_normal_(m.weight.data)
                    #m.bias.data.fill_(0)
                    m.weight.data.normal_(0, 0.2)
                    #m.bias.data.normal_(0, 0.01)
                    m.bias.data.fill_(0)

        for each_layer in self.other_layers:
            each_layer.apply( weights_init )

        self.new_score_weighting.weight.data.fill_(0.2) 
        self.new_score_weighting.bias.data.fill_(0)


    def forward(self, x):
        h, w = x.shape[2:]
        
        # backbone
        x = self.first_padding(x) 

        self.conv1_output = self.conv1(x)
        self.conv2_output = self.conv2(self.conv1_output)
        self.conv3_output = self.conv3(self.conv2_output)
        self.conv4_output = self.conv4(self.conv3_output)
        self.conv5_output = self.conv5(self.conv4_output)

        
        ############################# Side Connection
        ### dsn1
        dsn1 = self.dsn1(self.conv1_output) 
        dsn1_final = self.crop_layer(dsn1, h, w)
        #dsn1_final_bn = self.dsn1_bn(dsn1_final)
        #print('dsn1 ', dsn1_final.shape)

        ### dsn2
        dsn2 = self.dsn2(self.conv2_output)
        if self.cfg.MODEL.upsample_layer == 'deconv':
            dsn2_up = self.dsn2_up(dsn2)
        elif self.cfg.MODEL.upsample_layer == 'bilinear':
            h2,w2 = dsn2.shape[2:]
            if self.cfg.MODEL.interpolate_mode=='nearest':
                dsn2_up = F.interpolate(dsn2, size=(2*(h2+1), 2*(w2+1)), mode=self.cfg.MODEL.interpolate_mode)
            elif self.cfg.MODEL.interpolate_mode=='bilinear':
                dsn2_up = F.interpolate(dsn2, size=(2*(h2+1), 2*(w2+1)), mode=self.cfg.MODEL.interpolate_mode, align_corners=True)
        elif self.cfg.MODEL.upsample_layer == 'github':
            weight_deconv2 =  self.make_bilinear_weights(4, 1).cuda() 
            dsn2_up = torch.nn.functional.conv_transpose2d(dsn2, weight_deconv2, stride=2)
        dsn2_final = self.crop_layer(dsn2_up, h, w)
        #dsn2_final_bn = self.dsn2_bn(dsn2_final)
        #print('dsn2 ', dsn2_final.shape)

        ### dsn3
        dsn3 = self.dsn3(self.conv3_output)
        if self.cfg.MODEL.upsample_layer == 'deconv':
            dsn3_up = self.dsn3_up(dsn3)
        elif self.cfg.MODEL.upsample_layer == 'bilinear':
            h3,w3 = dsn3.shape[2:]
            if self.cfg.MODEL.interpolate_mode=='nearest':
                dsn3_up = F.interpolate(dsn3, size=(4*(h3+1), 4*(w3+1)), mode=self.cfg.MODEL.interpolate_mode)
            elif self.cfg.MODEL.interpolate_mode=='bilinear':
                dsn3_up = F.interpolate(dsn3, size=(4*(h3+1), 4*(w3+1)), mode=self.cfg.MODEL.interpolate_mode, align_corners=True)
        elif self.cfg.MODEL.upsample_layer == 'github':
            weight_deconv3 =  self.make_bilinear_weights(8, 1).cuda() 
            dsn3_up = torch.nn.functional.conv_transpose2d(dsn3, weight_deconv3, stride=4)
        dsn3_final = self.crop_layer(dsn3_up, h, w)
        #dsn3_final_bn = self.dsn3_bn(dsn3_final)
        #dsn4_final_bn = self.dsn4_bn(dsn4_final)
        #print('dsn3 ', dsn3_final.shape)

        ### dsn4
        dsn4 = self.dsn4(self.conv4_output)
        if self.cfg.MODEL.upsample_layer == 'deconv':
            dsn4_up = self.dsn4_up(dsn4)
        elif self.cfg.MODEL.upsample_layer == 'bilinear':
            h4,w4 = dsn4.shape[2:]
            if self.cfg.MODEL.interpolate_mode=='nearest':
                dsn4_up = F.interpolate(dsn4, size=(8*(h4+1),8*(w4+1)), mode=self.cfg.MODEL.interpolate_mode)
            elif self.cfg.MODEL.interpolate_mode=='bilinear':
                dsn4_up = F.interpolate(dsn4, size=(8*(h4+1),8*(w4+1)), mode=self.cfg.MODEL.interpolate_mode, align_corners=True)
        elif self.cfg.MODEL.upsample_layer == 'github':
            weight_deconv4 =  self.make_bilinear_weights(16, 1).cuda() 
            dsn4_up = torch.nn.functional.conv_transpose2d(dsn4, weight_deconv4, stride=8)
        dsn4_final = self.crop_layer(dsn4_up, h, w)
        #dsn4_final_bn = self.dsn4_bn(dsn4_final)
        #print('dsn4 ', dsn4_final.shape)

        ### dsn5
        dsn5 = self.dsn5(self.conv5_output)
        if self.cfg.MODEL.upsample_layer == 'deconv':
            dsn5_up = self.dsn5_up(dsn5)
        elif self.cfg.MODEL.upsample_layer == 'bilinear':
            h5,w5 = dsn5.shape[2:]
            if self.cfg.MODEL.interpolate_mode=='nearest':
                dsn5_up = F.interpolate(dsn5, size=(16*(h5+1), 16*(w5+1)), mode=self.cfg.MODEL.interpolate_mode)
            elif self.cfg.MODEL.interpolate_mode=='bilinear':
                dsn5_up = F.interpolate(dsn5, size=(16*(h5+1), 16*(w5+1)), mode=self.cfg.MODEL.interpolate_mode, align_corners=True)
        elif self.cfg.MODEL.upsample_layer == 'github':
            weight_deconv5 =  self.make_bilinear_weights(32, 1).cuda() 
            dsn5_up = torch.nn.functional.conv_transpose2d(dsn5, weight_deconv5, stride=16)
        dsn5_final = self.crop_layer(dsn5_up, h, w)
        #dsn5_final_bn = self.dsn5_bn(dsn5_final)
        #print('dsn5 ', dsn5_final.shape)

        concat = torch.cat( (dsn1_final, dsn2_final, dsn3_final, dsn4_final, dsn5_final), 1 )
        #concat = torch.cat( (dsn1_final_bn, dsn2_final_bn, dsn3_final_bn, dsn4_final_bn, dsn5_final_bn), 1 )
        dsn6_final = self.new_score_weighting( concat )
        
        return dsn1_final, dsn2_final, dsn3_final, dsn4_final, dsn5_final, dsn6_final


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(HED, self).train(mode)

        contain_bn_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]

        if self.cfg.MODEL.freeze_bn:
            print("----Freezing Mean/Var of BatchNorm2D.")

            for each_block in contain_bn_layers:
                for m in each_block.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        print("---- in bn layer")
                        print(m)
                        m.eval()

                        if self.cfg.MODEL.freeze_bn_affine:
                            print("---- Freezing Weight/Bias of BatchNorm2D.")
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
         

    ################################################## 
    ### helper functions  
    ################################################## 
    
#    def init_weights(self):
#        ### initialize weights
#        def weights_init(m):
#            if isinstance(m, nn.Conv2d):
#                nn.init.xavier_normal_(m.weight.data)
#                m.bias.data.fill_(0)
#            elif isinstance(m, nn.ConvTranspose2d):
#                nn.init.xavier_normal_(m.weight.data)
#                m.bias.data.fill_(0)
#
#        #for each_layer in self.other_layers:
#        #    each_layer.apply( weights_init )
#
#        #self.new_score_weighting.weight.data.fill_(0.2)
#        #self.new_score_weighting.bias.data.fill_(0)
#        
#        self.model.apply( weights_init )
#


    def extract_layer(self, model, backbone_mode, ind):
        #pdb.set_trace()
        if backbone_mode=='vgg16':
            index_dict = {
                1: (0,4), 
                2: (4,9), 
                3: (9,16), 
                4: (16,23),
                5: (23,30) }
        elif backbone_mode=='vgg16_bn':
            index_dict = {
                1: (0,6), 
                2: (6,13), 
                3: (13,23), 
                4: (23,33),
                5: (33,43) }

        start, end = index_dict[ind]
        modified_model = nn.Sequential(*list(model.features.children())[start:end])
        return modified_model


    def make_bilinear_weights(self, size, num_channels):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        # print(filt)
        filt = torch.from_numpy(filt)
        w = torch.zeros(num_channels, num_channels, size, size)
        w.requires_grad = False
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    w[i, j] = filt
        return w


    def crop_layer(self, x, h, w):
        input_h, input_w = x.shape[2:]
        ref_h, ref_w = h, w
        
        assert( input_h > ref_h, "input_h should be larger than ref_h")
        assert( input_w > ref_w, "input_w should be larger than ref_w")
        
        #h_start = math.floor( (input_h - ref_h) / 2 )
        #w_start = math.floor( (input_w - ref_w) / 2 )
        h_start = int(round( (input_h - ref_h) / 2 ))
        w_start = int(round( (input_w - ref_w) / 2 ))
        x_new = x[:, :, h_start:h_start+ref_h, w_start:w_start+ref_w] 

        return x_new

    



        




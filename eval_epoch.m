function eval_epoch(exp_dir, epoch) 
    addpath('/mnt/lustre/wuchongruo/projects/my_hed/matlab_code/edges/')
    addpath('/mnt/lustre/wuchongruo/projects/my_hed/matlab_code/edges/my')
    addpath('/mnt/lustre/wuchongruo/projects/my_hed/matlab_code/edges/models')
    addpath('/mnt/lustre/wuchongruo/projects/my_hed/matlab_code/toolbox/')
    addpath('/mnt/lustre/wuchongruo/projects/my_hed/matlab_code/toolbox/channels')
    addpath('/mnt/lustre/wuchongruo/projects/my_hed/matlab_code/toolbox/classify')
    addpath('/mnt/lustre/wuchongruo/projects/my_hed/matlab_code/toolbox/detector')
    addpath('/mnt/lustre/wuchongruo/projects/my_hed/matlab_code/toolbox/filters')
    addpath('/mnt/lustre/wuchongruo/projects/my_hed/matlab_code/toolbox/images')
    addpath('/mnt/lustre/wuchongruo/projects/my_hed/matlab_code/toolbox/matlab')
    addpath('/mnt/lustre/wuchongruo/projects/my_hed/matlab_code/toolbox/videos')
    savepath;
    root_root = '/mnt/lustre/wuchongruo/projects/my_hed/ckpt/BSD500/standard/log';



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %exp_dir = 'adam_lr2e-4_nobn_ui10_Sep17_02-30-35'
    %epoch = 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    root = fullfile(root_root, exp_dir, 'results_mat',  num2str(epoch))
    save_root = fullfile(root_root, exp_dir, 'results_mat',  [num2str(epoch), '_nms']);
    mkdir(save_root);

    dsn_folders = dir(root);
    dsn_folders = dsn_folders(3:end, :);

    for dsn_folder_ind = 1:size(dsn_folders,1)

        cur_dsn_folder = dsn_folders(dsn_folder_ind).name

        cur_save_root = fullfile(save_root, cur_dsn_folder);
        mkdir(cur_save_root);

        if dsn_folder_ind==6 || dsn_folder_ind==7
            cur_save_root_mat = fullfile(save_root, [cur_dsn_folder, '_mat' ]);
            mkdir(cur_save_root_mat);
        end
        
        files = dir( fullfile(root, cur_dsn_folder) );
        files = files(3:end, :);
        
        for image_ind = 1:size(files,1)
            file_name = fullfile(root, cur_dsn_folder, files(image_ind).name);
            matObj = matfile(file_name);
            varlist = who(matObj);
            x = matObj.(char(varlist));

            E=convTri(single(x),1);
            [Ox,Oy]=gradient2(convTri(E,4));
            [Oxx,~]=gradient2(Ox); [Oxy,Oyy]=gradient2(Oy);
            O=mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
            E=edgesNmsMex(E,O,1,5,1.01,4);

            [~, name, ~] = fileparts(files(image_ind).name);
            save_path = fullfile( save_root, cur_dsn_folder, [name,'.png'] );
            imwrite( uint8(E*255), save_path );
            
            if dsn_folder_ind==6 || dsn_folder_ind==7
                save_path_mat = fullfile( cur_save_root_mat, [name,'.mat'] );
                save(save_path_mat, 'E');
            end
        end
    end
    
    %%%% merge two dsn
    dsn6_dir = fullfile(save_root, 'dsn6_mat');
    dsn7_dir = fullfile(save_root, 'dsn7_mat');
    dsn_merge_dir = fullfile(save_root, 'dsn_merge');
    mkdir(dsn_merge_dir);

    files = dir( dsn6_dir );
    files = files(3:end, :);
    for image_ind = 1:size(files,1)
        file_name6 = fullfile(dsn6_dir, files(image_ind).name);
        file_name7 = fullfile(dsn7_dir, files(image_ind).name);

        a = load(file_name6);
        dsn_fuse = a.E;

        b = load(file_name7);
        dsn12345 = b.E;
        
        merged = (double(dsn_fuse) + double(dsn12345)) / 2.0;

        [~, name, ~] = fileparts(files(image_ind).name);
        save_path = fullfile( dsn_merge_dir, [name,'.png'] );
        imwrite( uint8(merged*255), save_path );
    end
    


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    gtDir = '/mnt/lustre/wuchongruo/projects/my_hed/data/BSDS500/BSDS500/data/groundTruth/test';

    resDir = fullfile(save_root, 'dsn6')
    [ODS_F_fuse, ~, ~, ~, OIS_F_fuse, ~, ~, AP_fuse, ~] = edgesEvalDir('resDir',resDir,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.0075)

    resDir = fullfile(save_root, 'dsn_merge')
    [ODS_F_merge, ~, ~, ~, OIS_F_merge, ~, ~, AP_merge, ~] = edgesEvalDir('resDir',resDir,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.0075)

    result_txt_path = fullfile(root_root, exp_dir, 'result.txt')
    if exist(result_txt_path)==2
        f = fopen(result_txt_path, 'a');
    else
        f = fopen(result_txt_path, 'w');
    end

    fprintf(f, '%d: %0.5f %0.5f %0.5f %0.5f %0.5f %0.5f\n', str2num(epoch), ODS_F_fuse, OIS_F_fuse, AP_fuse, ODS_F_merge, OIS_F_merge, AP_merge);
    fprintf('cur_test: %s %d\n', exp_dir, str2num(epoch))
    
    exit
end






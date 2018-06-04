clc
clear all
close all

disp('Making lmdb for Middlebury2014, Kitti 2015 and ETH3D 2017 ...');

dataset_folder = './datasets_middlebury2014/training';
target_folder  = './datasets_lmdbs';
target_middle  = './datasets_lmdbs/middlebury2014';
target_kitti   = './datasets_lmdbs/kitti2015';
target_eth3d   = './datasets_lmdbs/eth3d2017';


if(~exist(target_folder))    mkdir(target_folder);  end
if(~exist(target_middle))    mkdir(target_middle); end
if(~exist(target_kitti))     mkdir(target_kitti);  end
if(~exist(target_eth3d))     mkdir(target_eth3d);  end



% target height and width for three datasets
kitti_height = 350;
kitti_width  = 1242;

eth3d_height = 425;
eth3d_width  = 707;

middle_height= 554;
middle_width = 694;

% 
train_list_middle = './datasets_lmdbs/rob_middle.list';
train_list_kitti = './datasets_lmdbs/rob_kitti.list';
train_list_eth3d = './datasets_lmdbs/rob_eth3d.list';

train_lmdb_middle = './datasets_lmdbs/middlebury_lmdb';
train_lmdb_kitti = './datasets_lmdbs/kitti_lmdb';
train_lmdb_eth3d = './datasets_lmdbs/eth3d_lmdb';

if(exist(train_lmdb_middle))    rmdir(train_lmdb_middle,'s'); end
if(exist(train_lmdb_kitti))     rmdir(train_lmdb_kitti),'s';  end
if(exist(train_lmdb_eth3d))     rmdir(train_lmdb_eth3d,'s');  end


fid_middle  = fopen(train_list_middle,'w');
fid_kitti  = fopen(train_list_kitti,'w');
fid_eth3d  = fopen(train_list_eth3d,'w');


dataset_dir = dir(dataset_folder);
j_middle = 0;
j_kitti  = 0;
j_eth3d  = 0;

for i = 1:length(dataset_dir)
    curr_dir = dataset_dir(i).name;
    full_dir = fullfile(dataset_folder,curr_dir);
    
    if length(curr_dir) < 9
        continue;
    end
    % read images and gt disparity
    img0 = imread([full_dir '/im0.png']);       
    img1 = imread([full_dir '/im1.png']);
    disp0 = parsePfm([full_dir '/disp0GT.pfm']);
    h_ = size(disp0,1);
    w_ = size(disp0,2);
    disp0(find(disp0==Inf)) = nan;       
    
    
    if(curr_dir(1) == 'M') %     
        
       img0 = imresize(img0,0.5,'nearest');
       img1 = imresize(img1,0.5,'nearest');
       disp0 = imresize(disp0,0.5,'nearest')/2;
       
       h_ = size(disp0,1);
       w_ = size(disp0,2);       
       
       % crop image and disparity       
       
       % initial
       disp_l = zeros(middle_height,middle_width) + nan;
       disp_r = zeros(middle_height,middle_width) + nan;
       disp_u = zeros(middle_height,middle_width) + nan;
       disp_b = zeros(middle_height,middle_width) + nan;
       
       % crop image and disparity
       sw_point = []; 
       sw_point(1) = 1;       
       tw_point = [];
       tw_point(1) = 1;       
       crop_w = 0;      
       if(w_ < middle_width)    
           crop_w = w_;
           sw_point(2) = 1;
           tw_point(2) = middle_width - w_ + 1;
       else
           crop_w = middle_width;
           sw_point(2) = w_ - middle_width + 1;
           tw_point(2) = 1;
       end
       %----------------
       sh_point = []; 
       sh_point(1) = 1;       
       th_point = [];
       th_point(1) = 1;       
       crop_h = 0;      
       if(h_ < middle_height)    
           crop_h = h_;
           sh_point(2) = 1;
           th_point(2) = middle_height - h_ + 1;
       else
           crop_h = middle_height;
           sh_point(2) = h_ - middle_height + 1;
           th_point(2) = 1;
       end
      
       for m=1:length(sw_point)
          for n=1:length(sh_point)
               % initial
               new_img0 = uint8(zeros(middle_height,middle_width,3) + 128);
               new_img1 = uint8(zeros(middle_height,middle_width,3) + 128);
               new_disp = zeros(middle_height,middle_width) + nan;
               
               new_img0(th_point(n):th_point(n)+crop_h-1,tw_point(m):tw_point(m)+crop_w-1,:) = img0(sh_point(n):sh_point(n)+crop_h-1,sw_point(m):sw_point(m)+crop_w-1,:);
               new_img1(th_point(n):th_point(n)+crop_h-1,tw_point(m):tw_point(m)+crop_w-1,:) = img1(sh_point(n):sh_point(n)+crop_h-1,sw_point(m):sw_point(m)+crop_w-1,:);
               new_disp(th_point(n):th_point(n)+crop_h-1,tw_point(m):tw_point(m)+crop_w-1,:) = disp0(sh_point(n):sh_point(n)+crop_h-1,sw_point(m):sw_point(m)+crop_w-1,:);

             
               % name
               name_img0 = fullfile(target_middle,sprintf('im0_%06d.png',j));
               name_img1 = fullfile(target_middle,sprintf('im1_%06d.png',j));
               name_disp = fullfile(target_middle,sprintf('disp0GT_%06d.pfm',j));
               j=j+1;
      
               % write list
               tline = [name_img0, '\t', name_img1, '\t', name_disp,'\n'];
               fprintf(fid_middle ,tline);      
       
               % write image and disparity
               imwrite(new_img0, name_img0);
               imwrite(new_img1, name_img1);
               pfmwrite(single(new_disp), name_disp);
          end
       end  
       
       
    end  
    
    
    if(curr_dir(1) == 'K') % Kitti 2015       
       new_img0 = uint8(zeros(350,1242,3)+128);
       new_img1 = uint8(zeros(350,1242,3)+128);
       new_disp = zeros(350,1242)+nan;
       
       new_img0(:,1:w_,:) = img0(end-350+1:end,:,:);
       new_img1(:,1:w_,:) = img1(end-350+1:end,:,:);
       new_disp(:,1:w_) = disp0(end-350+1:end,:);       
       
       name_img0 = fullfile(target_kitti,sprintf('im0_%06d.png',j_kitti));
       name_img1 = fullfile(target_kitti,sprintf('im1_%06d.png',j_kitti));
       name_disp = fullfile(target_kitti,sprintf('disp0GT_%06d.pfm',j_kitti));
       
       % write list
       tline = [name_img0, '\t', name_img1, '\t', name_disp,'\n'];
       fprintf(fid_kitti ,tline);
       % write image and disparity
       imwrite(new_img0, name_img0);
       imwrite(new_img1, name_img1);
       pfmwrite(single(new_disp), name_disp);
       
       j_kitti = j_kitti + 1;    
    end
    
    if(curr_dir(1) == 'E') % ETH3D 2017  
    % crop image and disparity
       new_img0_0 = img0(1:eth3d_height,1:eth3d_width,:);
       new_img0_1 = img0(1:eth3d_height,end-eth3d_width+1:end,:);
       
       new_img1_0 = img1(1:eth3d_height,1:eth3d_width,:);
       new_img1_1 = img1(1:eth3d_height,end-eth3d_width+1:end,:);
       
       new_disp_0 = disp0(1:eth3d_height,1:eth3d_width,:);
       new_disp_1 = disp0(1:eth3d_height,end-eth3d_width+1:end,:);
       
       new_img0_2 = img0(end-eth3d_height+1:end,1:eth3d_width,:);
       new_img0_3 = img0(end-eth3d_height+1:end,end-eth3d_width+1:end,:);
       
       new_img1_2 = img1(end-eth3d_height+1:end,1:eth3d_width,:);
       new_img1_3 = img1(end-eth3d_height+1:end,end-eth3d_width+1:end,:);
       new_disp_2 = disp0(end-eth3d_height+1:end,1:eth3d_width,:);
       new_disp_3 = disp0(end-eth3d_height+1:end,end-eth3d_width+1:end,:);
       % rename
       name_img0_0 = fullfile(target_eth3d,sprintf('im0_%06d.png',j_eth3d));
       name_img1_0 = fullfile(target_eth3d,sprintf('im1_%06d.png',j_eth3d));
       name_disp_0 = fullfile(target_eth3d,sprintf('disp0GT_%06d.pfm',j_eth3d));
       j_eth3d = j_eth3d + 1;
       name_img0_1 = fullfile(target_eth3d,sprintf('im0_%06d.png',j_eth3d));
       name_img1_1 = fullfile(target_eth3d,sprintf('im1_%06d.png',j_eth3d));
       name_disp_1 = fullfile(target_eth3d,sprintf('disp0GT_%06d.pfm',j_eth3d));
       j_eth3d = j_eth3d + 1;
       name_img0_2 = fullfile(target_eth3d,sprintf('im0_%06d.png',j_eth3d));
       name_img1_2 = fullfile(target_eth3d,sprintf('im1_%06d.png',j_eth3d));
       name_disp_2 = fullfile(target_eth3d,sprintf('disp0GT_%06d.pfm',j_eth3d));
       j_eth3d = j_eth3d + 1;
       name_img0_3 = fullfile(target_eth3d,sprintf('im0_%06d.png',j_eth3d));
       name_img1_3 = fullfile(target_eth3d,sprintf('im1_%06d.png',j_eth3d));
       name_disp_3 = fullfile(target_eth3d,sprintf('disp0GT_%06d.pfm',j_eth3d));
       j_eth3d = j_eth3d + 1;
       
       % write list
       tline = [name_img0_0, '\t', name_img1_0, '\t', name_disp_0,'\n'];
       fprintf(fid_eth3d ,tline);
       tline = [name_img0_1, '\t', name_img1_1, '\t', name_disp_1,'\n'];
       fprintf(fid_eth3d ,tline);
       tline = [name_img0_2, '\t', name_img1_2, '\t', name_disp_2,'\n'];
       fprintf(fid_eth3d ,tline);
       tline = [name_img0_3, '\t', name_img1_3, '\t', name_disp_3,'\n'];
       fprintf(fid_eth3d ,tline);
       
       % write image and disparity
       imwrite(new_img0_0, name_img0_0);
       imwrite(new_img0_1, name_img0_1);
       imwrite(new_img0_2, name_img0_2);
       imwrite(new_img0_3, name_img0_3);
       imwrite(new_img1_0, name_img1_0);
       imwrite(new_img1_1, name_img1_1);
       imwrite(new_img1_2, name_img1_2);
       imwrite(new_img1_3, name_img1_3);
       pfmwrite(single(new_disp_0), name_disp_0);
       pfmwrite(single(new_disp_1), name_disp_1);
       pfmwrite(single(new_disp_2), name_disp_2);
       pfmwrite(single(new_disp_3), name_disp_3);       
    end
end

fclose(fid_middle);
fclose(fid_kitti);
fclose(fid_eth3d);

! /home/leo/caffe_lzf/build/tools/convert_imageset_and_disparity.bin ./datasets_lmdbs/rob_middle.list ./datasets_lmdbs/middlebury_lmdb 0 lmdb
! /home/leo/caffe_lzf/build/tools/convert_imageset_and_disparity.bin ./datasets_lmdbs/rob_kitti.list ./datasets_lmdbs/kitti_lmdb 0 lmdb
! /home/leo/caffe_lzf/build/tools/convert_imageset_and_disparity.bin ./datasets_lmdbs/rob_eth3d.list ./datasets_lmdbs/eth3d_lmdb 0 lmdb

disp('lmdb made.');

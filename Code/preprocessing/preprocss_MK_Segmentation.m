clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB Script for Pre-Processing Merck Dataset
% 4 Trails: Lung Tumor, Lymph Node, Liver tumor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Get the tumor middle slice
% 2. Crop tumor bbox (<70pixels: 70x70 ; >70pixels: resample->70x70)

%% load raw 3d data
data_dir = '../../Data/merck_data/Raw_DATA/ALL/';
xlsx = '../../Data/merck_data/Raw_DATA/training_tumor_info.xlsx';
save_dir = '../../Data/merck_data/';

fileID_train = fopen(strcat(save_dir,'dir/','train_list.txt'),'wt');
% fileID_test = fopen(strcat(save_dir,'dir/','test_list.txt'),'wt');

[num, txt, raw] = xlsread(xlsx);
CT_list = raw(:,1);
label_list = raw(:,2);
cls_list = raw(:,5);

ind_slice = 0;
for i = 1:length(CT_list)
    ind_slice = ind_slice + 1  % count how many CT slice
    
    %% imread data
    % imread CT slice
    filename_ct = strcat(data_dir, CT_list{i});
    I_ct = uint16(imread(filename_ct));
    
    % imread tumor label
    filename_label = strcat(data_dir, label_list{i});
    I_label = uint16(imread(filename_label));
    
    % imread tumor class (1:lung tumor 2:lymph node 3:liver tumor)
    tumor_class = cls_list{i};
    
    if tumor_class==1
        %% crop the tumor (only lung tumor)
        [x,y] = ind2sub(size(I_label), find(I_label==1));
        min_x = min(x); max_x = max(x);
        min_y = min(y); max_y = max(y);
        x_size = max_x - min_x; y_size = max_y - min_y;

        if x_size<=60 && y_size<=60  % if tumor<60x60, directly crop 70x70
            x_start = max(1, min_x - (70 - x_size)/2); x_end = min(512, max_x + (70 - x_size)/2);
            y_start = max(1, min_y - (70 - y_size)/2); y_end = min(512, max_y + (70 - y_size)/2); 

        else     % if tumor>60x60, crop the original size add 15% width
            x_start = max(1, round(min(x) - 0.15 * x_size)); x_end = min(512, round(max(x) + 0.15 * x_size));
            y_start = max(1, round(min(y) - 0.15 * y_size)); y_end = min(512, round(max(y) + 0.15 * y_size));    

        end

        img_patch_tumor = I_ct(x_start:x_end, y_start:y_end);
        mask_patch_tumor = I_label(x_start:x_end, y_start:y_end);
        edge_patch_tumor = uint16(edge(mask_patch_tumor,'canny',0.5));

        % save the img & label & txt information
        name_label = label_list{i};
        name_main = name_label(1:strfind(name_label, '_label.png')-1);
        name_main = name_main(~isspace(name_main));
        
        img_file_name = strcat(name_main, '_tumor', '_img', '.png');
        mask_file_name = strcat(name_main, '_tumor', '_mask', '.png');
        edge_file_name = strcat(name_main, '_tumor', '_edge', '.png');

        img_save = char(strcat(save_dir, 'image/', img_file_name));
        imwrite(img_patch_tumor, img_save);
        mask_save = char(strcat(save_dir, 'mask/', mask_file_name));
        imwrite(mask_patch_tumor, mask_save);
        edge_save = char(strcat(save_dir, 'edge/', edge_file_name));
        imwrite(edge_patch_tumor, edge_save);
        
        dis_to_center = sprintf('%0.2f', 0);

        line = char(strcat(string(0), " ", string(dis_to_center), " ", img_file_name, " ", ...
            mask_file_name, " ", ...
            edge_file_name, ...
            " 0 0 0 0 \n"));
        fprintf(fileID_train, line);

%         figure(1),
%         subplot(1,4,1); I=imread(img_save); imshow(I,[-1000 3000]);
%         subplot(1,4,2); I=imread(mask_save); imshow(I,[0 1]);
%         subplot(1,4,3); I=imread(edge_save); imshow(I,[0 1]);
%         subplot(1,4,4); I=imread(dismap_save); imshow(I,[]);
%         drawnow;
        
%         %% crop random location
%         mat = zeros(size(I_ct));
%         mat(512*0.3:512 - 512*0.3, 512*0.2:512 - 512*0.2) = 1;
%         I_sel = mat.*(I_ct < 400);
%         [x,y] = ind2sub(size(I_sel), find(I_sel==1));
%         rand_n = randi([1 length(x)], 1);
% 
%         x_start = max(1, x(rand_n)-35); x_end = min(512, x(rand_n)+34);
%         y_start = max(1, y(rand_n)-35); y_end = min(512, y(rand_n)+34); 
% 
%         img_patch_tumor = I_ct(x_start:x_end, y_start:y_end);
%         mask_patch_tumor = I_label(x_start:x_end, y_start:y_end);
%         edge_patch_tumor = uint16(edge(mask_patch_tumor,'canny',0.5));
%         dismap_patch_tumor = uint16(bwdist(~mask_patch_tumor));
% 
%         % save the img & label & txt information
%         name_label = label_list{i};
%         name_main = name_label(1:strfind(name_label, '_label.png')-1);
%         name_main = name_main(~isspace(name_main));
%         
%         img_file_name = strcat(name_main, '_rand', '_img', '.png');
%         mask_file_name = strcat(name_main, '_rand', '_mask', '.png');
%         edge_file_name = strcat(name_main, '_rand', '_edge', '.png');
%         dismap_file_name = strcat(name_main, '_rand', '_dismap', '.png');
% 
%         img_save = char(strcat(save_dir, 'image/', img_file_name));
%         imwrite(img_patch_tumor, img_save);
%         mask_save = char(strcat(save_dir, 'mask/', mask_file_name));
%         imwrite(mask_patch_tumor, mask_save);
%         edge_save = char(strcat(save_dir, 'edge/', edge_file_name));
%         imwrite(edge_patch_tumor, edge_save);
%         dismap_save = char(strcat(save_dir, 'dismap/', dismap_file_name));
%         imwrite(dismap_patch_tumor, dismap_save);
% 
%         line = char(strcat(string(0), " ", img_file_name, " ", ...
%             mask_file_name, " ", ...
%             edge_file_name, " ", ...
%             dismap_file_name, " 0 0 0 0 \n"));
%         fprintf(fileID_train, line);

%         figure(2),
%         subplot(1,4,1); I=imread(img_save); imshow(I,[-1000 3000]);
%         subplot(1,4,2); I=imread(mask_save); imshow(I,[0 1]);
%         subplot(1,4,3); I=imread(edge_save); imshow(I,[0 1]);
%         subplot(1,4,4); I=imread(dismap_save); imshow(I,[]);
%         drawnow;
    end

end

fclose(fileID_train);
% fclose(fileID_test);



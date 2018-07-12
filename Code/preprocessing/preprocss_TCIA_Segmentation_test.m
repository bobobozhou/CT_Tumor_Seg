clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB Script for Pre-Processing Public Dataset
% NSCLC Radiogenomics: The Cancer Imaging Archive (TCIA) Public Access
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Get the tumor middle slice
% 2. Crop tumor bbox (<70pixels: 70x70 ; >70pixels: resample->70x70)

%% load raw 3d data
data_dir = '../../Data_Segmentation/public_data/Raw_DATA/3D_normalized';
files = dir(data_dir);
save_dir = '../../Data_Segmentation/public_data/';

fileID_test = fopen(strcat(save_dir,'dir/','test_list.txt'),'wt');

ind_case = 0;
for i = 3:4:length(files)
    ind_case = ind_case + 1  % count how many CT cases
    
    %% imread data
    % imread CT volume
    filename = files(i).name;
    if contains(filename, '_CT.hdr')
        file_dir = strcat(files(i).folder, '/',...
            strtok(filename, '_CT.hdr'),'_CT');
        V_ct = analyze75read(file_dir);
    end
    V_ct = uint16(V_ct);

    % imread Label volume
    filename = files(i+2).name;
    if contains(filename, '_label.hdr')
        file_dir = strcat(files(i+2).folder, '/',...
            strtok(filename, '_label.hdr'),'_label');
        V_label = analyze75read(file_dir);
    end
    V_label = uint16(V_label);

    %% select middle slice & crop the tumor
    % get the middle slice index
    [~,~,z] = ind2sub(size(V_label), find(V_label==1));
    [ min_z, max_z ] = middle_cut(z);
    z_mid = round((max_z + min_z)/2);
    [x,y] = ind2sub(size(V_label(:,:,z_mid)), find(V_label(:,:,z_mid)==1));
    min_x = min(x); max_x = max(x);
    min_y = min(y); max_y = max(y);
    
    % crop the tumor in test
    x_size = max_x - min_x; y_size = max_y - min_y;
    
    for zz = min_z:max_z     
        if x_size<=30 && y_size<=30 % if tumor<30x30, crop square with 2xmax(x_size, y_size)
            l = max(x_size, y_size);
            x_start = min_x - l/2; x_end = max_x + l/2;
            y_start = min_y - l/2; y_end = max_y + l/2;
        else
            if x_size<=60 && y_size<=60  % if tumor<60x60, directly crop 70x70
                x_start = min_x - (70 - x_size)/2; x_end = max_x + (70 - x_size)/2;
                y_start = min_y - (70 - y_size)/2; y_end = max_y + (70 - y_size)/2; 
            
            else     % if tumor>60x60, crop the original size add 15% width
                l = max(x_size, y_size);
                x_start = round(min_x - 0.15 * l); x_end = round(max_x + 0.15 * l);
                y_start = round(min_y - 0.15 * l); y_end = round(max_y + 0.15 * l);    
            end
        end
        
        img_patch_tumor = V_ct(x_start:x_end, y_start:y_end, zz);
        mask_patch_tumor = V_label(x_start:x_end, y_start:y_end, zz);
        edge_patch_tumor = uint16(edge(mask_patch_tumor,'canny',0.5));

        % save the img & label & txt information
        img_file_name = strcat(strtok(filename, '_label.hdr'), '_img_test_', string(zz), '.png');
        mask_file_name = strcat(strtok(filename, '_label.hdr'), '_mask_test_', string(zz), '.png');
        edge_file_name = strcat(strtok(filename, '_label.hdr'), '_edge_test_', string(zz), '.png');
        
        img_save = char(strcat(save_dir, 'image/', img_file_name));
        imwrite(img_patch_tumor, img_save);
        mask_save = char(strcat(save_dir, 'mask/', mask_file_name));
        imwrite(mask_patch_tumor, mask_save);
        edge_save = char(strcat(save_dir, 'edge/', edge_file_name));
        imwrite(edge_patch_tumor, edge_save);
        
        dis_to_center = sprintf('%0.2f', abs(zz - z_mid)/(max_z - min_z));
        
        line = char(strcat(string(ind_case), " ", string(dis_to_center), " ", img_file_name, " ", ...
                mask_file_name, " ", ...
                edge_file_name, ...
                " 0 0 0 0 \r\n"));
            fprintf(fileID_test, line);
            
%             figure(1),
%             subplot(1,3,1); I=imread(img_save); imshow(I,[-1000 3000]);
%             subplot(1,3,2); I=imread(mask_save); imshow(I,[0 1]);
%             subplot(1,3,3); I=imread(edge_save); imshow(I,[0 1]);
% %             suptitle(char(img_file_name));
%             drawnow;
    end
    
end

fclose(fileID_test);



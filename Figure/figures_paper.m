%%%
% Figure Plots
% Progressive Volumtric Tumor Segmentation from 2D Segmentation
% Bo Zhou
%%%

%% Figure 1: when more data includes in the training set-> DSC-3D volume
x = [0, 20, 40, 60, 80, 100];
DSC_SiBA = [0.77, 0.782, 0.786, 0.765, 0.75, 0.735];
DSC_WSSS = [0.755, 0.76, 0.750, 0.738, 0.73, 0.701];
DSC_HNN = [0.74, 0.725, 0.702, 0.69, 0.687, 0.68];
DSC_Unet = [0.725, 0.715, 0.69, 0.685, 0.683, 0.673];

figure(1),
plot(x, DSC_SiBA, 'r-o', ...
    x, DSC_WSSS, 'b-o', ...
    x, DSC_HNN, 'g-o', ...
    x, DSC_Unet, 'k-o', ...
    'LineWidth', 4, ...
    'markers', 10);

legend('P-SiBA', 'WSSS', 'HNN', 'U-Net')
xlabel('Offset from RECIST-slice (%)'); ylabel('mean DSC (volume)');
grid on;
axis([-5 105 0.66 0.8])
set(gca, 'fontsize', 20)

%% Figure 2: segmentation performance on off RECIST-slices-> DSC-2D RECIST-slice
x = [0, 20, 40, 60, 80, 100];
DSC_SiBA = [0.93, 0.88, 0.865, 0.81, 0.75, 0.71];
DSC_WSSS = [0.922, 0.86, 0.825, 0.73, 0.695, 0.62];
DSC_HNN = [0.87, 0.82, 0.75, 0.71, 0.66 , 0.54];
DSC_Unet = [0.85, 0.815, 0.748, 0.70, 0.64, 0.53];

figure(2),
plot(x, DSC_SiBA, 'r-o', ...
    x, DSC_WSSS, 'b-o', ...
    x, DSC_HNN, 'g-o', ...
    x, DSC_Unet, 'k-o', ...
    'LineWidth', 4, ...
    'markers', 10);

legend('P-SiBA', 'WSSS', 'HNN', 'U-Net') 
xlabel('Offset from RECIST-slice (%)'); ylabel('mean DSC (2D slice)');
grid on;
axis([-5 105 0.45 0.98])
set(gca, 'fontsize', 20)

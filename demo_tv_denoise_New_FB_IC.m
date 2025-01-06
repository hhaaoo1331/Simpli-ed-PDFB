 
clc,clear
path(path,'./data_wu/');


% f = double(imread('lena256','png'));
% f = double(imread('house','png'));
% f = double(imread('cameraman512','tif'));
% f = double(imread('vendredi','tif'));
% f = double(imread('fingerprint512','png'));
% f = double(imread('peppers256','png'));
% f = double(imread('house256','png'));
% f = double(imread('house','png'));
% f = double(imread('barbara512','png'));
% f = double(imread('baboon512[1]','jpg'));
%  f = double(imread('untitled','png'));
%  f = double(imread('Pirate','tif'));
% f = double(imread('Lena512','png'));
%f = double(rgb2gray(imread('building_org','png')));
% f = f/255;

  f = double(rgb2gray(imread('0010','png')));
%读取照片0010.png


% f = double(imread('boat512','png'));
% f = double(imread('goldhill','png'));


% f = double(imread('text','png'));
% f = double(imread('05','png'));

% load data_simu;
 load castle_15.mat; 
 %load castle_25.mat; 
%load castle_50.mat; 

 %load building_15.mat;
 %load building_25.mat;
 %load building_50.mat;
 
% load building_15.mat;

 % f = phantom(256);

[m,n] = size(f);%图像的行列大小


[D1,L1] = differencematrix(m,n);
[D2] = Second_differencematrix(m,n);





 %noise level
% sigma = 15;
% 
%  g = f + sigma*randn(m,n);


% L2-IC model
L = speye(m*n);
K = D1;
M = D2;

%L2-MIC model
% L = D1;
% K = speye(2*m*n);
% M = L1;
% n1=normest(L)
% n2=normest(K)
% n3=normest(M)

%% image denoise 进行图像去噪
 epsilon = 1e-5;

 iter = 1000;
 
theta1 = 0.1;
theta2 = 0.1;
tau = 0.2;
gamma = 0.1;
a1 = 1.87;

%  lambda1 = 7.7; % 
%  lambda2 = 4.4;

 lambda1 = 7.7;
 lambda2 = 20.7;

%  
% 
% 
tic
[x_update1,k1,SNR1,SSIM1,PSNR1] = New_FB_Inf_Conv_denoise(f,g,iter,epsilon,L,K,M,lambda1,lambda2,theta1,theta2,gamma,tau,a1);
toc

 figure; colormap gray;
subplot(221); imagesc(f); axis image; axis off; title('Original');
subplot(222); imagesc(g); axis image; axis off; title('Noisy');
subplot(223); imagesc(reshape(x_update1,m,n));axis off; axis image; 
subplot(224); imagesc(reshape(x_update1,m,n));axis off; axis image; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  epsilon = 1e-2;
% 
%  iter = 100;
% 
% lambda1 = 4:0.1:8.5;
% lambda2 = 5.5:0.1:7.5;
% 
% SNR = zeros(length(lambda1),length(lambda2));
% PSNR = zeros(length(lambda1),length(lambda2));
% SSIM = zeros(length(lambda1),length(lambda2));
% x_update = cell(length(lambda1),length(lambda2));
% k = zeros(length(lambda1),length(lambda2));
% 
% for i = 1:length(lambda1)
%     for j =1:length(lambda2)
% [x_update{i,j},k(i,j),SNR(i,j),SSIM(i,j),PSNR(i,j)] =  FBF_Inf_Conv_denoise(f,g,iter,epsilon,L,K,M,lambda1(i),lambda2(j));
%     end
% end
% 
% 
% maxpsnr = max(max(PSNR));
% [x1,y1] = find(PSNR == maxpsnr);
% 
% lambda1(x1),lambda2(y1)
% 
% 
% figure; colormap gray;
% subplot(221); imagesc(f); axis image; axis off; title('Original');
% subplot(222); imagesc(g); axis image; axis off; title('Noisy');
% subplot(223); imagesc(reshape(x_update{x1,y1},m,n));axis off; axis image; 
% subplot(224); imagesc(reshape(x_update{x1,y1},m,n)-f);axis off; axis image; 

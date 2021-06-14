
clear all; close all; clc;  
warning off all   
p = pwd;
%addpath(fullfile(p, '/methods'));  % the upscaling methods
addpath(fullfile(p, '/ksvdbox')) % K-SVD dictionary training algorithm
addpath(fullfile(p, '/ompbox')) % Orthogonal Matching Pursuit algorithm
addpath(fullfile(p, '/include'))
imgscale = 1;   % the scale reference we work with
upscaling = [3];

    mat_file = ['conf_finalx' num2str(upscaling)];  
    load(mat_file, 'conf');
%imshow(conhimg{1}(:,:,5));
% h [4 5 6 7 11 15 17 18 21 23 26 27 29 31]
 layer_use=[4 5 6 7 11 15 17 18 21 23 26 27 29 31];
   
  conf.desc = {'Gnd','SRCDA'};
    conf.results = {};
 fname = ['wh_SRCDA_x' num2str(upscaling) '.mat'];
load(fname);

input_dirtest = 'testset';
pattern = '*.bmp'; % Pattern to process
conf.filenamestest = glob(input_dirtest, pattern); % Cell array 

%for 
 % the magnification factor

for ij = 31:numel(conf.filenamestest)
        ftest = conf.filenamestest{ij};
        [ptest, ntest, xtest] = fileparts(ftest);
input_dir = ['simimg/rota' ntest];
 conf.filenames = glob(input_dir, pattern); % Cell array 
% tag = [input_dir '_x' num2str(upscaling) ];
%input_dir = 'train';
%input_dir = 'MYSRCDA';
%input_dir = 'Set10TMM'; % Directory with input images
%image_number = 33;
% pattern = '*.bmp'; % Pattern to process

  % conf.result_dir = qmkdir(['Results-' sprintf('%s_x%d-', input_dir, upscaling)]);
   conf.result_dirRGB = qmkdir(['ResRGB-' sprintf('%s_x%d', input_dir, upscaling)]);
    %%
    t = cputime;    
        
    conf.countedtime = zeros(numel(conf.desc),numel(conf.filenames));
    
    res =[];comres=[];
   
    for i = 1:numel(conf.filenames)
        f = conf.filenames{i};
        [p, n, x] = fileparts(f);
        %{
        simgs= load_images(glob(['simimg/rota' n], '*.bmp'));
        shimg = modcrop(simgs, conf.scale); % crop a bit (to simplify scaling issues
        % Scale down images
        slimg = resize(shimg, 1/conf.scale, conf.interpolate_kernel);
        supimg = resize(slimg, conf.upsample_factor, conf.interpolate_kernel);
        conshimg=getconhimg(shimg);
        conslimg=getconlimg(supimg);
        
        for ij=1:size(conshimg{1},3)
           for ji=1:length(conshimg)
                 exshimg{ji}=conshimg{ji}(:,:,ij);
                %exlimg{j}=conlimg{j}(:,:,i);
           end
           conexsimg{ij}=exshimg;
          %conexlimg{i}=exlimg;
         end
        %}
        %%
        im=(imread(f));
        img1{1}=im(:,:,1);img2{1}=im(:,:,2);img3{1}=im(:,:,3);
        img1=modcrop(img1,conf.scale^conf.level);
        img2=modcrop(img2,conf.scale^conf.level);
        img3=modcrop(img3,conf.scale^conf.level);
        im = (cat(3,img1{1},img2{1},img3{1}));  
        %%%%
        [img, imgCB, imgCR] = load_images({f}); 
       

        if imgscale<1
            img = resize(img, imgscale, conf.interpolate_kernel);
            imgCB = resize(imgCB, imgscale, conf.interpolate_kernel);
            imgCR = resize(imgCR, imgscale, conf.interpolate_kernel);
        end
        sz = size(img{1});
        
        fprintf('%d/%d\t"%s" [%d x %d]\n', i, numel(conf.filenames), f, sz(1), sz(2));
    
        img = modcrop(img, conf.scale^conf.level);
        imgCB = modcrop(imgCB, conf.scale^conf.level);
        imgCR = modcrop(imgCR, conf.scale^conf.level);

            low = resize(img, 1/conf.scale^conf.level, conf.interpolate_kernel);
            if ~isempty(imgCB{1})
                lowCB = resize(imgCB, 1/conf.scale^conf.level, conf.interpolate_kernel);
                lowCR = resize(imgCR, 1/conf.scale^conf.level, conf.interpolate_kernel);
            end
            
        interpolated = resize(low, conf.scale^conf.level, conf.interpolate_kernel);
        if ~isempty(imgCB{1})
            interpolatedCB = resize(lowCB, conf.scale^conf.level, conf.interpolate_kernel);    
            interpolatedCR = resize(lowCR, conf.scale^conf.level, conf.interpolate_kernel);    
        end
%}
       
        %low=img;
           % load('auto_x2.mat');
            fprintf('SRCDA\n');
            startt = tic;
            res{1} = scaleup_DNksvd(conf, low,layer_use);
            toc(startt)
            conf.countedtime(1,i) = toc(startt);
            %res{1}=im_h;
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
        
        
          result = cat(3, img{1}, res{1}{1});
        %result(:,:,3);
        result = shave(uint8(result * 255), conf.border * conf.scale);
        
        if ~isempty(imgCB{1})
            resultCB = interpolatedCB{1};
            resultCR = interpolatedCR{1};           
            resultCB = shave(uint8(resultCB * 255), conf.border * conf.scale);
            resultCR = shave(uint8(resultCR * 255), conf.border * conf.scale);
        end

        conf.results{i} = {};
      %  for j = 1:numel(conf.desc)            
           % conf.results{i}{j} = fullfile(conf.result_dir, [n sprintf('_%s_x%d', conf.desc{j}, upscaling) x]); 
           % imwrite(result(:, :, j), conf.results{i}{j});
            j=2;
            conf.resultsRGB{i}{j} = fullfile(conf.result_dirRGB, [n sprintf('_%s_x%d', conf.desc{j}, upscaling) x]);
            if ~isempty(imgCB{1})
                rgbImg = cat(3,result(:,:,j),resultCB,resultCR);
                rgbImg = ycbcr2rgb(rgbImg);
            else
                rgbImg = result(:,:,j);
            end
            
           imwrite(rgbImg, conf.resultsRGB{i}{j});
          
      %  end      
      %{
        conf.filenames{i} = f;
        result = rgbImg;
        resultim = shave(im, conf.border * conf.scale);
        sp_rmse = compute_rmse(resultim, result);

        sp_psnr = 20*log10(255/sp_rmse);

        sp_ssim = ssim(resultim, result);
    
        comres=[comres;sp_psnr;sp_ssim;];
        %}
    end   
end
    %conf.duration = cputime - t;

%save mySRCDA_x3 comres;
%save('ccrtestset-3');
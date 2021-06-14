
clear all; close all; clc;  
warning off all   
p = pwd;
%addpath(fullfile(p, '/methods'));  % the upscaling methods
addpath(fullfile(p, '/ksvdbox')) % K-SVD dictionary training algorithm
addpath(fullfile(p, '/ompbox')) % Orthogonal Matching Pursuit algorithm
addpath(fullfile(p, '/include'))
imgscale = 1;   % the scale reference we work with

%for 
upscaling = [3]; % the magnification factor
input_dir = 'testset';
%input_dir = 'train';
%input_dir = 'MYSRCDA';
%input_dir = 'Set10TMM'; % Directory with input images
%image_number = 33;
% pattern = '*.bmp'; % Pattern to process
pattern = '*.bmp'; % Pattern to process

tag = [input_dir '_x' num2str(upscaling) ];
    mat_file = ['conf_finalx' num2str(upscaling)];  
    selfmat_file = ['selfconf_finalx' num2str(upscaling)];
    if exist([mat_file '.mat'],'file')
        %disp(['Load trained dictionary...' mat_file]);
        load(mat_file, 'conf');load(selfmat_file, 'selfconf');
    else  
        conf.scale = upscaling; % scale-up factor
        conf.level = 1; % # of scale-ups to perform
        conf.window = [3 3]; % low-res. window size
        conf.border = [1 1]; % border of the image (to ignore)

        % High-pass filters for feature extraction (defined for upsampled low-res.)
        conf.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf.filters = {G, G.', L, L.'}; % 2D versions
        conf.interpolate_kernel = 'bicubic';

        conf.overlap = [1 1]; % partial overlap (for faster training)
        if upscaling <= 2
            conf.overlap = [1 1]; % partial overlap (for faster training)
        end
  
        conf.overlap = conf.window - [1 1]; % full overlap scheme (for better reconstruction)    
     
        save(mat_file, 'conf');
        % train call        
    end

fprintf('\n\n');

%imshow(conhimg{1}(:,:,5));
% h [4 5 6 7 11 15 17 18 21 23 26 27 29 31]
 layer_use=[4 5 6 7 11 15 17 18 21 23 26 27 29 31];
  
    conf.filenames = glob(input_dir, pattern); % Cell array      
    
    conf.desc = {'Gnd','SRCDA'};
    conf.results = {};
    
 fname = ['wh_SRCDA_x' num2str(upscaling) '.mat'];
 
    if exist(fname,'file')
       load(fname);
    else
        %%
       disp('Compute SRCDA regressors');
       ttime = tic;
       tic
       
 oriimg= load_images(glob('CVPR08-SR/Data/Training', '*.bmp'));
 himg = modcrop(oriimg, conf.scale); % crop a bit (to simplify scaling issues
% Scale down images
limg = resize(himg, 1/conf.scale, conf.interpolate_kernel);
upimg = resize(limg, conf.upsample_factor, conf.interpolate_kernel);
conhimg=getconhimg(himg);
conlimg=getconlimg(upimg);
%imshow(conlimg{37});
%im_b=img{1};
for i=1:size(conhimg{1},3)
    for j=1:length(conhimg)
exhimg{j}=conhimg{j}(:,:,i);
%exlimg{j}=conlimg{j}(:,:,i);
    end
    coneximg{i}=exhimg;
    %conexlimg{i}=exlimg;
end
      % load('pre_images.mat');
     %  [plores phires] = collectSamplesScales(conf, images, 12, 0.98);  
    % [plores phires] = collectSamplesScales(conf, load_images(...            
    %   glob('CVPR08-SR/Data/Training', '*.bmp')), 12, 0.98);  
    dic_plores = collect(conf, conlimg, conf.upsample_factor, conf.filters);
         [dic_plores,conf] = pca(dic_plores,conf);
          save(mat_file, 'conf');
    [~, plores] = collectSamplesScalesl(conf, conlimg, 12, 0.98);  
    l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);    
        l2(l2<0.1) = 1;
        plores = plores./l2n;
        dic_size=1024;
      %  nn_patch_max=2048;lambda=0.1;
         
    %    center
    clu_num=16;
    folder_current = pwd;
    run([folder_current, '\vlfeat-0.9.19\toolbox\vl_setup.m']);
    [centers, assignments] = vl_kmeans(dic_plores, clu_num, 'Initialization', 'plusplus');
    num_centers_use=4;nn_patch_max=2048;lambda=0.1;
    assignments = double(assignments);
    %}
    for ii=1:length(layer_use)
        
        [~, phires] = collectSamplesScalesh(conf, coneximg{layer_use(ii)}, 12, 0.98); 
  % [~,plores] = collectfeature(conf, conlimg); 
   %[~, phires] = collectfeature(conf, coneximg{layer_use(ii)});  
   %{
        l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);    
        l2(l2<0.1) = 1;
        plores = plores./l2n;
        %}
        phires= phires./repmat(l2,size(phires,1),1);
        %{
        clear l2
        clear l2n
%}
 dic_phires = collect(conf, coneximg{layer_use(ii)}, conf.upsample_factor, {});
         conf_use=conf;
         disp('train')
         for ic = 1:clu_num
            D = pdist2(single(centers'), single(centers(:, ic)'));
            [~, idx_centers] = sort(D, 'ascend');
            
            idx_centers_use = idx_centers(1:num_centers_use);
            idx_patch_use = [];
            for i_temp = 1:num_centers_use
                idx_temp = find(assignments == idx_centers_use(i_temp));
                idx_patch_use = [idx_patch_use idx_temp];
            end
            sub_plores = dic_plores(:, idx_patch_use);
            sub_phires = dic_phires(:, idx_patch_use);
         
         
         
         conf_use = learn_dict_hire(conf_use, sub_plores,sub_phires, dic_size);    
         
         for i = 1:size(conf_use.dict_lores,2)
            D = pdist2(single(plores'),single(conf_use.dict_lores(:,i)'));
            [~, idx] = sort(D);                
            Lo = plores(:, idx(1: nn_patch_max));                                    
            Hi = phires(:, idx(1: nn_patch_max));
            lh{i} = Hi*((Lo'*Lo+lambda*eye(size(Lo,2)))\Lo'); 
            %Aplus_PPs{i} = Hi*(inv(Lo*Lo'+llambda*eye(size(Lo,1)))*Lo)'; 
        end        
      
         F{ii}{ic}=lh;
        % centers{ii}=conf_use.dict_lores;
         dicl{ii}{ic}=conf_use.dict_lores;
         disp('Done!');
         end
   %{
       for i = 1:clu_num
            D = pdist2(single(centers'), single(centers(:, i)'));
            [~, idx_centers] = sort(D, 'ascend');
            
            idx_centers_use = idx_centers(1:num_centers_use);
            idx_patch_use = [];
            for i_temp = 1:num_centers_use
                idx_temp = find(assignments == idx_centers_use(i_temp));
                idx_patch_use = [idx_patch_use idx_temp];
            end
            sub_plores = plores(:, idx_patch_use);
            sub_phires = phires(:, idx_patch_use);
            
            if nn_patch_max <= length(idx_patch_use)
                sub_D = pdist2(single(sub_plores'), single(centers(:, i)'));
                [~, sub_idx] = sort(sub_D, 'ascend');
                Lo = sub_plores(:, sub_idx(1:nn_patch_max));
                Hi = sub_phires(:, sub_idx(1:nn_patch_max));
            else
                Lo = sub_plores;
                Hi = sub_phires;
            end     
           lh{i} = Hi*((Lo'*Lo+lambda*eye(size(Lo,2)))\Lo');
       end
        F{ii}=lh;
        %}
    end
       clear l2
        clear l2n
     save(fname,'F','centers','dicl');  

       %%%%%%%%%%%%%%%%%%%%%%
       %%%%%%%% End Traning %%%%%%%%%
       %%%%%%%%%%%%%%%%%%%%%%
        
       %}
    
      %  ttime = toc(ttime);        
         
        %toc
    end 
  %}  

    %%    
   
  
   conf.result_dir = qmkdir(['Results-' sprintf('%s_x%d-', input_dir, upscaling)]);
   conf.result_dirRGB = qmkdir(['ResRGB-' sprintf('%s_x%d-', input_dir, upscaling)]);
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
           % res{1} = scaleup_DNksvd(conf, low,layer_use);
            res{1} = scaleup_DNksvdsimself(conf,selfconf, low,layer_use,n);
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
        conf.filenames{i} = f;
        result = rgbImg;
        resultim = shave(im, conf.border * conf.scale);
        sp_rmse = compute_rmse(resultim, result);

        sp_psnr = 20*log10(255/sp_rmse);

        sp_ssim = ssim(resultim, result);
    
        comres=[comres;sp_psnr;sp_ssim;];
    end   
    %conf.duration = cputime - t;

%save mySRCDA_x3 comres;
%save('ccrtestset-3');
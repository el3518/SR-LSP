
clear all; close all; clc;  
warning off all   
p = pwd;
%addpath(fullfile(p, '/methods'));  % the upscaling methods
addpath(fullfile(p, '/ksvdbox')) % K-SVD dictionary training algorithm
addpath(fullfile(p, '/ompbox')) % Orthogonal Matching Pursuit algorithm
addpath(fullfile(p, '/include'))
imgscale = 1;   % the scale reference we work with

upscaling = [2]; % the magnification factor
input_dir = 'testuse';

pattern = '*.bmp'; % Pattern to process
mat_file = ['conf_x' num2str(upscaling)];    
    if exist([mat_file '.mat'],'file')
       load(mat_file,'conf');%load(fname_A);
    %end
    else 
% Simulation settings
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
  
        % full overlap scheme (for better reconstruction)    
         conf = learn_pca(conf, load_images(...            
            glob('CVPR08-SR/Data/Training', '*.bmp') ...
            ), 1024);  
         conf.overlap = conf.window - [1 1];
        
        %save(mat_file, 'conf');
        % train call        
      end
    conf.filenames = glob(input_dir, pattern); % Cell array      
    
    conf.desc = {'Gnd','HierRes'};
    conf.results = {};
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    conf.winsize=4;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%   K-means clustering  %%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fname = ['MY_HierRes_x' num2str(upscaling) '.mat'];
    fname_A = ['MY_HierRes_A_x' num2str(upscaling) '.mat'];
    
    
    if exist(fname,'file')
       load(fname);load(fname_A);
    else
        %%
       disp('Compute HierRes regressors');
       ttime = tic;
       tic
       [plores  phires  plores_filt] = collectSamplesScales(conf, load_images(...            
           glob('CVPR08-SR/Data/Training', '*.bmp')), 12, 0.98); 
       
        num_samples = 5000000;
        plores=plores(:,1:num_samples);
        phires=phires(:,1:num_samples);
        plores_filt=plores_filt(:,1:num_samples);
           %glob('CVPR08-SR/Data/train', '*.bmp')), 12, 0.98);  
       [plores_hire conf]=ext_fea_hire(conf, plores);
%         if size(plores,2) > num_patches_custer                
%             plores = plores(:,1:num_patches_cluster);
%             phires = phires(:,1:num_patches_cluster);
%         end
       save(mat_file, 'conf');
         clear plores 
        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores_filt.^2).^0.5+eps;
        l2n = repmat(l2,size(plores_filt,1),1);    
        l2(l2<0.1) = 1;
        plores_filt= plores_filt./l2n;
        clear l2n
        l2n_h = repmat(l2,size(phires,1),1);
        clear l2
        phires = phires./l2n_h;
        clear l2n_h

        l2 = sum(plores_hire.^2).^0.5+eps;
        l2n = repmat(l2,size(plores_hire,1),1);    
        l2(l2<0.1) = 1;
        plores_hire= plores_hire./l2n;
        clear l2 l2n
        %llambda_kmeans_sub = 0.1;
        %cluster the whole data with kmeans plus plus
        clu_num=256;
        folder_current = pwd;
        run([folder_current, '\vlfeat-0.9.19\toolbox\vl_setup.m']);
        [centers, assignments] = vl_kmeans(plores_hire, clu_num, 'Initialization', 'plusplus');
        assignments = double(assignments);
        location=[assignments; zeros(1,size(plores_hire,2))]; j=1;
        pnum=(conf.window(1)*conf.scale-conf.winsize+1)^2;
        for i=1:pnum:size(plores_hire,2)
            location(2,i:i+pnum-1)=j;
            j=j+1;
        end
        llambda=0.1;dic_size=1024;nei_size=2048;cen_use=1;
        Dic=[]; Aplus=[];
        for i = 1:clu_num 
          D = pdist2(single(centers'), single(centers(:, i)'));
          [~, idx_centers] = sort(D, 'ascend');
            
          idx_centers_use = idx_centers(1:cen_use);
          idx_patch_use = [];
         for i_temp = 1:cen_use
              idx_temp = find(assignments == idx_centers_use(i_temp));
               % idx_temp = find(assignments == i);
              %  idx_patch_use= idx_temp;
              idx_patch_use = [idx_patch_use idx_temp];
         end
            idx_patch_uni=unique(location(2,idx_patch_use));
            %{
            if length(idx_patch_uni)
            D = pdist2(single(centers'), single(centers(:, i)'));
          [~, idx_centers] = sort(D, 'ascend');
            
            %}
            sub_plores = plores_filt(:, idx_patch_uni);
            sub_phires = phires(:, idx_patch_uni);
            conf_use=conf;
            
        startt = tic;
        conf_use = learn_dict_hire(conf_use, sub_plores,sub_phires, dic_size);       
        conf_use.trainingtime = toc(startt);
        toc(startt)   
        Dic{1}{i}= conf_use.dict_lores;
        Dic{2}{i}= conf_use.dict_hires;
       for j = 1:size(conf_use.dict_lores,2)
            D = pdist2(single(sub_plores'),single(conf_use.dict_lores(:,j)'));
            [~, idx] = sort(D);                
            Lo = sub_plores(:, idx(1:nei_size));                                    
            Hi = sub_phires(:, idx(1:nei_size));
            clear sub_plores
            clear sub_phires
            Aplus{i}{j} = Hi*inv(Lo'*Lo+llambda*eye(size(Lo,2)))*Lo'; 
            %Aplus_PPs{i} = Hi*(inv(Lo*Lo'+llambda*eye(size(Lo,1)))*Lo)'; 
        end        
        end
       
        clear plores_filt
      
        clear phires
       
        
        ttime = toc(ttime);        
        save(fname,'Dic','centers');   
         save(fname_A,'Aplus');
        toc
    
    end
 %save(mat_file, 'conf');
    %%    
    conf.result_dir = qmkdir(['Results-' sprintf('%s_x%d-', input_dir, upscaling)]);
    conf.result_dirRGB = qmkdir(['ResRGB-' sprintf('%s_x%d-', input_dir, upscaling)]);
    %%
    t = cputime;    
        
    conf.countedtime = zeros(numel(conf.desc),numel(conf.filenames));
    
    comres =[];
    for i = 1:numel(conf.filenames)
        f = conf.filenames{i};
        [p, n, x] = fileparts(f);
         im=(imread(f));
        img1{1}=im(:,:,1);img2{1}=im(:,:,2);img3{1}=im(:,:,3);
        img1=modcrop(img1,conf.scale^conf.level);
        img2=modcrop(img2,conf.scale^conf.level);
        img3=modcrop(img3,conf.scale^conf.level);
        im = (cat(3,img1{1},img2{1},img3{1}));

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

            fprintf('HireRes\n');

            startt = tic;
            res{1} = scaleup_HireRes(conf, low, Dic, Aplus,centers);
            toc(startt)
            conf.countedtime(1,i) = toc(startt);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        result = cat(3, img{1}, res{1}{1});
        
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
       %comYUV_Tr3=[comYUV_Tr3;sp_psnr; sp_ssim; sp_rmse;];
        comres=[comres;sp_psnr;sp_ssim;];
        
    end
     ps(:,1)=comres(1:2:end);
  %  conf.duration = cputime - t;
   %image_number=19;
    % Test performance
    % scores = run_comparison(conf);
    % scores = run_comparison(conf,image_number);
    % PSNR
  %  run_comparisonRGB_PSNR(conf); 
 % save HireRes_x2 comres;
  save HireRes_gau01_winsize5_x3 comres;
     %imshow(imgres);imshow(im)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
  
  
  
  

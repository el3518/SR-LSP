clear all; close all; clc;  
warning off all   
p = pwd;
%addpath(fullfile(p, '/methods'));  % the upscaling methods
addpath(fullfile(p, '/ksvdbox')) % K-SVD dictionary training algorithm
addpath(fullfile(p, '/ompbox')) % Orthogonal Matching Pursuit algorithm
addpath(fullfile(p, '/include'))
imgscale = 1;   % the scale reference we work with
upscaling = [3];

    mat_file = ['selfconf_finalx' num2str(upscaling)];  
    load(mat_file, 'selfconf');
    
      selfconf.desc = {'Gnd','SRCDA'};
    selfconf.results = {};


%for 
 % the magnification factor
 %{
dic_plores = collect(selfconf, load_images(...            
           glob('selftrainRGB_x3', '*.bmp')), selfconf.upsample_factor, selfconf.filters);
[dic_plores,selfconf] = pca(dic_plores,selfconf);
gndim=load_images(glob('selftrainGND_x3', '*.bmp'));
rgbim=load_images(glob('selftrainRGB_x3', '*.bmp'));
for i=1:length(gndim)
    fgndim{i}=gndim{i}-rgbim{i};
end
dic_phires = collect(selfconf, fgndim, selfconf.upsample_factor, {});
dic_size=1024;         
selfconf = learn_dict_hire(selfconf, dic_plores,dic_phires, dic_size);
 save(mat_file, 'selfconf');
 %}
 
input_dirtest = 'testset';
pattern = '*.bmp'; % Pattern to process
selfconf.filenamestest = glob(input_dirtest, pattern); % Cell array 
 
[tplores tphires] = collectSamplesScaleslh(selfconf,load_images(glob('selftrainRGB_x3', '*.bmp')),...
    load_images(glob('selftrainGND_x3', '*.bmp')), 12, 0.98);
 
% upscaling = [3]; 
 nn_patch_max=2048;lambda=0.1;dic_size=1024;   
for ij = 1:numel(selfconf.filenamestest)
        ftest = selfconf.filenamestest{ij};
        [ptest, ntest, xtest] = fileparts(ftest);
         fname = [ntest '_SRCDA_x' num2str(upscaling) '.mat'];
 if exist(fname,'file')
       disp('existed!')
 else
     disp(['Compute' fname 'regressors']);
     
      
     [splores sphires] = collectSamplesScaleslh(selfconf,...
         load_images(glob(['selfSRx3\train' ntest 'RGB_x' num2str(upscaling)], '*.bmp')),...
    load_images(glob(['selfSRx3\train' ntest 'GND_x' num2str(upscaling)], '*.bmp')), 12, 0.98);
   plores=[tplores,splores];
   phires=[tphires,sphires];
     l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);    
        l2(l2<0.1) = 1;
        plores = plores./l2n;
        phires= phires./repmat(l2,size(phires,1),1);
        clear l2
        clear l2n
        
       % dic_size=1024;
       % nn_patch_max=2048;lambda=0.1;
       %{
      clu_num=1024;
    folder_current = pwd;
    run([folder_current, '\vlfeat-0.9.19\toolbox\vl_setup.m']);
    [simcenters, simassignments] = vl_kmeans(plores, clu_num, 'Initialization', 'plusplus');
    num_centers_use=6;nn_patch_max=2048;lambda=0.1;
    simassignments = double(simassignments);
        %}
   tic;
       for i = 1:size(selfconf.dict_lores,2)
            D = pdist2(single(plores'),single(selfconf.dict_lores(:,i)'));
            [~, idx] = sort(D);                
            Lo = plores(:, idx(1: nn_patch_max));                                    
            Hi = phires(:, idx(1: nn_patch_max));
           simD{i} = Hi*((Lo'*Lo+lambda*eye(size(Lo,2)))\Lo'); 
            %Aplus_PPs{i} = Hi*(inv(Lo*Lo'+llambda*eye(size(Lo,1)))*Lo)'; 
        end       
       %{
    for i = 1:clu_num
            D = pdist2(single(simcenters'), single(simcenters(:, i)'));
            [~, idx_centers] = sort(D, 'ascend');
            
            idx_centers_use = idx_centers(1:num_centers_use);
            idx_patch_use = [];
            for i_temp = 1:num_centers_use
                idx_temp = find(simassignments == idx_centers_use(i_temp));
                idx_patch_use = [idx_patch_use idx_temp];
            end
            sub_plores = plores(:, idx_patch_use);
            sub_phires = phires(:, idx_patch_use);
            
            if nn_patch_max <= length(idx_patch_use)
                sub_D = pdist2(single(sub_plores'), single(simcenters(:, i)'));
                [~, sub_idx] = sort(sub_D, 'ascend');
                Lo = sub_plores(:, sub_idx(1:nn_patch_max));
                Hi = sub_phires(:, sub_idx(1:nn_patch_max));
            else
                Lo = sub_plores;
                Hi = sub_phires;
            end     
           simD{i} = Hi*((Lo'*Lo+lambda*eye(size(Lo,2)))\Lo');
    end
     %}
     save(fname,'simD'); 
     plores=[];
     phires=[];
     disp(['Done' fname 'regressors']);
     toc;
    % input_dir = ['simimg/rota' ntest];
    % conf.filenames = glob(input_dir, pattern); % Cell array 
 end

end
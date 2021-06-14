function [imgs, midres] = scaleup_DNksvdsimself(conf,selfconf, imgs,layer_use,n)
%[imgs, midres] = scaleup_DNksvd(conf, imgs,layer_use,conexsimg,conslimg)
% Super-Resolution Iteration imgs=low;imshow(imgs{1});
    fprintf('Scale-Up deep network');
        
   load(['wh_SRCDA_x' num2str(conf.upsample_factor) '.mat']);
    midres = resize(imgs, conf.upsample_factor, conf.interpolate_kernel);

    interpolated = resize(imgs, conf.scale, conf.interpolate_kernel);
    
    for r = 1:numel(midres)
        
    load(['SRCNNlr_x' num2str(conf.upsample_factor)  '.mat']);
    [conv1_patchsize2,conv1_filters] = size(weights_conv1);
    conv1_patchsize = sqrt(conv1_patchsize2);
    [conv2_channels,conv2_patchsize2,conv2_filters] = size(weights_conv2);
    conv2_patchsize = sqrt(conv2_patchsize2);
    im_b=midres{r};
    [hei, wid] = size(im_b);
    
    weights_conv1 = reshape(weights_conv1, conv1_patchsize, conv1_patchsize, conv1_filters);
conv1_data = zeros(hei, wid, conv1_filters);
for i = 1 : conv1_filters
    conv1_data(:,:,i) = imfilter(im_b, weights_conv1(:,:,i), 'same', 'replicate');
    conv1_data(:,:,i) = max(conv1_data(:,:,i) + biases_conv1(i), 0);
end
%conv1_data(:,:,6);
%% conv2
conv2_data = zeros(hei, wid, conv2_filters);
conlimg{r}=zeros(hei, wid);

for i = 1 : conv2_filters
    for j = 1 : conv2_channels
        conv2_subfilter = reshape(weights_conv2(j,:,i), conv2_patchsize, conv2_patchsize);
        conv2_data(:,:,i)= conv2_data(:,:,i) + imfilter(conv1_data(:,:,j), conv2_subfilter, 'same', 'replicate');
    end
     conlimg{r} = conlimg{r}+conv2_data(:,:,i) + biases_conv2(i);
end
conlimg{r} = max(conlimg{r}, 0);
%simfeatures = collect(conf, conslimg, conf.upsample_factor, {}); 
 features = collect(conf, conlimg, conf.upsample_factor, {});   
    end
conv2h_data = zeros(hei, wid, conv2_filters);
num=20;
for ii=1:length(layer_use)
   % simhfeatures = collect(conf, conexsimg{ii}, conf.upsample_factor, {});
       D = pdist2(single(centers{ii})', single(features)');
       
        % D = pdist2(single(rand(81,12))', single(rand(81,24))');
       [val idx] = min(D);
        for l = 1:size(features,2) 
            %{
             simD = pdist2(single(simfeatures)', single(features(:,l))');
            [simval, simidx] = sort(simD, 'ascend');
            sumhfea=exp(-val(l))*F{ii}{idx(l)} * features(:,l);
            dfea=exp(-val(l));
            for ll=1:num
            dfea=dfea+ exp(-simval(ll)) ;  
            sumhfea =sumhfea + exp(-simval(ll))*simhfeatures(simidx(ll)) ;
            end
             hfea(:,l)=sumhfea/dfea;
           %}
      hfea(:,l) = F{ii}{idx(l)} * features(:,l);
        end

        img_size = size(imgs{r}) * conf.scale;
        grid = sampling_grid(img_size, ...
            conf.window, conf.overlap, conf.border, conf.scale);
        result = overlap_add(hfea, img_size, grid);
        ressultinte=result(conf.scale+1:end-conf.scale,conf.scale+1:end-conf.scale);
        interresult = imresize(ressultinte, [hei, wid], conf.interpolate_kernel);
        intelabel=find(result<=1e-03);
        result(intelabel)=interresult(intelabel);
       % imgs{i} = result; % for the next iteration
        conv2h_data(:,:,layer_use(ii))=result;
        % conv2h_data(:,:,4);
        fprintf('.');
end
load(['SRCNNhr_x' num2str(conf.upsample_factor) '.mat']);
[conv3_channels,conv3_patchsize2] = size(weights_conv3);
conv3_patchsize = sqrt(conv3_patchsize2);

conv3_data = zeros(hei, wid);
for i = 1 : conv3_channels
    conv3_subfilter = reshape(weights_conv3(i,:), conv3_patchsize, conv3_patchsize);
    conv3_data(:,:) = conv3_data(:,:) + imfilter(conv2h_data(:,:,i), conv3_subfilter, 'same', 'replicate');
end

%% SRCNN reconstruction
im_h = conv3_data(:,:) + biases_conv3;
%im_h = max(conv3_data(:,:) + biases_conv3,0);
%imshow(im_h)
label=find(im_h<=1e-03);
im_h(label)=interpolated{1}(label);
%imgs{r} = im_h;

load(['SRCNNlh_x' num2str(conf.upsample_factor) '.mat']);
[conv1_patchsize2,conv1_filters] = size(weights_conv1);
conv1_patchsize = sqrt(conv1_patchsize2);
[conv2_channels,conv2_patchsize2,conv2_filters] = size(weights_conv2);
conv2_patchsize = sqrt(conv2_patchsize2);
[conv3_channels,conv3_patchsize2] = size(weights_conv3);
conv3_patchsize = sqrt(conv3_patchsize2);
[hei, wid] = size(im_h);

%% conv1
weights_conv1 = reshape(weights_conv1, conv1_patchsize, conv1_patchsize, conv1_filters);
conv1_data = zeros(hei, wid, conv1_filters);
for i = 1 : conv1_filters
    conv1_data(:,:,i) = imfilter(im_h, weights_conv1(:,:,i), 'same', 'replicate');
    conv1_data(:,:,i) = max(conv1_data(:,:,i) + biases_conv1(i), 0);
end

%% conv2
conv2_data = zeros(hei, wid, conv2_filters);
for i = 1 : conv2_filters
    for j = 1 : conv2_channels
        conv2_subfilter = reshape(weights_conv2(j,:,i), conv2_patchsize, conv2_patchsize);
        conv2_data(:,:,i)= conv2_data(:,:,i) + imfilter(conv1_data(:,:,j), conv2_subfilter, 'same', 'replicate');
    end
    conv2_data(:,:,i) = max(conv2_data(:,:,i) + biases_conv2(i), 0);
end

%% conv3
conv3_data = zeros(hei, wid);
for i = 1 : conv3_channels
    conv3_subfilter = reshape(weights_conv3(i,:), conv3_patchsize, conv3_patchsize);
    conv3_data(:,:) = conv3_data(:,:) + imfilter(conv2_data(:,:,i), conv3_subfilter, 'same', 'replicate');
end

%% SRCNN reconstruction
im_h1 = conv3_data(:,:) + biases_conv3;


%{
imgsim{r} = im_h;
featuresim = selfconf.V_pca'*collect(conf, imgsim, conf.upsample_factor, conf.filters); 
load([n '_SRCDA_x' num2str(conf.upsample_factor) '.mat'])
Dsim = pdist2(single(selfconf.dict_lores)', single(featuresim)');
       
        % D = pdist2(single(rand(81,12))', single(rand(81,24))');
       [valsim idxsim] = min(Dsim);
        for l = 1:size(featuresim,2) 
           
      hfeasim(:,l) = simD{idxsim(l)} * featuresim(:,l);
        end
        hfeasim = hfeasim + collect(conf, imgsim, conf.upsample_factor, {});
       img_size = size(imgs{r}) * conf.scale;
        grid = sampling_grid(img_size, ...
            conf.window, conf.overlap, conf.border, conf.scale);
        himsim = overlap_add(hfeasim, img_size, grid);
        imgs{r} = himsim; % for the next iteration
        
        %}
imgs{r} = im_h1+im_h;
%imshow(im_h);
fprintf('\n');
end


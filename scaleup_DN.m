function [imgs, midres] = scaleup_DN(conf, imgs,layer_use)
% Super-Resolution Iteration imgs=low;imshow(imgs{1});
    fprintf('Scale-Up deep network');
        
    load('wh_SRCDA_x3' );
    midres = resize(imgs, conf.upsample_factor, conf.interpolate_kernel);

    interpolated = resize(imgs, conf.scale, conf.interpolate_kernel);
    
    for r = 1:numel(midres)
        
    load('SRCNNlr_15000000' );
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

 features = collect(conf, conlimg, conf.upsample_factor, {});   
    end
conv2h_data = zeros(hei, wid, conv2_filters);
for ii=1:length(layer_use)
       D = pdist2(single(centers)', single(features)');
       [val idx] = min(D);
        for l = 1:size(features,2) 
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
load('SRCNNhr_15000000' );
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
label=find(im_h<=1e-03);
im_h(label)=interpolated{1}(label);
imgs{r} = im_h;
%imshow(im_h);
fprintf('\n');
end


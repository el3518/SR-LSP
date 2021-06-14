function [imgmy, midresR,midresG,midresB] = scaleup_MY(Dic,conf_KmeansR,conf_KmeansG,conf_KmeansB, lowR,lowG,lowB)

% Super-Resolution Iteration
    fprintf('Scale-Up MY');
    
       %{
    midresR = resize(lowR, sz, conf_KmeansR.interpolate_kernel);    
    midresG = resize(lowG, sz, conf_KmeansG.interpolate_kernel);
    midresB = resize(lowB, sz, conf_KmeansB.interpolate_kernel);
    %}
    midresR = resize(lowR, conf_KmeansR.upsample_factor, conf_KmeansR.interpolate_kernel);    
    midresG = resize(lowG, conf_KmeansG.upsample_factor, conf_KmeansG.interpolate_kernel);
    midresB = resize(lowB, conf_KmeansB.upsample_factor, conf_KmeansB.interpolate_kernel);
    %}
    for i = 1:numel(midresR)
        featuresR = collect3(conf_KmeansR, {midresR{i}}, conf_KmeansR.upsample_factor, conf_KmeansR.filters);
        featuresR = double(featuresR);
    end
     for i = 1:numel(midresG)
        featuresG = collect3(conf_KmeansG, {midresG{i}}, conf_KmeansG.upsample_factor, conf_KmeansG.filters);
        featuresG = double(featuresG);
     end
      for i = 1:numel(midresB)
        featuresB = collect3(conf_KmeansB, {midresB{i}}, conf_KmeansB.upsample_factor, conf_KmeansB.filters);
        featuresB = double(featuresB);
      end
        % Reconstruct using patches' dictionary and their anchored
        % projections
          
        featuresR = conf_KmeansR.V_pca'*featuresR;
        featuresG = conf_KmeansG.V_pca'*featuresG;
        featuresB = conf_KmeansB.V_pca'*featuresB;
        features=[featuresR;featuresG;featuresB];
        dict_lore=[conf_KmeansR.dict_lore_kmeans;conf_KmeansG.dict_lore_kmeans;conf_KmeansB.dict_lore_kmeans];
        patches = zeros(size(Dic.MY{1},1),size(featuresR,2));
        patchesR = zeros(size(Dic.MY{1},1)/3,size(featuresR,2));
        patchesG = zeros(size(Dic.MY{1},1)/3,size(featuresG,2));
        patchesB = zeros(size(Dic.MY{1},1)/3,size(featuresB,2));
        %D = pdist2(single(dict_lore)', single(features)');
        
         D = 0.8*pdist2(single(conf_KmeansR.dict_lore_kmeans)', single(featuresR)')...
              +0.1*pdist2(single(conf_KmeansG.dict_lore_kmeans)', single(featuresG)')...
              +0.1*pdist2(single(conf_KmeansB.dict_lore_kmeans)', single(featuresB)');
        [val idx] = min(D);
        for l = 1:size(features,2)            
            patches(:,l) = Dic.MY{idx(l)} * features(:,l);
            patchesR(:,l)=patches(1:size(Dic.MY{1},1)/3,l);
            patchesG(:,l)=patches(size(Dic.MY{1},1)/3+1:2*size(Dic.MY{1},1)/3,l);
            patchesB(:,l)=patches(2*size(Dic.MY{1},1)/3+1:3*size(Dic.MY{1},1)/3,l);
        end
       

        
        % Add low frequencies to each reconstructed patch        
        patchesR = patchesR + collect(conf_KmeansR, {midresR{i}}, conf_KmeansR.scale, {});
        patchesG = patchesG + collect(conf_KmeansG, {midresG{i}}, conf_KmeansG.scale, {});
        patchesB = patchesB + collect(conf_KmeansB, {midresB{i}}, conf_KmeansB.scale, {});
        % Combine all patches into one image
        img_size = size(lowR{i}) * conf_KmeansR.scale;%size(lowR{i}) * conf_KmeansR.scale;
        grid = sampling_grid(img_size, ...
            conf_KmeansR.window, conf_KmeansR.overlap, conf_KmeansR.border, conf_KmeansR.scale);
        resultR = overlap_add(patchesR, img_size, grid);
        resultG = overlap_add(patchesG, img_size, grid);
        resultB = overlap_add(patchesB, img_size, grid);
        resultR=uint8(resultR* 255);
        resultG=uint8(resultG* 255);
        resultB=uint8(resultB* 255);
        imgf(:,:,1)=resultR;
        imgf(:,:,2)=resultG;
        imgf(:,:,3)=resultB;
        imgmy{i}=imgf;
       clear imgf
        %imgs{i} = result; % for the next iteration
        fprintf('.');
    end


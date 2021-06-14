simfeatures = collect(conf, conslimg, conf.upsample_factor, {}); 
num=20;
for ii=1:length(layer_use)
    simhfeatures = collect(conf, conexsimg{ii}, conf.upsample_factor, {});
     %  D = pdist2(single(centers{ii})', single(features)');
       
        % D = pdist2(single(rand(81,12))', single(rand(81,24))');
      % [val idx] = min(D);
      simv=[];simid=[];
       for l = 1:size(features,2) 
             simD = pdist2(single(simfeatures)', single(features(:,l))');
            [simval, simidx] = sort(simD, 'ascend');
           simv=[simv,simval(1:num)];
           simid=[simid,simidx(1:num)];
          % simD=rand(20,10);
     %  hfea(:,l) = F{ii}{idx(l)} * features(:,l);
        end
end
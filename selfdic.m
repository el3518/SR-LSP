
clear;clc;
%sefd=load_images(glob('downtestset_3','*.bmp'));
sefimg=load_images(glob('testset','*.bmp'));
  
   % midres = resize(clores, selfconf.upsample_factor, selfconf.interpolate_kernel);
   %{
   if result==im
   disp('right')
   end
   %}
   % selfconf=conf;
    upscaling=3;
   % mat_file = ['selfconf_finalx' num2str(upscaling)];
   %   save(mat_file, 'selfconf');
      downfile = qmkdir([ 'downtestset_' num2str(upscaling)]);
     %  im=imread('downtestset_x3/baboon_x3.bmp');
      %     im2single(im);
      numscales=20;scalefactor=0.98;
      for scale =1:numscales
    sfactor = scalefactor^(scale-1);
    chires = resize(sefimg, sfactor, 'bicubic');
     chires = modcrop(chires, selfconf.scale); % crop a bit (to simplify scaling issues)
    % Scale down images
    clores = resize(chires, 1/selfconf.scale, selfconf.interpolate_kernel);
      for i=1:length(clores)
           f = selfconf.filenames{i};
        [p, n, x] = fileparts(f);
        resultsRGB{i} = fullfile(downfile, [n sprintf('%d_x%d', scale, upscaling) x]);
      result = uint8(clores{i} * 255);
       imwrite(result, resultsRGB{i});
      end
      end
  %%image prepare%%%normalization   
  clear;clc;close all;
      [img,imgCB,imgCR]=load_images(glob('CVPR08-SR/Data/Training', '*.bmp'));  
      image=[];
      for i=1:size(img,1)
        a=img{i};
        meana=mean(a(:));
        vara=sqrt(var(a(:)));
        a=(a-meana)/vara;
        image{i}=a;
      end
save pre_image image;
images=image';
save pre_images images;

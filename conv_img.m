function imgs=conv_img(w_con,b_con,img)
imgs=cell(length(img),1);
for j=1:length(img)
    imgs{j}=zeros(size(img{j}));
for i=1:length(w_con)
    conimg{i}=imfilter(img{j},w_con{i},'same','replicate');
    imgs{j}= max(imgs{j}+ conimg{i},0);
end
 %imgs{j}= imgs{j}/length(w_con);
 %   imgs{j}= imgs{j}/length(w_con);
end
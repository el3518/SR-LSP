function patches = normalize_fea(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer
%patches=patche;
% Remove DC (mean of images).
%patches = bsxfun(@minus, patches, means);
patches = bsxfun(@minus, patches, mean(patches));
%{
[nsamples, nfeatures] = size(patches);
sigma = patches * patches' ./ nfeatures;
[U S V] = svd(sigma);
epsilon = 0.1;
ZCA=U*diag(1./sqrt(diag(S)+epsilon))*U';
patches=ZCA*patches;
% Truncate to +/-3 standard deviations and scale to -1 to 1
%patches = max(min(patches, pstd), -pstd) / pstd;
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;
% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.45 + 0.05;
%}
%{
%%Whitening%%
[nsamples, nfeatures] = size(patches);
sigma = patches * patches' ./ nfeatures;
[U S V] = svd(sigma);
epsilon = 0.1;
ZCA=U*diag(1./sqrt(diag(S)+epsilon))*U';
patches=ZCA*patches;
%}
end

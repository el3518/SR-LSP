% pca_gen.m
%close all;
%clear all;clc;
%%================================================================
%% Step 0a: Load data  
  [plores phires] = collectSamplesScales(conf, load_images(...            
        glob('CVPR08-SR/Data/Training', '*.bmp')), 12, 0.98);  
   img= load_images(glob('CVPR08-SR/Data/Training', '*.bmp'));
   n=randperm(size(img,1));
 %  images=[];
for i=1:10
    images{i}=img{n(i)};
end
  [plores phires] = collectSamplesScales(conf, images' , 12, 0.98);  
  plores=plores(:,randperm(size(plores,2)));
  patch=plores(:,1:10000);
% Here we provide the code to load natural image data into x.
% x will be a 144 * 10000 matrix, where the kth column x(:, k) corresponds to
% the raw image data from the kth 12x12 image patch sampled.
% You do not need to change the code below.
x =sampleIMAGESRAW();
x = plores(:,randperm(size(plores,2)));
x=plores(:,1:10000);
figure('name','Raw images');
randsel = randi(size(x,2),200,1); % A random selection of samples for visualization
display_network(x(:,randsel));
%%================================================================
%% Step 0b: Zero-mean the data (by row)
% You can make use of the mean and repmat/bsxfun functions.
% -------------------- YOUR CODE HERE -------------------- 

xmean = mean(x, 1);
x = bsxfun(@minus, x, xmean);
%x=(x-mean(x(:)))/std(x(:));
%%================================================================
%% Step 1a: Implement PCA to obtain xRot
% Implement PCA to obtain xRot, the matrix in which the data is expressed
% with respect to the eigenbasis of sigma, which is the matrix U.
% -------------------- YOUR CODE HERE -------------------- 
%xRot = zeros(size(x)); % You need to compute this
[nsamples, nfeatures] = size(x);
sigma = x * x' ./ nfeatures;
[U S V] = svd(sigma);
epsilon = 0.1;
ZCA=U*diag(1./sqrt(diag(S)+epsilon))*U';
x=ZCA*x;

xRot = U' * x;
display_network(xRot(:,randsel));
%%================================================================
%% Step 1b: Check your implementation of PCA
% The covariance matrix for the data expressed with respect to the basis U
% should be a diagonal matrix with non-zero entries only along the main
% diagonal. We will verify this here.
% Write code to compute the covariance matrix, covar. 
% When visualised as an image, you should see a straight line across the
% diagonal (non-zero entries) against a blue background (zero entries).
% -------------------- YOUR CODE HERE -------------------- 
covar = zeros(size(x, 1)); % You need to compute this
covar = xRot * xRot' ./ nfeatures;
% Visualise the covariance matrix. You should see a line across the
% diagonal against a blue background.
figure('name','Visualisation of covariance matrix');
imagesc(covar);
%%================================================================
%% Step 2: Find k, the number of components to retain
% Write code to determine k, the number of components to retain in order
% to retain at least 99% of the variance.
% -------------------- YOUR CODE HERE -------------------- 
k = 0; % Set k accordingly
lambda = sum(S, 2);
Sum = sum(lambda);
temp = Sum;
for i=size(lambda, 1):-1:1
lambda(i);
temp = temp - lambda(i);
if (temp / Sum < 0.99) 
k = i;
break; 
end
end
%%================================================================
%% Step 3: Implement PCA with dimension reduction
% Now that you have found k, you can reduce the dimension of the data by
% discarding the remaining dimensions. In this way, you can represent the
% data in k dimensions instead of the original 144, which will save you
% computational time when running learning algorithms on the reduced
% representation.
% 
% Following the dimension reduction, invert the PCA transformation to produce 
% the matrix xHat, the dimension-reduced data with respect to the original basis.
% Visualise the data and compare it to the raw data. You will observe that
% there is little loss due to throwing away the principal components that
% correspond to dimensions with low variation.
% -------------------- YOUR CODE HERE -------------------- 
xHat = zeros(size(x)); % You need to compute this
xHat = U(:, 1:k) * U(:, 1:k)' * x;
% Visualise the data, and compare it to the raw data
% You should observe that the raw and processed data are of comparable quality.
% For comparison, you may wish to generate a PCA reduced image which
% retains only 90% of the variance.
figure('name',['PCA processed images ',sprintf('(%d / %d dimensions)', k, size(x, 1)),'']);
display_network(xHat(:,randsel));
figure('name','Raw images');
display_network(x(:,randsel));
%%================================================================
%% Step 4a: Implement PCA with whitening and regularisation
% Implement PCA with whitening and regularisation to produce the matrix
% xPCAWhite. 
epsilon = 0.1;
xPCAWhite = zeros(size(x));
% -------------------- YOUR CODE HERE -------------------- 
xPCAWhite = diag(1./sqrt(diag(S) + epsilon)) * U' * x; 
%%================================================================
%% Step 4b: Check your implementation of PCA whitening 
% Check your implementation of PCA whitening with and without regularisation. 
% PCA whitening without regularisation results a covariance matrix 
% that is equal to the identity matrix. PCA whitening with regularisation
% results in a covariance matrix with diagonal entries starting close to 
% 1 and gradually becoming smaller. We will verify these properties here.
% Write code to compute the covariance matrix, covar. 
%
% Without regularisation (set epsilon to 0 or close to 0), 
% when visualised as an image, you should see a red line across the
% diagonal (one entries) against a blue background (zero entries).
% With regularisation, you should see a red line that slowly turns
% blue across the diagonal, corresponding to the one entries slowly
% becoming smaller.
% -------------------- YOUR CODE HERE -------------------- 
covar = xPCAWhite * xPCAWhite' ./ nfeatures;
% Visualise the covariance matrix. You should see a red line across the
% diagonal against a blue background.
figure('name','Visualisation of covariance matrix');
imagesc(covar);
%%================================================================
%% Step 5: Implement ZCA whitening
% Now implement ZCA whitening to produce the matrix xZCAWhite. 
% Visualise the data and compare it to the raw data. You should observe
% that whitening results in, among other things, enhanced edges.
xZCAWhite = zeros(size(x));
% -------------------- YOUR CODE HERE -------------------- 
xZCAWhite = U * xPCAWhite;
% Visualise the data, and compare it to the raw data.
% You should observe that the whitened images have enhanced edges.
figure('name','ZCA whitened images');
display_network(xZCAWhite(:,randsel));
figure('name','Raw images');
display_network(x(:,randsel));
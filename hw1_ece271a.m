%%%% Q5
% a)
% The priors PY (cheetah) 
p_c = 250/(1035+250) % 0.1946

% PY (grass) 
p_g = 1035/(1035+250) % 0.8054

%b) 
% plot the index of the second largest abs value 
% plot histogram of  PX|Y (x|grass) 
TrainsampleDCT_BG_abs = abs(TrainsampleDCT_BG);
[m,BG_i] = max(TrainsampleDCT_BG_abs,[],2);
%replace the max(s) by -Inf
TrainsampleDCT_BG_abs(bsxfun(@eq, TrainsampleDCT_BG_abs, m)) = -Inf;
%find the new max which is the second largest.
[m,BG_i] = max(TrainsampleDCT_BG_abs, [], 2);

histogram(BG_i);

% plot histogram of PX|Y (x|cheetah)
TrainsampleDCT_FG_abs = abs(TrainsampleDCT_FG);
[m,FG_i] = max(TrainsampleDCT_FG_abs,[],2);
%replace the max(s) by -Inf
TrainsampleDCT_FG_abs(bsxfun(@eq, TrainsampleDCT_FG_abs, m)) = -Inf;
%find the new max which is the second largest.
[m,FG_i] = max(TrainsampleDCT_FG_abs, [], 2);

histogram(FG_i);

%c)
IMG = imread('cheetah.bmp');
IMG = im2double(IMG);
IMG = IMG(1:248,1:264);

imagesc(IMG);
colormap(gray(255));


T = dctmtx(8);
dct = @(block_struct) T * block_struct.data * T';
%dct = @(block_struct) dct2(block_struct.data );

B = blockproc(IMG,[8 8],dct);
% read zigzag
zigzag = readtable('Zig-Zag Pattern.txt');
zigzag = table2array(zigzag)+1;

% 64 small matrices
B = abs(B)
N = 8 * ones(1,31)
M = 8 * ones(1,33)
C = mat2cell(B,N,M)
celldisp(C)

% test of the first row ONLY
%[m,i] = max(C{1,1}(:))
%replace the max(s) by -Inf
%C{1,1}(bsxfun(@eq, C{1,1}, m)) = -Inf
%find the new max which is the second largest.
%[m,i] = max(C{1,1}(:)) 

%zigzag(i)
feature = zeros(31,33)

for nrow = 1:33
    for ncol = 1:31
        [m,i] = max(C{ncol,nrow}(:));
        C{ncol,nrow}(bsxfun(@eq, C{ncol,nrow}, m)) = -Inf;
        [m,i] = max(C{ncol,nrow}(:));
        feature(ncol,nrow) = zigzag(i);
    end
end
    
    
% estimate the conditional probabilities
phat_BG = gamfit(BG_i);
phat_FG = gamfit(FG_i);
    
% fit the features for these two pdf
feature_vec = reshape(feature,[],1);
BG_fit = gampdf(feature_vec,4.2096,0.8816);
FG_fit = gampdf(feature_vec,2.0386,3.8183);

% joint probablities
p_x_g = BG_fit * p_g;
p_x_c = FG_fit * p_c;

% state variable Y: 1 = cheetah; 0 = grass
Y = p_x_c > p_x_g;
state_Y = reshape(Y,31,33);

% plot
figure
imagesc(state_Y);
colormap(gray(255));

% d)error rate
Img_mask = imread('cheetah_mask.bmp');
Img_mask = im2double(Img_mask);
% resize the image into 8x8
Img_mask = Img_mask(1:248,1:264);

imagesc(Img_mask);
colormap(gray(255));

Img_mask = abs(Img_mask);
mask = Img_mask > 0;

state_Y_1 = kron(state_Y,ones(8));
err = Img_mask == state_Y_1;
sum(err(:)==0)/(numel(Img_mask)); % 0.1729 error rate



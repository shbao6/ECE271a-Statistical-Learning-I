%%%%HW2
% a)
trainsample = load('TrainingSamplesDCT_8_new.mat');
zigzag = load('Zig-Zag Pattern.txt');
zigzag1 = zigzag + 1;
%cheetah(FG)

FG = trainsample.TrainsampleDCT_FG;
BG = trainsample.TrainsampleDCT_BG;
% histogram estimate
% MLE
PY_grass = 1053/(1053+250)
PY_cheetah = 250/(1053+250)


% b)

[m_FG,n] = size(FG);
[m_BG,n] = size(BG);
plot_FG = zeros(m_BG,n);
plot_BG = zeros(m_BG,n); % in order to take the same range

figure
for i = 1:n
    data_FG = FG(:,i);
    mu_FG = mean(data_FG);
    sigma_sqr_FG = (1/(m_FG-1))*sum((data_FG-mu_FG).^2);
    sigma_FG = sqrt(sigma_sqr_FG);
    
    data_BG = BG(:,i);
    mu_BG = mean(data_BG);
    sigma_sqr_BG = (1/(m_BG-1))*sum((data_BG-mu_BG).^2);
    sigma_BG = sqrt(sigma_sqr_BG);
    
    range = linspace(min(min(data_FG),min(data_BG)),max(max(data_FG),max(data_BG)),m_BG);
    plot_FG(:,i) = normpdf(range,mu_FG,sigma_FG);
    plot_BG(:,i) = normpdf(range,mu_BG,sigma_BG);
    
    subplot(8,8,i);
    plot(range,plot_FG(:,i),'-',range,plot_BG(:,i),'-.');
   
end

% Best 8 features: 1,21,24,25,32,40,41,42

i = [1,21,24,25,32,40,41,42]
figure(1)
      for p = 1:8
      subplot(4,2,p);
      plot(range,plot_FG(:,i(p)),'-',range,plot_BG(:,i(p)),'-.');
      title(sprintf('feature %d',i(p)));
      end
  legend('FG Distribution','BG Distribution')
  
  
% Worst 8 features: 2,3,4,5,59,62,63,64

i = [2,3,4,5,59,62,63,64]
figure(2)
      for p = 1:8
      subplot(4,2,p);
      plot(range,plot_FG(:,i(p)),'-',range,plot_BG(:,i(p)),'-.');
      title(sprintf('feature %d',i(p)));
      end
    legend('FG Distribution','BG Distribution')

% c) 
image1 = im2double(imread('cheetah.bmp'));
[R C] = size(image1);

% Dividing the cheetah image into 8x8 matrices

A = zeros([size(image1),64]);
for u = 1: R-7
    for v = 1: C-7
         dct = dct2(image1(u:u+7,v:v+7));
         for n = 1:64
             [c,d] = find(zigzag1 == n);
             A(u,v,n) = dct(c,d);
         end
    end
end

% 1) Using 64 features
% BG and FG
% MLE function for mean and covariance
BG_mle_mean = guassian_mle(BG);
BG_mle_cov = guassian_mle_cov(BG);
FG_mle_mean = guassian_mle(FG);
FG_mle_cov = guassian_mle_cov(FG);

% discriminant functions
B = zeros(size(image1));
for u =1: R-7
    for v = 1 : C-7
       % write a log multivariate guassian density function as log_pdf
        prob_C_X = log_pdf(squeeze(A(u,v,:)),FG_mle_mean',FG_mle_cov) + log(PY_cheetah); 
        prob_G_X = log_pdf(squeeze(A(u,v,:)),BG_mle_mean',BG_mle_cov) + log(PY_grass);
        
        B(u,v) = prob_G_X < prob_C_X; % state variable based on the decison rule 
        
    end
end


%My mask image
figure(3)
colormap(gray(255));
imshow(B)

% d)error rate
Img_mask = imread('cheetah_mask.bmp');
Img_mask = im2double(Img_mask);

imagesc(Img_mask);
colormap(gray(255));

% detection rate: P(g_x = cheetah|cheetah) 0.9272
cheetah_pos = Img_mask == 1;
sum(B(cheetah_pos)) / sum(Img_mask(cheetah_pos));
% false alarm rate: P(g_x = cheetah|grass) 0.0938
grass_pos = Img_mask == 0;
sum(B(grass_pos)) / sum(grass_pos,'all');
% error rate
0.0938 * 0.8082 + (1-0.9272) * 0.1919; % 0.0898


% 2) Using 8 features
best = [1,21,24,25,32,40,41,42];
% BG and FG
% MLE mean and covariance for the best features
BG_mle_mean_best = guassian_mle(BG(:,(best)));
BG_mle_cov_best = guassian_mle_cov(BG(:,(best)));
FG_mle_mean_best = guassian_mle(FG(:,(best)));
FG_mle_cov_best = guassian_mle_cov(FG(:,(best)));

% discriminant functions
A_best = A(:,:,best);
B_best = zeros(size(image1));

for u =1: R-7
    for v = 1 : C-7
       % write a log multivariate guassian density function as log_pdf
        prob_C_X = log_pdf(squeeze(A_best(u,v,:)),FG_mle_mean_best',FG_mle_cov_best) + log(PY_cheetah); 
        prob_G_X = log_pdf(squeeze(A_best(u,v,:)),BG_mle_mean_best',BG_mle_cov_best) + log(PY_grass);
        
        B_best(u,v) = prob_G_X < prob_C_X; % state variable based on the decison rule 
        
    end
end


%My mask image
figure(4)
colormap(gray(255));
imshow(B_best)


% detection rate: P(g_x = cheetah|cheetah) 0.9001
cheetah_pos = Img_mask == 1;
sum(B_best(cheetah_pos)) / sum(Img_mask(cheetah_pos));
% false alarm rate: P(g_x = cheetah|grass) 0.0420
grass_pos = Img_mask == 0;
sum(B_best(grass_pos)) / sum(grass_pos,'all');
% error rate
0.0420 * 0.8082 + (1-0.9001) * 0.1919; % 0.0531



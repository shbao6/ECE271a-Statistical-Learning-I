%%%%EM
% a)
trainsample = load('TrainingSamplesDCT_8_new.mat');
zigzag = load('Zig-Zag Pattern.txt');
zigzag1 = zigzag + 1;
%cheetah(FG)

FG = trainsample.TrainsampleDCT_FG;
BG = trainsample.TrainsampleDCT_BG;

% Read the cheetah image into 8x*8 matrices
image1 = im2double(imread('cheetah.bmp'));
[R C] = size(image1);

A=zeros((R-7)*(C-7),64);
index=1;
for i=1:R-7
    for j=1:C-7
        DCT=dct2(image1(i:i+7,j:j+7));
        DCT_64(zigzag1)=DCT;  
        A(index,:)=DCT_64;
        index=index+1;
    end
end

% Parameter for EM
dimen=64;
c=8;

% Choose 5 random mixtures of 8 components
    
    pi_BG = cell(5,1);
    mu_BG = cell(5,1);
    var_BG= cell(5,1);
    pi_FG = cell(5,1);
    mu_FG = cell(5,1);
    var_FG= cell(5,1);
    
% EM for background
    for bg = 1:1:5
        bg
        % paramter initialization
        pi_i = rand(1,c);
        pi_i = pi_i / sum(pi_i); % regularization: sum up to 1
        mu_i = rand(c, dimen);
        var_i = rand(c,dimen);
        var_i(var_i < 0.0001) = 0.0001; % correct for variance matrix near to 0
        [mu,var,prior] = EM(dimen,mu_i,var_i,pi_i,c,BG);
      
        % save the EM results
        pi_BG{bg} = prior;
        mu_BG{bg} = mu;
        var_BG{bg} = var;
    end
    

    % EM for foreground
    for fg = 1:1:5
        fg
        % parameter initialization
        pi_i = rand(1,c);
        pi_i = pi_i / sum(pi_i);
        mu_i = rand(c, dimen);
        var_i = rand(c,dimen);
        var_i(var_i < 0.0001) = 0.0001;
        
        [mu,var,prior] = EM (dimen,mu_i,var_i,pi_i,c,FG);
        
        % save the EM results
        pi_FG{fg} = prior;
        mu_FG{fg} = mu;
        var_FG{fg} = var;
    end
    
        
Img_mask = imread('cheetah_mask.bmp');
Img_mask = im2double(Img_mask);
       
% set the prior
[R_F,C_F] = size(FG);
FG_index=zeros(1,R_F);
[R_B,C_B] = size(BG);
BG_index=zeros(1,R_B);
% between (0,1) and sums to 1
Prior_FG=R_F/(R_F+R_B);    
Prior_BG=R_B/(R_F+R_B);

row = 1;
col = 1;
error_rate = zeros(25,11);

for dim = [1,2,4,8,16,24,32,40,48,56,64]
    for n1=1:1:5
        for n2=1:1:5
            
            predict=zeros(1,(C-7)*(R-7));    
            
            for index=1:(C-7)*(R-7)
                Prob_BG=0;
                for i=1:c
                    Prob_BG = Prob_BG + mvnpdf(A(index,1:dim),mu_BG{n1}(i,1:dim),diag(var_BG{n1}(i,1:dim)))* pi_BG{n1}(i);    
                end
                Prob_FG=0;
                for i=1:c
                    Prob_FG = Prob_FG + mvnpdf(A(index,1:dim),mu_FG{n2}(i,1:dim),diag(var_FG{n2}(i,1:dim)))* pi_FG{n2}(i);    
                end
                
                if (Prob_BG * Prior_BG < Prob_FG *Prior_FG) % BDR 
                    predict(1,index)=1;
                end

            end
            
            predict_2d=reshape(predict,[C-7,R-7]);
            predict_2d=predict_2d';
            
            result=zeros(255,270);
            result(1:R-7,1:C-7)=predict_2d;
            
            error_rate(row,col) = Count_Error(result,Img_mask);
            row = row + 1;
        end
    end
    col = col + 1;
end

figure;
imagesc(predict_2d);
colormap('gray') 

% plots
for i=1:1:25
    if mod(i,5)==1
        figure
    end
    plot([1,2,4,8,16,24,32,40,48,56,64], new_mat(i,:),'*-')
    
    hold on
    
    if mod(i,5)==0
        grid on
        legend('FG mixture 1','FG mixture 2','FG mixture 3','FG mixture 4','FG mixture 5')
        title(['Prob. Of Error VS. number of features (BackGround No.',num2str(floor(i/5)),' )'])
    end
end


function [mu_final,var_final,pi_final]=EM(dimen,mu_i,var_i,pi_i,c,data)

        [N, col] = size(data);
        
        % with reference to class slide page 24 and the following slides
        % h matrix
        h = zeros(N, c);
        % iterating 200 times
        pi_iter = pi_i;
        mu_iter = mu_i;
        var_iter = var_i;
        
        for times = 1 : 200   
            % E-step: create h matrix
            for i = 1 : N
                for j = 1:c
                    h(i, j) = pi_i(j)* mvnpdf(data(i, :), mu_i(j, :), diag(var_i(j, :))) ;
                end
                h(i, :) = h(i, :) / sum(h(i, :)); % regularization
            end

             % M-step: calculate new parameters
             % refer to fomulas on page 27 (EM2)
            for j = 1:c
                sum_h = sum(h(:, j));
                mu_iter(j, :) = h(:, j)'*data(:, :)./sum_h;
                pi_iter(j) = sum_h / N;
                
                sum_i=0;

                for i=1: size(data,1)
                   sum_i=sum_i+h(i,j)*(data(i,:)-mu_iter(j,:)).^2;
                end
                var_iter(j, :) = sum_i / sum_h;
                var_iter(var_iter < 0.0001) = 0.0001;
            end
            
            % when the change is too small, then break
            if all(abs(mu_iter - mu_i)./abs(mu_i) < 0.0001)
                break;
            end
            
            mu_i = mu_iter;
            pi_i = pi_iter;
            var_i = var_iter;
            
        end

        pi_final = pi_i;
        mu_final = mu_i;
        var_final = var_i;
   
end

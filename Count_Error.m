function [error_rate] = Count_Error(predict,Img_mask)
Img_mask = abs(Img_mask);
mask = Img_mask > 0;

err = Img_mask == predict;

error_rate = sum(err(:)==0)/(numel(Img_mask));
end


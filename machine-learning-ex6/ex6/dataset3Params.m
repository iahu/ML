function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.03;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

mini_error = 1;
steps = 5;
idx_i = 1;
idx_j = 1;

for i=1:6;
    for j=1:6;
        model = svmTrain(X, y, steps^i * C,...
            @(x1, x2) gaussianKernel(x1, x2, steps^j * steps^j * sigma));
        %model = svmTrain(X, y, steps^i * C, @gaussianKernel, 1e-3, 20);
        predictions = svmPredict(model, Xval);
        m = mean(double(predictions ~= yval));
        if m < mini_error;
            mini_error = m;
            idx_i = i;
            idx_j = j;
        end
    end
end

C = C * steps^idx_i
sigma = sigma * steps^idx_j




% =========================================================================

end

function loss = LossFcn( alfa , theta, data)
% Loss function for calibration calculating the sum of the squared residuals
% input: calibrated preference parameter, alfa
%        predeteremined production parameter, theta
%        data inputs (structure), data
% output: sum of the squared residuals

% Calculate the model's solution for hours worked for a given alpha
hmodel9396 = 100.*(1-theta ) ./ ( alfa./(1-data.tau9396) .* data.c2y9396  + (1-theta ) ) ;
hmodel7074 = 100.*(1-theta ) ./ ( alfa./(1-data.tau7074) .* data.c2y7074  + (1-theta ) ) ;
% Calculate the sum of the squared residuals
loss = sum( ( data.hdata9396 - hmodel9396 ).^2) + sum ( ( data.hdata7074 - hmodel7074 ).^2 );
% Note that .* multiplies one vector by another component-by-component.
end
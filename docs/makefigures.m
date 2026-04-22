function [ ] = makefigures ( num, hdata , hmodel , country_name, heading )
% Make a graph plotting the data versus model for the 7 countries
% Input: Num is the figure number 
%        hdata is the data vector of hours worked for the countries
%        hmodel is the same thing for the model
%        heading is a character string giving the title for the graph
% Note that output for the function is empty , or [ ] 
figure(num)
plot ( hdata , hmodel , 'rd' , ...
       hdata , hdata , 'k-' ,...
       'LineWidth',1,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',5)
% The plot command asks MATLAB to plot hdata versus hmodel using a red symbol.
% There is no line for the first plot.
% The second plot shows the 45 degree line, using a straight line.
title(heading ,'fontsize',22,'fontname','times')
xlabel( 'Data', 'fontsize',22,'fontname','times')
ylabel( 'Model', 'fontsize',22,'fontname','times')
for i = 1:7
    text( hdata(i) , hmodel(i) , country_name(i) );
end
end
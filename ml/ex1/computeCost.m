function J = computeCosit(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values


% You need to return the following variables correctly 


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
J = 0;
m = length(y); % number of training examples
% for i=1:m
%     J=J+1/(2*m)*power((X(i,:)*theta-y(i)),2);
% end
J=sum(power((X*theta-y),2),'all')/(2*m);
% =========================================================================

end

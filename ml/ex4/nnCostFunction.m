function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                               
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));


Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

             
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
%hidden_layer_size*(input_layer_size + 1)
Theta2_grad = zeros(size(Theta2));
%num_labels*(hidden_layer_size + 1)

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
% X row martix
Theta1=Theta1';
Theta2=Theta2';
X=[ones(size(X,1),1),X];
% disp(size(X));
% disp(size(Theta1));
temp=sigmoid(X*Theta1);
temp=[ones(size(temp,1),1),temp];
h_theta_X=sigmoid(temp*Theta2);
%transform y
temp_y=zeros(num_labels,size(X,1));
for i=1:size(temp_y,2)
    temp_y(y(i),i)=1;
end
y=temp_y';%实际的output
J=-1/m*sum(y.*log(h_theta_X)+(1-y).*log(1-h_theta_X),'all');

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
delta3=h_theta_X-y;
delta2=(Theta2'*delta3).*sigmoidGradient(temp*Theta2);
delta1=(Theta1'*delta2).*sigmoidGradient(X*Theta1);







% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
J=J+lambda/(2*m)*(sum(Theta1.*Theta1,'all')+sum(Theta2.*Theta2,'all')-(sum(Theta1(1,:).*Theta1(1,:),'all')+sum(Theta2(1,:).*Theta2(1,:),'all')));



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

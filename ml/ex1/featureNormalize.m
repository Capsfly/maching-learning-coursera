function [X_norm, mu , sigma] = featureNormalize(X) 
    
%     %没有AC的
    mu=mean(X);
    sigma=std(X);
    X_norm=(X-mu)./repmat(sigma,size(X,1),1);
%     

%     mu=mean(X);
%     sigma=std(X);
%     X_norm=(X-repmat(mu,size(X,1),1))./repmat(sigma,size(X,1),1);



   
% ============================================================

end

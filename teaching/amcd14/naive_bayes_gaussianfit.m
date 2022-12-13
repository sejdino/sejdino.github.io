%Naive Bayes classifier - gaussian fit along each dimension
%last update: February 2013

%Xtrn --- training data
%ytrn --- labels for training data
%Xtst --- testing data
%ytst -- correct labels for testing data

function [ypred] = naive_bayes_gaussianfit(Xtrn, ytrn, Xtst)

classes = unique(ytrn);
K = length(classes);

[ntrn, D] = size(Xtrn);
[ntst, D] = size(Xtst);
mean_est = zeros(K,D);
var_est = zeros(K,D);

for k=1:K
    % Priors
    prior(k) = mean(ytrn==classes(k));

    mean_est(k,:)=mean(Xtrn(ytrn==classes(k),:));
    
    for dd=1:D
        
        var_est(k,dd)=var(Xtrn(ytrn==classes(k),dd));
    end
    
    % Calculate Log Likelihood
    LL = sum( -0.5*log((2*pi).*repmat(var_est(k,:),ntst,1)) - ...
        ((Xtst-repmat(mean_est(k,:),ntst,1)).^2 ) ./ (2*repmat(var_est(k,:),ntst,1)), 2 );
           
    % Calculate Log Posterior
    LP(:,k) = LL + (log(prior(k))*ones(ntst,1));
end
[max_post, t_pred] = max(LP,[],2);
Posterior = exp(LP);
Posterior = Posterior./repmat(sum(Posterior,2),1,K);
Posterior(isnan(Posterior)) = 0.5;


for k=1:K
    subplot(K,1,k)
    plot(Posterior(1:ntst,k),'x');
    title(sprintf('Predictive Posterior for class %d',k));
end

ypred = classes(t_pred);

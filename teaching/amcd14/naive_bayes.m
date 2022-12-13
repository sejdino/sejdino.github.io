%Naive Bayes classifier - text example
%last update: February 2013

%Xtrn --- training data
%ytrn --- labels for training data
%Xtst --- testing data
%ytst -- correct labels for testing data

function [ypred] = naive_bayes(Xtrn, ytrn, Xtst)

%the vector of unique class labels
classes = unique(ytrn);

%the number of classes
K = length(classes);

[ntrn, D] = size(Xtrn);
[ntst, D] = size(Xtst);

prior=zeros(K,1);
p=zeros(K,D);
LP=zeros(ntst,K);

for k=1:K %for each class k
    % Priors (which are just empirical frequencies in the training set!)
    prior(k) = mean(ytrn==classes(k));

    %count the documents in class k that contain each word
    %note that we added +1 to deal with words that do not appear in
    %the training set
    p(k,:) = (sum(Xtrn(ytrn==classes(k),:)>0) + 1);

           
    %log-likelihoods for test data
    LL = (Xtst>0)*(log(p(k,:))-log(sum(p(k,:))))';
           
           
    % Calculate Log Posterior (discriminant function)
    LP(:,k) = LL + (log(prior(k))*ones(ntst,1));
    aux=max(LP(:,k));
    LP(:,k)=LP(:,k)-aux;
end

%take maximums of Log Posteriors to predict classes
[max_post, t_pred] = max(LP,[],2);

%Actual predictive posterior - could lead to some numerical instability
%because these are typically very small numbers
%aux=max(LP);
%disp(LP);
Posterior = exp(LP);
Posterior = Posterior./repmat(sum(Posterior,2),1,K);
if any(isnan(Posterior))
    Posterior(isnan(Posterior)) = 0.5;
end


for k=1:K
    subplot(K,1,k)
    plot(Posterior(1:ntst,k),'x');
    title(sprintf('Predictive Posterior for class %d',k));
end

%predicted class labels
ypred = classes(t_pred);


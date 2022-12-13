%Classification examples with mixtures of gaussians
%DS (February 2013)


%% generate data
clear all
close all
%means of the positive examples
mux1=[1; -1];

%means of the negative examples
muy1=[-1; 1];



%covariance defined below via eigendecomposition C=U*inv(S)*U'

%U controls the eigenvectors of the covariance matrix
theta=-pi/4;
U   =  [cos(theta) -sin(theta);...
        sin(theta)  cos(theta)];
    
%the inverse of eigenvalues along each eigenvector
s1=2; s2=0.01;
S=[s1,0;0,s2];

%number of data examples
nx=400;
ny=400;

%
xx =  [(sqrt(S)*U')\randn(2,nx) + mux1*ones(1,nx)]';
   
yy =  [(sqrt(S)*U')\randn(2,ny) + muy1*ones(1,ny)]';

   
%add noise_dim dimensions of "noise" 
noise_dim=0; %increase to make the problem harder

xx=[xx randn(nx,noise_dim)];
yy=[yy randn(nx,noise_dim)];

%plot the data (first two dimensions)
figure(1)
plot(xx(:,1),xx(:,2),'r*',yy(:,1),yy(:,2),'bo');
title('all data')

%%
%if you set noise_dim>0, try plotting other dimensions in your data
% figure(2)
% plot(xx(:,3),xx(:,4),'r*',yy(:,3),yy(:,4),'bo');
%%
TD=[xx; yy]';
Target=[ones(1,400) 2*ones(1,400)];

perms = [randperm(400), randperm(400)+400];

%how many training examples from each class
howmany1=50;
howmany2=50;

idxstrn = [perms(1:howmany1), perms(401:(400+howmany2))];
idxstst = [perms(howmany1+1:400), perms((400+howmany2+1):end)];

Xtrn = TD(:,idxstrn)';
ytrn = Target(idxstrn)';
vals = unique(ytrn);
ytrn(ytrn==vals(1)) = -1;
ytrn(ytrn==vals(2)) = 1;
Xtst = TD(:,idxstst)';
ytst = Target(idxstst)';
vals = unique(ytst);
ytst(ytst==vals(1)) = -1;
ytst(ytst==vals(2)) = 1;


%% plot training data (first 2 dimensions)
figure(3)
plot(Xtrn(ytrn==-1,1),Xtrn(ytrn==-1,2),'r*',Xtrn(ytrn==1,1),Xtrn(ytrn==1,2),'bo');
title('training examples')


%% run naive bayes
figure(4)
[ypred_NB] = naive_bayes_gaussianfit(Xtrn, ytrn, Xtst);
err_NB=mean(ypred_NB~=ytst);
display(sprintf('Naive Bayes (Gaussian fit)): %.2f', 100*(1-err_NB)));

%plot predictions on test data
figure(5)
plot(Xtst(ypred_NB==-1,1),Xtst(ypred_NB==-1,2),'r*',Xtst(ypred_NB==1,1),Xtst(ypred_NB==1,2),'bo');
title('predictions on test data (naive bayes)');

%% run svm with a linear kernel 



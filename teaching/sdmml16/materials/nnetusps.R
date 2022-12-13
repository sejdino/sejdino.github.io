library(nnet)

trainx <- read.table("usps_trainx.data")
trainy <- read.table("usps_trainy.data")
testx  <- read.table("usps_testx.data")
testy  <- read.table("usps_testy.data")

image(matrix(as.matrix(trainx[500,]),16,16),col=grey(seq(0,1,length=256)))
trainy[500,]

ntr<- dim(trainx)[1]
trainx <- trainx / 256
testx  <- testx  / 256
trainy <- as.factor(trainy[,1])
testy  <- as.factor(testy[,1])

subsample_idx <- seq(1,ntr,by=10)
trainx <- trainx[subsample_idx,]
trainy <- trainy[subsample_idx]

#regularization parameter
decay=0.01

#train a net for 10 iterations -- softmax=TRUE for log-linear model 
net <- nnet(trainx,class.ind(trainy),size=10,softmax=TRUE,
        MaxNWts=10000,maxit=10,decay=decay)
trainp <- predict(net,trainx,type="class")
testp  <- predict(net,testx ,type="class")
trainerr10 <- sum(trainy!=trainp)/length(trainp)
testerr10  <- sum(testy!=testp)/length(testp)
print(paste("..train error after 10 iterations:",trainerr10))
print(paste("..test error after 10 iterations:",testerr10))


#train for 10 more iterations; passing previously learned weights as Wts 
net <- nnet(trainx,class.ind(trainy),size=10,softmax=TRUE,
        MaxNWts=10000,maxit=10,decay=decay,Wts=net$wts)
trainp <- predict(net,trainx,type="class")
testp  <- predict(net,testx ,type="class")
trainerr20 <- sum(trainy!=trainp)/length(trainp)
testerr20  <- sum(testy!=testp)/length(testp)
print(paste("..train error after 20 iterations:",trainerr20))
print(paste("..test error after 20 iterations:",testerr20))


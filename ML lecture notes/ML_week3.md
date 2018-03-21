# Machine Learning: Week 3

- Math prep

  analsis, algebra (matrix algebra), optimization (convex optimization), probability, statistics, mathmatical logics

Eg. 微分几何—流形学习，测地线



----

## [textbook][1] Chapter 1 Introduction

### Basic Terms

- __测试(test)__: 用模型用来预测/计算
- __instance__ —$x^m$, __example__— $(x^m ,y^m)$
- __sample__?
- __Attribute space__ — expanded by attribues. __sample space__, __input space__ — pretty similar
- __feature vector__ — every input of instance
- __Output space__



- __hypothesis__ — waht the model learns
- __ground-truth__, __learner__



- __tasks__: classification, regression for eg.

binary regression, multi-regression



- __supervised learning__: classification(for eg.)
- __unsupervised learning__: cluster(for eg.)
- __reinforcement learning__



- unseen instance
- unkown distribution
- __i.i.d__: basic assumption, so we could treat data sampled from a r.v. $X$, for probabilistic modelling/

but "indepdency" could be relaxed some times (user preferences, for eg.)

- generalization



### Hypothesis space

Size = $\sum_{i=1}^{\text{number of features} } (n_i+1)+1$ , 1 for * 通配符， 1 for empty set

__version space__ ={hypothesis consistent with training set| hypothesis $\in$ hypothesis space}



### Inductive bias

learning algorithm's bias/preference for some type of hypothesis

every learning algorithm must have its bias.

choose the simplest one (model)

trade-off: simplicity v.s. consistency with training set(empricial data)

which algorithm is better — NFL theorem (a brief proof given, see [textbook][1] P8-9)



-----

## [textbook][1] Chapter 2 Model Evalutation and Selection

a good model — generalizes well

### empirical error and overfitting

- Generalization error: error in "unseen/future" test
- training error: error in training set



- overfitting v.s. Underfitting



3 key questions in model selection

- how to obtain the results of testing — __evaluation methods__
- how to evaluate the perfomance — __performance measure__
- how to judge the substantial differences(reasons: random errors / nosises for eg.) — comparison test? (比较检验)

### evaluation methods

keep mutual exclusive between training set and testing set

1. __Hold-out__

comments:

- consistency of distribution (stratified sampling, for example)
- To-do
- To-do

in real life:

(training, test) — select the best-performing model

what we give to customer in the end — the selected best-performning model trained after utilizing the whole dataset

statistical tests (more than 30 samples)

2. __K-fold cross validtion__

   e.g. 10x10 (most common) , k=m — leave-one-out (LOO)

   no free lunch, LOO does not guarantee better accuracy. 

   E.g. 50 girls, 50 boys, predict a new peroson's gender. LOO- accuracy 0, k-fold, 50%

3. __Bootstrapping__

out-of-bag estimation

pros:

at the same scale with original sample set

some changes in distribution

----

Reference

[1]: http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/MLbook2016.htm "Machine Learning (watermelon book) by Prof Zhihua Zhou"
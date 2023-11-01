library(tidyverse)
library(tidymodels)
library(vroom)
#library(ggmosaic)
library(embed)
#library(doParallel)

## Change working directory

setwd(dir = "~/School/F2023/STAT348/STAT348/AmazonEmployeeAccess")



########################## EDA ################################################



at <- vroom("train.csv") %>% 
  mutate(ACTION = as.factor(ACTION))

atest <- vroom("test.csv")

my_recipe <- recipe(ACTION ~ ., data=at) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value
  step_dummy(all_nominal_predictors()) %>% # dummy variable encoding
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(target_var)) #target encoding
# also step_lencode_glm() and step_lencode_bayes()


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = at)

# ggplot(data=baked) + geom_mosaic(aes(x=baked$MGR_ID_other, fill=ACTION))



############################## Logistic Regression #############################



my_mod_l <- logistic_reg() %>% #Type of model
  set_engine("glm")

amazon_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(my_mod_l) %>%
fit(data = at) # Fit the workflow

amazon_predictions <- predict(amazon_workflow,
                              new_data=atest,
                              type="prob") # "class" or "prob" (see doc)

amazon_predictions <- amazon_predictions %>%
  mutate(Action = ifelse(.pred_1 >= 0.7, 1, 0))

amazon_predictions <- amazon_predictions %>% #This predicts
  bind_cols(., atest) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(Action=.pred_1)

vroom_write(x=amazon_predictions, file="logistic.csv", delim=",")



######################## Penalized Logistic Regression ########################



my_recipe <- recipe(ACTION ~ ., data=at) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

my_mod_pl <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_pl)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(at, v = 5, repeats=1)

## Run the CV
CV_results <- amazon_workflow %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(roc_auc)) #Or leave metrics NULL

# Find Best Tuning Parameters1
best_tune_pl <- CV_results %>%
                select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <- amazon_workflow %>%
            finalize_workflow(best_tune_pl) %>%
            fit(data=at)

## Predict

amazon_predictions_pl <- final_wf %>% predict(new_data=atest,
                              type="prob")

amazon_predictions_pl <- amazon_predictions_pl %>% #This predicts
  bind_cols(., atest) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(Action=.pred_1)

vroom_write(x=amazon_predictions_pl, file="Plogistic.csv", delim=",")

save(file="./Plogistic.csv", list=c("best_tune_pl", "amazon_predictions_pl"))



################################## RF Binary ###################################

## recipe

library(doParallel)

num_cores <- parallel::detectCores()

cl <- makePSOCKcluster(num_cores)

registerDoParallel(cl)

my_recipe <- recipe(ACTION ~ ., data=at) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

## Create a workflow with model & recipe

my_mod_rf <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
                      set_engine("ranger") %>%
                      set_mode("classification")


amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_rf)

## Set up grid of tuning values

tuning_grid <- grid_regular(mtry(range = c(1,(ncol(at)-1))),
                            min_n(),
                            levels = 3)

## Set up K-fold CV

folds <- vfold_cv(at, v = 3, repeats=1)

## Find best tuning parameters

CV_results <- amazon_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

best_tune_rf <- CV_results %>%
  select_best("roc_auc")

## Finalize workflow and predict

final_wf <- amazon_workflow %>%
  finalize_workflow(best_tune_rf) %>%
  fit(data=at)

amazon_predictions_rf <- final_wf %>% predict(new_data=atest,
                                              type="prob")

ap_rf <- amazon_predictions_rf %>% #This predicts
  bind_cols(., atest) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(Action=.pred_1)

stopCluster(cl)

vroom_write(x=ap_rf, file="rfamazon.csv", delim=",")

write.csv(ap_rf, file = "amazonrf3.csv", quote = FALSE, row.names = FALSE)

save(file="./rfamazon.csv", list=c("best_tune_rf", "amazon_predictions_rf"))

############################### Naive Bayes ####################################

library(discrim)
library(naivebayes)

## Recipe

my_recipe <- recipe(ACTION ~ ., data=at) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

## model and workflow

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naive bayes eng

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here

## Grid of values to tune over
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(at, v = 5, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

# Find Best Tuning Parameters1
best_tune_nb <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <- nb_wf %>%
  finalize_workflow(best_tune_nb) %>%
  fit(data=at)


## Predict

#predict(final_wf, new_data=atest, type="prob")

amazon_predictions_nb <- final_wf %>% predict(new_data=atest,
                                              type="prob")

amazon_predictions_nb <- amazon_predictions_nb %>% #This predicts
  bind_cols(., atest) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(Action=.pred_1)

vroom_write(x=amazon_predictions_nb, file="nbPreds.csv", delim=",")



#################################### KNN ######################################



library(doParallel)
library(kknn)

num_cores <- parallel::detectCores()

cl <- makePSOCKcluster(num_cores)

registerDoParallel(cl)

## Recipe

my_recipe <- recipe(ACTION ~ ., data=at) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize()

## model and workflow

knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune4
             set_mode("classification") %>%
             set_engine("kknn")

knn_wf <- workflow() %>%
          add_recipe(my_recipe) %>%
          add_model(knn_model)

## Grid of values to tune over
tuning_grid <- grid_regular(neighbors(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(at, v = 5, repeats=1)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

# Find Best Tuning Parameters1
best_tune_knn <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <- knn_wf %>%
  finalize_workflow(best_tune_knn) %>%
  fit(data=at)

## Predict

amazon_predictions_knn <- final_wf %>% predict(new_data=atest,
                                              type="prob")

amazon_predictions_knn <- amazon_predictions_knn %>% #This predicts
  bind_cols(., atest) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(Action=.pred_1)

stopCluster(cl)

vroom_write(x=amazon_predictions_knn, file="knnPreds2.csv", delim=",")



################################### PCA ########################################



library(doParallel)
library(kknn)

num_cores <- parallel::detectCores()

cl <- makePSOCKcluster(num_cores)

registerDoParallel(cl)

my_recipe <- recipe(ACTION ~ ., data=at) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), threshold = 0.8)

## model and workflow

knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune4
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

## Grid of values to tune over
tuning_grid <- grid_regular(neighbors(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(at, v = 5, repeats=1)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

# Find Best Tuning Parameters1
best_tune_knn <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <- knn_wf %>%
  finalize_workflow(best_tune_knn) %>%
  fit(data=at)

## Predict

amazon_predictions_knn <- final_wf %>% predict(new_data=atest,
                                               type="prob")

amazon_predictions_knn <- amazon_predictions_knn %>% #This predicts
  bind_cols(., atest) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(Action=.pred_1)

stopCluster(cl)

vroom_write(x=amazon_predictions_knn, file="knnPreds3.csv", delim=",")


## Naive Bayes PCA



library(discrim)
library(naivebayes)

## Recipe

my_recipe <- recipe(ACTION ~ ., data=at) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  #step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), threshold = 0.9)

## model and workflow

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naive bayes eng

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here

## Grid of values to tune over
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(at, v = 5, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

# Find Best Tuning Parameters1
best_tune_nb <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <- nb_wf %>%
  finalize_workflow(best_tune_nb) %>%
  fit(data=at)


## Predict

#predict(final_wf, new_data=atest, type="prob")

amazon_predictions_nb <- final_wf %>% predict(new_data=atest,
                                              type="prob")

amazon_predictions_nb <- amazon_predictions_nb %>% #This predicts
  bind_cols(., atest) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(Action=.pred_1)

vroom_write(x=amazon_predictions_nb, file="nbPredsPCA.csv", delim=",")



################################### SVM ########################################



library(doParallel)

num_cores <- parallel::detectCores()

cl <- makePSOCKcluster(num_cores)

registerDoParallel(cl)

my_recipe <- recipe(ACTION ~ ., data=at) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), threshold = 0.95)

## SVM models
# svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
# set_engine("kernlab")

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
set_engine("kernlab")

# svmLinear <- svm_linear(cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
# set_engine("kernlab")

## Fit or Tune Model HERE

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)

## Tune smoothness and Laplace here

## Grid of values to tune over
tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 2) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(at, v = 2, repeats=1)

## Run the CV
CV_results <- svm_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

# Find Best Tuning Parameters1
best_tune_svm <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <- svm_wf %>%
  finalize_workflow(best_tune_svm) %>%
  fit(data=at)


## Predict
amazon_predictions_svm <- final_wf %>% predict(new_data=atest,
                                              type="prob")

amazon_predictions_svm <- amazon_predictions_svm %>% #This predicts
  bind_cols(., atest) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(Action=.pred_1)

vroom_write(x=amazon_predictions_svm, file="predsSVM.csv", delim=",")

stopCluster(cl)



################################ Balancing Data ################################



library(doParallel)

num_cores <- parallel::detectCores()

cl <- makePSOCKcluster(num_cores)

registerDoParallel(cl)

library(tidymodels)
library(themis) # for smote

my_recipe <- recipe(ACTION ~ ., data=at) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_upsample()

my_mod_rf <- rand_forest(mtry = tune(),
                         min_n=tune(),
                         trees=700) %>%
  set_engine("ranger") %>%
  set_mode("classification")


amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_rf)

## Set up grid of tuning values

tuning_grid <- grid_regular(mtry(range = c(1,(ncol(at)-1))),
                            min_n(),
                            levels = 3)

## Set up K-fold CV

folds <- vfold_cv(at, v = 3, repeats=1)

## Find best tuning parameters

CV_results <- amazon_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

best_tune_rf <- CV_results %>%
  select_best("roc_auc")

## Finalize workflow and predict

final_wf <- amazon_workflow %>%
  finalize_workflow(best_tune_rf) %>%
  fit(data=at)

amazon_predictions_rf <- final_wf %>% predict(new_data=atest,
                                              type="prob")

ap_rf <- amazon_predictions_rf %>% #This predicts
  bind_cols(., atest) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(Action=.pred_1)

vroom_write(x=ap_rf, file="rfamazon.csv", delim=",")

stopCluster(cl)

write.csv(ap_rf, file = "amazonrf3bd.csv", quote = FALSE, row.names = FALSE)



## Logistic



my_recipe <- recipe(ACTION ~ ., data=at) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_downsample()

my_mod_l <- logistic_reg() %>% #Type of model
  set_engine("glm")

amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_l) %>%
  fit(data = at) # Fit the workflow

amazon_predictions <- predict(amazon_workflow,
                              new_data=atest,
                              type="prob") # "class" or "prob" (see doc)

amazon_predictions <- amazon_predictions %>%
  mutate(Action = ifelse(.pred_1 >= 0.9, 1, 0))

amazon_predictions <- amazon_predictions %>% #This predicts
  bind_cols(., atest) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(Action=.pred_1)

vroom_write(x=amazon_predictions, file="logistic.csv", delim=",")



## Penalized



my_mod_pl <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_pl)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(at, v = 5, repeats=1)

## Run the CV
CV_results <- amazon_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

# Find Best Tuning Parameters1
best_tune_pl <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <- amazon_workflow %>%
  finalize_workflow(best_tune_pl) %>%
  fit(data=at)

## Predict

amazon_predictions_pl <- final_wf %>% predict(new_data=atest,
                                              type="prob")

amazon_predictions_pl <- amazon_predictions_pl %>% #This predicts
  bind_cols(., atest) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(Action=.pred_1)

vroom_write(x=amazon_predictions_pl, file="Plogistic.csv", delim=",")



## Naive Bayes



library(discrim)
library(naivebayes)
library(themis)

## Recipe

my_recipe <- recipe(ACTION ~ ., data=at) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_downsample()

## model and workflow

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naive bayes eng

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here

## Grid of values to tune over
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(at, v = 5, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

# Find Best Tuning Parameters1
best_tune_nb <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <- nb_wf %>%
  finalize_workflow(best_tune_nb) %>%
  fit(data=at)


## Predict

#predict(final_wf, new_data=atest, type="prob")

amazon_predictions_nb <- final_wf %>% predict(new_data=atest,
                                              type="prob")

amazon_predictions_nb <- amazon_predictions_nb %>% #This predicts
  bind_cols(., atest) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(Action=.pred_1)

vroom_write(x=amazon_predictions_nb, file="nbPreds.csv", delim=",")



## KNN



library(doParallel)
library(kknn)

num_cores <- parallel::detectCores()

cl <- makePSOCKcluster(num_cores)

registerDoParallel(cl)

## Recipe

my_recipe <- recipe(ACTION ~ ., data=at) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_downsample()

## model and workflow

knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune4
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

## Grid of values to tune over
tuning_grid <- grid_regular(neighbors(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(at, v = 5, repeats=1)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

# Find Best Tuning Parameters1
best_tune_knn <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <- knn_wf %>%
  finalize_workflow(best_tune_knn) %>%
  fit(data=at)

## Predict

amazon_predictions_knn <- final_wf %>% predict(new_data=atest,
                                               type="prob")

amazon_predictions_knn <- amazon_predictions_knn %>% #This predicts
  bind_cols(., atest) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(Action=.pred_1)

stopCluster(cl)

vroom_write(x=amazon_predictions_knn, file="knnPreds2.csv", delim=",")



## PCA naive bayes



library(discrim)
library(naivebayes)
library(themis)

## Recipe

my_recipe <- recipe(ACTION ~ ., data=at) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  #step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), threshold = 0.9) %>% 
  step_downsample()

## model and workflow

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naive bayes eng

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here

## Grid of values to tune over
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(at, v = 5, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

# Find Best Tuning Parameters1
best_tune_nb <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <- nb_wf %>%
  finalize_workflow(best_tune_nb) %>%
  fit(data=at)


## Predict

#predict(final_wf, new_data=atest, type="prob")

amazon_predictions_nb <- final_wf %>% predict(new_data=atest,
                                              type="prob")

amazon_predictions_nb <- amazon_predictions_nb %>% #This predicts
  bind_cols(., atest) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(Action=.pred_1)

vroom_write(x=amazon_predictions_nb, file="nbPredsPCA.csv", delim=",")



## SVM



library(doParallel)

num_cores <- parallel::detectCores()

cl <- makePSOCKcluster(num_cores)

registerDoParallel(cl)

my_recipe <- recipe(ACTION ~ ., data=at) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), threshold = 0.95) %>% 
  step_downsample()

## SVM models
# svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
# set_engine("kernlab")

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

# svmLinear <- svm_linear(cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
# set_engine("kernlab")

## Fit or Tune Model HERE

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)

## Tune smoothness and Laplace here

## Grid of values to tune over
tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 2) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(at, v = 2, repeats=1)

## Run the CV
CV_results <- svm_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

# Find Best Tuning Parameters1
best_tune_svm <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <- svm_wf %>%
  finalize_workflow(best_tune_svm) %>%
  fit(data=at)


## Predict
amazon_predictions_svm <- final_wf %>% predict(new_data=atest,
                                               type="prob")

amazon_predictions_svm <- amazon_predictions_svm %>% #This predicts
  bind_cols(., atest) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(Action=.pred_1)

vroom_write(x=amazon_predictions_svm, file="predsSVM.csv", delim=",")

stopCluster(cl)




























##################################### FINAL ###################################

library(doParallel)
library(tidyverse)
library(tidymodels)
library(vroom)
library(stacks)

num_cores <- parallel::detectCores()

cl <- makePSOCKcluster(num_cores)

registerDoParallel(cl)

library(tidymodels)
library(themis) # for smote

my_recipe <- recipe(ACTION ~ ., data=at) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_downsample()

my_mod_rf <- rand_forest(mtry = tune(),
                         min_n=tune(),
                         trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")


amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_rf)

## Set up grid of tuning values

tuning_grid <- grid_regular(mtry(range = c(1,(ncol(at)-1))),
                            min_n(),
                            levels = 3)

## Set up K-fold CV

folds <- vfold_cv(at, v = 3, repeats=3)

## Find best tuning parameters

CV_results <- amazon_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

best_tune_rf <- CV_results %>%
  select_best("roc_auc")

## Finalize workflow and predict

final_wf_1 <- amazon_workflow %>%
  finalize_workflow(best_tune_rf) %>%
  fit(data=at)

amazon_predictions_rf <- final_wf_1 %>% predict(new_data=atest,
                                              type="prob")

ap_rf <- amazon_predictions_rf %>% #This predicts
  bind_cols(., atest) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(Action=.pred_1)

stopCluster(cl)

write.csv(ap_rf, file = "finalAmazon.csv", quote = FALSE, row.names = FALSE)











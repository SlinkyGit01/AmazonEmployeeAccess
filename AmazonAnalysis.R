library(tidyverse)
library(tidymodels)
library(vroom)
library(ggmosaic)
library(embed)

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

ggplot(data=baked) + geom_mosaic(aes(x=baked$MGR_ID_other, fill=ACTION))



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































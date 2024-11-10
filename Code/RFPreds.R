library(tidymodels)
library(vroom)
library(embed)

GGGtrain <- vroom("train.csv")
GGGtest <- vroom("test.csv")

# Create recipe
GGGrecipe <- recipe(type ~ ., data = GGGtrain) |> 
  
  step_mutate_at(all_nominal_predictors(), fn = factor) |>
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) |> 
  step_normalize(all_numeric_predictors())

# Create random forest model
GGG_rf <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) |> 
  set_engine("ranger") |> 
  set_mode("classification")

# Create workflow
rf_workflow <- workflow() |> 
  add_model(GGG_rf) |> 
  add_recipe(GGGrecipe)

# Grid of values to tune over
tuning_grid <- grid_regular(mtry(range=c(1,6)),
                            min_n(),
                            levels = 5)

# Split data for CV
folds <- vfold_cv(GGGtrain, v = 10, repeats = 1)

# Run CV
CV_results <- rf_workflow |> 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy, roc_auc, f_meas,
                                 sens, recall, precision))

# Find best tuning parameters
best_tune <- CV_results |> 
  select_best(metric = "f_meas")

# Finalize workflow and fit
final_workflow <- rf_workflow |> 
  finalize_workflow(best_tune) |> 
  fit(data = GGGtrain)

# Make predictions
rf_preds <- final_workflow |>
  predict(new_data = GGGtest, type = "class")

# Prep for Kaggle submission
kaggle_sub <- rf_preds %>%
  bind_cols(., GGGtest) |> # Bind predictions to test data
  rename(type = .pred_class) |> # Rename .pred_class to type for Kaggle submission
  select(id, type) # Keep id and type variables

# Write out file
vroom_write(x = kaggle_sub, file = "./RFPreds.csv", delim = ",")

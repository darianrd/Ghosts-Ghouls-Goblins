library(tidymodels)
library(vroom)

# Read in data
GGGtrain <- vroom("train.csv")
GGGtest <- vroom("test.csv")

# Create recipe
GGG_recipe <- recipe(type ~ ., data = GGGtrain) |>
  step_mutate(color = as.factor(color)) |> 
  step_dummy(color) |> 
  step_range(all_numeric_predictors(), min = 0, max = 1)

# Create neural network model
nn_mod <- mlp(hidden_units = tune(),
              epochs = 100) |> 
  set_engine("keras") |> 
  set_mode("classification")

# Create workflow
nn_workflow <- workflow() |> 
  add_model(nn_mod) |> 
  add_recipe(GGG_recipe)

# Grid of values to tune over
tuning_grid <- grid_regular(hidden_units(),
                            levels = 5)
# Split data for CV
folds <- vfold_cv(GGGtrain, v = 5, repeats = 1)

# Run CV
CV_results <- nn_workflow |> 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy, roc_auc, f_meas,
                                 sens, recall, precision))

# Find best tuning parameters
best_tune <- CV_results |> 
  select_best(metric = "accuracy")

# Finalize workflow and fit
final_workflow <- nn_workflow |> 
  finalize_workflow(best_tune) |> 
  fit(data = GGGtrain)

# Make predictions
nn_preds <- final_workflow |>
  predict(new_data = GGGtest, type = "class")

# Create plot with hidden objects on x-axis and mean on y-axis
nn_tune_grid <- grid_regular(hidden_units(),
                             levels = 5)

tuned_nn <- nn_workflow |> 
  tune_grid(resamples = folds,
            grid = nn_tune_grid,
            metrics = metric_set(accuracy, roc_auc, f_meas,
                                 sens, recall, precision))

tuned_nn |> collect_metrics() |> 
  filter(.metric == "accuracy") |> 
  ggplot(aes(x = hidden_units, y = mean)) + geom_line()

# Prep for Kaggle submission
kaggle_sub <- nn_preds %>%
  bind_cols(., GGGtest) |> # Bind predictions to test data
  rename(type = .pred_class) |> # Rename .pred_class to type for Kaggle submission
  select(id, type) # Keep id and type variables

# Write out file
vroom_write(x = kaggle_sub, file = "./NNPreds.csv", delim = ",")

library(tidymodels)
library(vroom)
library(bonsai)
library(lightgbm)

# Read in data
GGGtrain <- vroom("train.csv")
GGGtest <- vroom("test.csv")

# Create recipe
GGGrecipe <- recipe(type ~ ., data = GGGtrain) |> 
  step_mutate_at(all_nominal_predictors(), fn = factor) |> 
  step_dummy(all_nominal_predictors())

# Create boosted tree model
boost_mod <- boost_tree(tree_depth = tune(),
                        trees = tune(),
                        learn_rate = tune()) |> 
  set_engine("lightgbm") |> 
  set_mode("classification")

# Create boosted tree workflow
boost_wf <- workflow() |> 
  add_recipe(GGGrecipe) |> 
  add_model(boost_mod)

# Grid of values to tune over
grid_tuning <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 5)

# Split data for cross-validation
folds <- vfold_cv(GGGtrain, v = 5, repeats = 1)

# Run cross-validation
boost_CV <- boost_wf |> 
  tune_grid(resamples = folds,
            grid = grid_tuning,
            metrics = metric_set(accuracy))

# Find best tuning parameters
best_tuning <- boost_CV |> 
  select_best(metric = "accuracy")

# Finalize workflow
final_workflow <- boost_wf |> 
  finalize_workflow(best_tuning) |> 
  fit(data = GGGtrain)

# Get predictions
boost_preds <- final_workflow |> 
  predict(new_data = GGGtest)

# Prep for Kaggle submission
kaggle_sub <- boost_preds %>%
  bind_cols(., GGGtest) |> # Bind predictions to test data
  rename(type = .pred_class) |> # Rename .pred_class to type for Kaggle submission
  select(id, type) # Keep id and type variables

# Write out file
vroom_write(x = kaggle_sub, file = "./BoostPreds.csv", delim = ",")

library(tidymodels)
library(vroom)

GGGtrain <- vroom("train.csv")
GGGtest <- vroom("test.csv")
NAtrain <- vroom("GGGtrainNA.csv")

NArecipe <- recipe(type ~ ., data = NAtrain) |> 
  step_impute_knn(all_predictors(),
                  neighbors = 10,
                  impute_with = imp_vars(bone_length,
                                         rotting_flesh,
                                         hair_length))

NA_prep <- prep(NArecipe)
NA_impute <- bake(NA_prep, new_data = NAtrain)

rmse_vec(GGGtrain[is.na(NAtrain)],
         NA_impute[is.na(NAtrain)])
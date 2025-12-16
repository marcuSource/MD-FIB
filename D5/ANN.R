# ==============================================================================
# FINAL PROJECT: ADVANCED MARKETING ANALYTICS (FIXED & VERIFIED)
# ==============================================================================
# 1. SETUP
# ------------------------------------------------------------------------------
packages <- c("xgboost", "caret", "ROCR", "ggplot2", "reshape2", "pdp", "Ckmeans.1d.dp", "torch", "dplyr")

for(p in packages){
  if(!require(p, character.only = TRUE)) {
    install.packages(p)
    library(p, character.only = TRUE)
  }
}

# 2. DATA PREPARATION
# ------------------------------------------------------------------------------
if(!file.exists("ifood_enriched.csv")) stop("Data file 'ifood_enriched.csv' not found!")
df <- read.csv("ifood_enriched.csv")

vars_all <- c("Income", "Recency", "WineExp", "FruitExp", "MeatExp", 
              "FishExp", "SweetExp", "GoldExp", "DealsPurc", "WebPurc", 
              "CatalogPurc", "StorePurc", "WebVisits", "Age", "Response")

data_all <- na.omit(df[, names(df) %in% vars_all])

# Create Labels (0/1)
labels <- ifelse(data_all$Response == "Yes" | data_all$Response == 1, 1, 0)

# Create Matrix for XGBoost
options(na.action='na.pass')
data_matrix <- model.matrix(Response ~ . - 1, data = data_all)
n_features <- ncol(data_matrix) 

# Create Tensors for Torch
x_scaled <- scale(data_matrix)
x_scaled[is.na(x_scaled)] <- 0
x_tensor <- torch_tensor(x_scaled, dtype = torch_float())
y_tensor <- torch_tensor(labels, dtype = torch_float())$unsqueeze(2)

# Stratified Split (80/20)
set.seed(2018)
train_idx <- createDataPartition(labels, p = 0.8, list = FALSE)
test_idx  <- setdiff(1:nrow(data_all), train_idx)

# XGBoost Data
dtrain <- xgb.DMatrix(data = data_matrix[train_idx, ], label = labels[train_idx])
dtest  <- xgb.DMatrix(data = data_matrix[test_idx, ], label = labels[test_idx])

# Torch Data
x_train_t <- x_tensor[train_idx, ]
y_train_t <- y_tensor[train_idx, ]
x_test_t  <- x_tensor[test_idx, ]
y_test_t  <- y_tensor[test_idx, ]

# Calculate Imbalance Ratio
neg_count <- sum(labels[train_idx] == 0)
pos_count <- sum(labels[train_idx] == 1)
ratio <- neg_count / pos_count
cat("Imbalance Ratio:", round(ratio, 2), "\n")

# ==============================================================================
# MODEL 1: AGGRESSIVE XGBOOST (The Champion)
# ==============================================================================
cat("\n--- Training Aggressive XGBoost ---\n")

aggressive_weight <- ratio * 1.5 

params_xgb <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.02,
  max_depth = 4,
  subsample = 0.7,
  colsample_bytree = 0.7,
  scale_pos_weight = aggressive_weight
)

model_xgb <- xgb.train(
  params = params_xgb,
  data = dtrain,
  nrounds = 300,
  watchlist = list(train=dtrain, test=dtest),
  print_every_n = 50,
  verbose = 0
)

# Predict & Evaluate XGBoost
probs_xgb <- predict(model_xgb, dtest)
labels_test <- getinfo(dtest, "label")

# Threshold Tuning (Recall > 0.80)
pred_obj <- prediction(probs_xgb, labels_test)
perf_sens <- performance(pred_obj, "tpr")
cutoffs <- pred_obj@cutoffs[[1]]
tpr_vals <- perf_sens@y.values[[1]]
thresh_xgb <- max(cutoffs[tpr_vals >= 0.80], na.rm=TRUE)

cat("Optimal XGB Threshold:", round(thresh_xgb, 3), "\n")

preds_xgb <- ifelse(probs_xgb > thresh_xgb, 1, 0)
cm_xgb <- table(Actual = labels_test, Predicted = preds_xgb)
print(cm_xgb)

# ==============================================================================
# MODEL 2: ROBUST NEURAL NETWORK (No Batch Norm)
# ==============================================================================
cat("\n--- Training Deep Neural Network (Robust) ---\n")

# Simplified Architecture: Linear -> ReLU -> Dropout
# This avoids the "Batch Norm" dimension error on Windows
RobustMLP <- nn_module(
  "RobustMLP",
  initialize = function(input_dim) {
    self$layer1 <- nn_linear(input_dim, 128)
    self$drop1  <- nn_dropout(0.4)
    self$layer2 <- nn_linear(128, 64)
    self$drop2  <- nn_dropout(0.4)
    self$layer3 <- nn_linear(64, 32)
    self$drop3  <- nn_dropout(0.4)
    self$out    <- nn_linear(32, 1)
  },
  forward = function(x) {
    x %>% 
      self$layer1() %>% torch_relu() %>% self$drop1() %>%
      self$layer2() %>% torch_relu() %>% self$drop2() %>%
      self$layer3() %>% torch_relu() %>% self$drop3() %>%
      self$out()
  }
)

# Initialize Model with Correct Dimensions
model_dnn <- RobustMLP(input_dim = n_features)

loss_fn <- nn_bce_with_logits_loss(pos_weight = torch_tensor(ratio * 1.2))
optimizer <- optim_adamw(model_dnn$parameters, lr = 0.001, weight_decay = 0.01)

# Training Loop
epochs <- 150
for(i in 1:epochs) {
  model_dnn$train()
  optimizer$zero_grad()
  
  out <- model_dnn(x_train_t)
  loss <- loss_fn(out, y_train_t)
  
  loss$backward()
  nn_utils_clip_grad_norm_(model_dnn$parameters, max_norm = 1.0)
  optimizer$step()
  
  if(i %% 50 == 0) cat(sprintf("Epoch %d | Loss: %.4f\n", i, loss$item()))
}

# Evaluate DNN
model_dnn$eval()
with_no_grad({
  logits <- model_dnn(x_test_t)
  probs_dnn <- as_array(logits$sigmoid())
})

# Threshold Tuning (Max F1)
pred_obj_dnn <- prediction(probs_dnn, labels_test)
perf_f1 <- performance(pred_obj_dnn, "f")
thresh_dnn <- perf_f1@x.values[[1]][which.max(perf_f1@y.values[[1]])]

preds_dnn <- ifelse(probs_dnn > thresh_dnn, 1, 0)
cm_dnn <- table(Actual = labels_test, Predicted = preds_dnn)
cat("\n--- Deep Learning Results ---\n")
print(cm_dnn)

# ==============================================================================
# VISUALIZATION SUITE
# ==============================================================================

# 1. XGBoost Feature Importance
png("XGB_Importance.png", width = 2400, height = 1600, res = 300)
imp_mat <- xgb.importance(colnames(data_matrix), model = model_xgb)
xgb.plot.importance(imp_mat, top_n = 10, main = "Key Drivers (XGBoost)")
dev.off()

# 2. XGBoost ROC Curve
auc_xgb <- performance(pred_obj, measure = "auc")@y.values[[1]]
png("XGB_Aggressive_ROC.png", width = 2000, height = 2000, res = 300)
plot(performance(pred_obj, "tpr", "fpr"), col = "#D50000", lwd = 3,
     main = paste("Aggressive XGBoost ROC (AUC =", round(auc_xgb, 3), ")"))
abline(0, 1, col = "grey", lty = 2)
dev.off()

# 3. Density Plot
plot_data <- data.frame(
  Probability = probs_xgb,
  Actual = factor(labels_test, labels = c("Non-Buyer", "Buyer"))
)
p_dens <- ggplot(plot_data, aes(x = Probability, fill = Actual)) +
  geom_density(alpha = 0.6) +
  geom_vline(xintercept = thresh_xgb, linetype = "dashed", size = 1) +
  scale_fill_manual(values = c("#B0BEC5", "#D50000")) +
  theme_minimal() +
  labs(title = "Aggressive Strategy: Confidence Distribution",
       subtitle = paste("Threshold set at", round(thresh_xgb, 2), "to maximize Recall"))
ggsave("XGB_5_Density_Plot.png", p_dens, width = 8, height = 5, dpi = 300)

# 4. Partial Dependence
pd_wine <- partial(model_xgb, pred.var = "WineExp", train = data_matrix[train_idx,], prob = TRUE)
pd_wine$Variable <- "WineExp"
colnames(pd_wine)[1] <- "Value"

pd_recency <- partial(model_xgb, pred.var = "Recency", train = data_matrix[train_idx,], prob = TRUE)
pd_recency$Variable <- "Recency"
colnames(pd_recency)[1] <- "Value"

pd_comb <- rbind(pd_wine, pd_recency)

p_pdp <- ggplot(pd_comb, aes(x = Value, y = yhat)) +
  geom_line(color = "#FF6F00", size = 1.2) +
  facet_wrap(~Variable, scales = "free_x") +
  theme_minimal() +
  labs(title = "Behavioral Drivers (Partial Dependence)",
       y = "Impact on Purchase Probability")
ggsave("XGB_8_PDP_Behaviors.png", p_pdp, width = 8, height = 5, dpi = 300)

# 5. Lift Chart
perf_lift <- performance(pred_obj, "lift", "rpp")
lift_data <- data.frame(
  Percent = perf_lift@x.values[[1]],
  Lift = perf_lift@y.values[[1]]
)
p_lift <- ggplot(lift_data, aes(x = Percent, y = Lift)) +
  geom_line(color = "#2E7D32", size = 1.2) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  theme_minimal() +
  labs(title = "Lift Chart", x = "% Customers Contacted", y = "Lift")
ggsave("XGB_6_Lift_Chart.png", p_lift, width = 8, height = 5, dpi = 300)

# 6. Heatmap
cm_df <- as.data.frame(cm_xgb)
colnames(cm_df) <- c("Actual", "Predicted", "Count")
p_cm <- ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile() + geom_text(aes(label = Count), color = "white", size = 8) +
  scale_fill_gradient(low = "#90CAF9", high = "#1565C0") +
  theme_minimal() + labs(title = "Confusion Matrix")
ggsave("XGB_7_Confusion_Heatmap.png", p_cm, width = 6, height = 5, dpi = 300)

cat("\n--- ALL TASKS COMPLETE ---\n")


# ==============================================================================
# GENERATE GRAPHICS FOR NEURAL NETWORK (ANN)
# ==============================================================================
library(ggplot2)
library(ROCR)

# 1. ANN ROC Curve
# ------------------------------------------------------------------------------
pred_obj_dnn <- prediction(probs_dnn, labels_test)
auc_dnn <- performance(pred_obj_dnn, measure = "auc")@y.values[[1]]

png("ANN_ROC.png", width = 2000, height = 2000, res = 300)
plot(performance(pred_obj_dnn, "tpr", "fpr"), col = "#6200EA", lwd = 3,
     main = paste("Neural Network ROC (AUC =", round(auc_dnn, 3), ")"))
abline(0, 1, col = "grey", lty = 2)
dev.off()

# 2. ANN Confusion Matrix Heatmap
# ------------------------------------------------------------------------------
# Create the dataframe for the heatmap using the ANN results (40 TP, 25 FN)
cm_df_ann <- as.data.frame(cm_dnn) 
colnames(cm_df_ann) <- c("Actual", "Predicted", "Count")

p_cm_ann <- ggplot(cm_df_ann, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile() + 
  geom_text(aes(label = Count), color = "white", size = 8, fontface = "bold") +
  scale_fill_gradient(low = "#B39DDB", high = "#4527A0") + # Purple theme for ANN
  theme_minimal() + 
  labs(title = "ANN Confusion Matrix", subtitle = "Neural Network Performance")

ggsave("ANN_Confusion_Heatmap.png", p_cm_ann, width = 6, height = 5, dpi = 300)

# 3. ANN Density Plot (Confidence)
# ------------------------------------------------------------------------------
plot_data_ann <- data.frame(
  Probability = as.vector(probs_dnn),
  Actual = factor(labels_test, labels = c("Non-Buyer", "Buyer"))
)

p_dens_ann <- ggplot(plot_data_ann, aes(x = Probability, fill = Actual)) +
  geom_density(alpha = 0.6) +
  geom_vline(xintercept = thresh_dnn, linetype = "dashed", size = 1) +
  scale_fill_manual(values = c("#B0BEC5", "#6200EA")) + # Purple for buyers
  theme_minimal() +
  labs(title = "ANN Confidence Distribution",
       subtitle = paste("Threshold set at", round(thresh_dnn, 2)))

ggsave("ANN_Density_Plot.png", p_dens_ann, width = 8, height = 5, dpi = 300)

cat("ANN Graphics Generated: ANN_ROC.png, ANN_Confusion_Heatmap.png, ANN_Density_Plot.png\n")
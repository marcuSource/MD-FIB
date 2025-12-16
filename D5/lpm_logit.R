# ============================================================
# Linear Regression (LPM) + Logistic Regression for Response (0/1) — Report-ready script
# Dataset: ifood_enriched.csv
#
# Adds on top of your current version:
# - test metrics (accuracy/recall/precision/F1/balanced acc/MCC) saved to outputs/test_metrics.csv
# - baseline (always-0) accuracy saved too
# - optional copy of all PNGs to Images/ to match LaTeX paths
# ============================================================

library(dplyr)
library(ggplot2)
library(broom)

has_lmtest   <- requireNamespace("lmtest", quietly = TRUE)
has_sandwich <- requireNamespace("sandwich", quietly = TRUE)
has_pROC     <- requireNamespace("pROC", quietly = TRUE)

# Wider console output (avoids truncated prints)
options(width = 120)

# -----------------------
# 1) Load data
# -----------------------
dd <- read.csv("ifood_enriched.csv")
dir.create("outputs", showWarnings = FALSE)

# -----------------------
# 2) Prepare types
# -----------------------
dd$Response <- as.numeric(dd$Response)

dd$Education <- as.factor(dd$Education)
dd$MaritalSts <- as.factor(dd$MaritalSts)
dd$PreferredProductCategory <- as.factor(dd$PreferredProductCategory)
dd$PreferredChannel <- as.factor(dd$PreferredChannel)
dd$IncomeSegment <- as.factor(dd$IncomeSegment)
dd$CustomerSegment <- as.factor(dd$CustomerSegment)

# -----------------------
# 3) Minimal EDA for report
# -----------------------
p_recency <- ggplot(dd, aes(x=Recency, y=Response)) +
  geom_jitter(height=0.05, width=0, alpha=0.25) +
  geom_smooth(method="lm", se=TRUE) +
  labs(title="Response vs Recency (LPM fit)") +
  theme_minimal()

ggsave("outputs/response_vs_recency.png", p_recency + labs(title="Response vs Recency"),
       width=10, height=7, dpi=200)

# -----------------------
# 4) Model building (m1..m4)
# -----------------------
m1 <- lm(Response ~ Income + Age + Education + MaritalSts + Kidhome + Teenhome + Recency,
         data = dd)

m2 <- lm(Response ~ Income + Age + Education + MaritalSts + Kidhome + Teenhome + Recency +
           WebVisits + WebPurc + CatalogPurc + StorePurc + DealsPurc + Complain,
         data = dd)

m3 <- lm(Response ~ Income + Age + Education + MaritalSts + Kidhome + Teenhome + Recency +
           WebVisits + WebPurc + CatalogPurc + StorePurc + DealsPurc + Complain +
           WineExp + FruitExp + MeatExp + FishExp + SweetExp + GoldExp,
         data = dd)

m4 <- lm(Response ~ Income + Age + Education + MaritalSts +
           Kidhome + Teenhome + Recency +
           WebVisits + WebPurc + CatalogPurc + StorePurc + DealsPurc + Complain +
           WineExp + FruitExp + MeatExp + FishExp + SweetExp + GoldExp +
           AccCmp1 + AccCmp2 + AccCmp3 + AccCmp4 + AccCmp5,
         data = dd)

write.csv(data.frame(
  model = c("m1","m2","m3","m4"),
  n = c(nobs(m1), nobs(m2), nobs(m3), nobs(m4)),
  adj_r2 = c(summary(m1)$adj.r.squared, summary(m2)$adj.r.squared,
             summary(m3)$adj.r.squared, summary(m4)$adj.r.squared),
  r2 = c(summary(m1)$r.squared, summary(m2)$r.squared,
         summary(m3)$r.squared, summary(m4)$r.squared),
  AIC = c(AIC(m1), AIC(m2), AIC(m3), AIC(m4))
), "outputs/model_comparison.csv", row.names = FALSE)

# -----------------------
# 5) Stepwise model selection (AIC)
# -----------------------
m_step <- step(m4, direction = "both", trace = 0)

# -----------------------
# 6) Tables for report (pp)
# -----------------------
tidy_pp <- function(mod){
  broom::tidy(mod, conf.int = TRUE) %>%
    mutate(
      estimate_pp = 100 * estimate,
      conf.low_pp = 100 * conf.low,
      conf.high_pp = 100 * conf.high
    )
}

write.csv(tidy_pp(m4) %>% arrange(p.value), "outputs/coef_table_m4.csv", row.names = FALSE)
write.csv(tidy_pp(m_step) %>% arrange(p.value), "outputs/coef_table_m_step.csv", row.names = FALSE)

# Effect sizes (LPM)
eff <- coef(m_step)
calc_effect <- function(var, delta){
  if(!var %in% names(eff)) return(NA_real_)
  eff[var] * delta
}
lpm_effect_sizes <- data.frame(
  effect = c("Recency +10", "WebVisits +1", "WebPurc +1", "StorePurc +1", "DealsPurc +1", "MeatExp +100", "GoldExp +100"),
  delta = c(10, 1, 1, 1, 1, 100, 100),
  pp_change = c(
    100*calc_effect("Recency", 10),
    100*calc_effect("WebVisits", 1),
    100*calc_effect("WebPurc", 1),
    100*calc_effect("StorePurc", 1),
    100*calc_effect("DealsPurc", 1),
    100*calc_effect("MeatExp", 100),
    100*calc_effect("GoldExp", 100)
  )
)
write.csv(lpm_effect_sizes, "outputs/lpm_effect_sizes_pp.csv", row.names = FALSE)

# -----------------------
# 7) Diagnostics plots (LPM)
# -----------------------
save_diag <- function(mod, name){
  png(paste0("outputs/diagnostics_", name, "_1_resid_fitted.png"), width=1200, height=800)
  plot(mod, which=1); dev.off()
  png(paste0("outputs/diagnostics_", name, "_2_qq.png"), width=1200, height=800)
  plot(mod, which=2); dev.off()
  png(paste0("outputs/diagnostics_", name, "_4_cook.png"), width=1200, height=800)
  plot(mod, which=4); dev.off()
  png(paste0("outputs/diagnostics_", name, "_5_leverage.png"), width=1200, height=800)
  plot(mod, which=5); dev.off()
}
save_diag(m4, "m4")
save_diag(m_step, "m_step")

file.copy("outputs/diagnostics_m_step_1_resid_fitted.png", "outputs/residuals_fitted_m_step.png", overwrite = TRUE)
file.copy("outputs/diagnostics_m_step_2_qq.png", "outputs/qq_plot_m_step.png", overwrite = TRUE)
file.copy("outputs/diagnostics_m_step_4_cook.png", "outputs/cooks_distance_m_step.png", overwrite = TRUE)

# coefplot LPM
coefs <- summary(m_step)$coefficients
ci <- confint(m_step)
coef_df <- data.frame(
  term = rownames(coefs),
  estimate = coefs[, "Estimate"],
  pvalue = coefs[, "Pr(>|t|)"],
  conf_low = ci[, 1],
  conf_high = ci[, 2],
  stringsAsFactors = FALSE
)
coef_df <- subset(coef_df, term != "(Intercept)")
coef_df <- coef_df[order(abs(coef_df$estimate), decreasing = TRUE), ]
coef_df$term <- factor(coef_df$term, levels = coef_df$term)

p_coef <- ggplot(coef_df, aes(x = term, y = estimate)) +
  geom_hline(yintercept = 0, linetype = 2) +
  geom_errorbar(aes(ymin = conf_low, ymax = conf_high), width = 0.2) +
  geom_point(size = 2) +
  coord_flip() +
  labs(title = "Final LPM (m_step): Coefficients with 95% CI", x = "", y = "Estimated effect on P(Response=1)") +
  theme_minimal()
ggsave("outputs/coefplot_m_step.png", p_coef, width = 10, height = 7, dpi = 200)

# fitted distribution LPM
fitted_vals <- predict(m_step)
p_fit <- ggplot(data.frame(fitted = fitted_vals), aes(x = fitted)) +
  geom_histogram(bins = 40) +
  geom_vline(xintercept = 0, linetype = 2) +
  geom_vline(xintercept = 1, linetype = 2) +
  labs(title = "Distribution of fitted values (m_step)", x = "Fitted value (interpreted as probability)", y = "Count") +
  theme_minimal()
ggsave("outputs/fitted_distribution_m_step.png", p_fit, width = 9, height = 5, dpi = 200)

# compare resid-fitted
png("outputs/compare_resid_fitted_m4_vs_mstep.png", width = 1400, height = 600, res = 160)
par(mfrow = c(1, 2))
plot(m4, which = 1, main = "m4: Residuals vs Fitted")
plot(m_step, which = 1, main = "m_step: Residuals vs Fitted")
dev.off()
par(mfrow = c(1, 1))

# -----------------------
# 8) Robust SE (HC1) — LPM
# -----------------------
if (has_lmtest && has_sandwich) {
  robust_lpm <- lmtest::coeftest(m_step, vcov = sandwich::vcovHC(m_step, type = "HC1"))
  robust_lpm_df <- data.frame(
    term = rownames(robust_lpm),
    estimate = robust_lpm[,1],
    robust_se = robust_lpm[,2],
    z_or_t = robust_lpm[,3],
    p_value = robust_lpm[,4],
    row.names = NULL
  )
  write.csv(robust_lpm_df, "outputs/robust_se_m_step_lpm.csv", row.names = FALSE)
}

# -----------------------
# 9) Train/Test split evaluation (same as your current report)
# -----------------------
set.seed(123)
n <- nrow(dd)
idx <- sample(1:n, round(0.67*n))

train <- dd[idx, ]
test  <- dd[-idx, ]

# helper: metrics from predicted probabilities
metrics_from_probs <- function(y_true, p_hat, thr = 0.5){
  
  # Limpieza básica (por si hay NA)
  ok <- is.finite(y_true) & is.finite(p_hat)
  y_true <- y_true[ok]
  p_hat  <- p_hat[ok]
  
  # Asegura 0/1
  y_true <- ifelse(y_true >= 0.5, 1, 0)
  y_pred <- ifelse(p_hat  >= thr,  1, 0)
  
  # Tabla 2x2 garantizada
  tab <- table(
    Predicted = factor(y_pred, levels = c(0, 1)),
    Observed  = factor(y_true, levels = c(0, 1))
  )
  
  TN <- tab["0","0"]; FN <- tab["0","1"]; FP <- tab["1","0"]; TP <- tab["1","1"]
  
  acc <- (TP+TN) / sum(tab)
  recall <- ifelse((TP+FN)==0, NA, TP/(TP+FN))
  precision <- ifelse((TP+FP)==0, NA, TP/(TP+FP))
  specificity <- ifelse((TN+FP)==0, NA, TN/(TN+FP))
  f1 <- ifelse(is.na(precision) || is.na(recall) || (precision+recall)==0, NA, 2*precision*recall/(precision+recall))
  bal_acc <- mean(c(recall, specificity), na.rm = TRUE)
  mcc_den <- sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
  mcc <- ifelse(mcc_den==0, NA, (TP*TN - FP*FN)/mcc_den)
  
  list(tab=tab, acc=acc, recall=recall, precision=precision, specificity=specificity, f1=f1, bal_acc=bal_acc, mcc=mcc)
}

# LPM on train
m_final_lpm <- lm(formula(m_step), data = train)
pred_lpm <- predict(m_final_lpm, newdata = test)

# LPM metrics (treating predictions as probs)
m_lpm <- metrics_from_probs(test$Response, pred_lpm, thr = 0.5)

# plot pred vs obs
png("outputs/test_pred_vs_obs_lpm.png", width=1200, height=800)
plot(pred_lpm, test$Response, xlab="Predicted (LPM)", ylab="Observed Response",
     main="Test set: Predicted vs Observed (LPM)")
abline(h=0.5, lty=2); dev.off()

# -----------------------
# 10) Logistic Regression
# -----------------------
glm_from_lpm <- glm(formula(m_step), data = dd, family = binomial)
glm_test <- glm(formula(m_step), data = train, family = binomial)
pred_logit <- predict(glm_test, newdata = test, type = "response")

m_logit <- metrics_from_probs(test$Response, pred_logit, thr = 0.5)

# Log-loss
eps <- 1e-15
pclip <- pmin(pmax(pred_logit, eps), 1-eps)
logloss <- -mean(test$Response*log(pclip) + (1-test$Response)*log(1-pclip))

# AUC
auc_val <- NA
if (has_pROC) {
  roc_obj <- pROC::roc(test$Response, pred_logit, quiet = TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))
  png("outputs/roc_logit.png", width=1200, height=800)
  plot(roc_obj, main = paste0("ROC Curve (Logistic) — AUC = ", round(auc_val, 3)))
  dev.off()
}

# OR plot (glm_from_lpm on full data)
tidy_or <- function(mod){
  tt <- broom::tidy(mod, conf.int = TRUE)
  tt %>%
    mutate(odds_ratio = exp(estimate), or_low = exp(conf.low), or_high = exp(conf.high)) %>%
    arrange(p.value)
}
logit_or_from_lpm <- tidy_or(glm_from_lpm)
write.csv(logit_or_from_lpm, "outputs/coef_table_logit_from_lpm_or.csv", row.names = FALSE)

df2 <- logit_or_from_lpm %>% filter(term != "(Intercept)")
df2$term <- factor(df2$term, levels = df2$term[order(df2$odds_ratio)])
p_or <- ggplot(df2, aes(x = term, y = odds_ratio)) +
  geom_hline(yintercept = 1, linetype = 2) +
  geom_errorbar(aes(ymin = or_low, ymax = or_high), width = 0.2) +
  geom_point(size = 2) +
  coord_flip() +
  labs(title = "Logistic (glm_from_lpm): Odds Ratios with 95% CI", x = "", y = "Odds Ratio (95% CI)") +
  theme_minimal()
ggsave("outputs/coefplot_logit_from_lpm_or.png", p_or, width = 10, height = 7, dpi = 200)

# Average pp changes (finite differences) on full data
avg_pp_change <- function(mod, data, var, delta){
  base <- predict(mod, newdata = data, type = "response")
  data2 <- data
  data2[[var]] <- data2[[var]] + delta
  newp <- predict(mod, newdata = data2, type = "response")
  100 * mean(newp - base, na.rm = TRUE)
}
avg_pp_switch01 <- function(mod, data, var){
  base <- data
  base[[var]] <- 0
  p0 <- predict(mod, newdata = base, type = "response")
  base[[var]] <- 1
  p1 <- predict(mod, newdata = base, type = "response")
  100 * mean(p1 - p0, na.rm = TRUE)
}

logit_effects_pp <- data.frame(
  effect = c("Recency +10", "WebVisits +1", "WebPurc +1", "StorePurc +1", "DealsPurc +1", "MeatExp +100", "GoldExp +100",
             "AccCmp1: 0->1", "AccCmp2: 0->1", "AccCmp3: 0->1", "AccCmp4: 0->1", "AccCmp5: 0->1"),
  pp_change = c(
    avg_pp_change(glm_from_lpm, dd, "Recency", 10),
    avg_pp_change(glm_from_lpm, dd, "WebVisits", 1),
    avg_pp_change(glm_from_lpm, dd, "WebPurc", 1),
    avg_pp_change(glm_from_lpm, dd, "StorePurc", 1),
    avg_pp_change(glm_from_lpm, dd, "DealsPurc", 1),
    avg_pp_change(glm_from_lpm, dd, "MeatExp", 100),
    avg_pp_change(glm_from_lpm, dd, "GoldExp", 100),
    avg_pp_switch01(glm_from_lpm, dd, "AccCmp1"),
    avg_pp_switch01(glm_from_lpm, dd, "AccCmp2"),
    avg_pp_switch01(glm_from_lpm, dd, "AccCmp3"),
    avg_pp_switch01(glm_from_lpm, dd, "AccCmp4"),
    avg_pp_switch01(glm_from_lpm, dd, "AccCmp5")
  )
)
write.csv(logit_effects_pp, "outputs/logit_effect_sizes_pp.csv", row.names = FALSE)

# Predicted probability distribution (logit, full data)
pred_logit_all <- predict(glm_from_lpm, type = "response")
p_logit_dist <- ggplot(data.frame(p = pred_logit_all), aes(x = p)) +
  geom_histogram(bins = 40) +
  labs(title = "Distribution of predicted probabilities (Logistic: glm_from_lpm)",
       x = "Predicted probability", y = "Count") +
  theme_minimal()
ggsave("outputs/pred_distribution_logit_from_lpm.png", p_logit_dist, width = 9, height = 5, dpi = 200)

# Robust SE for logit (optional)
if (has_lmtest && has_sandwich) {
  robust_logit <- lmtest::coeftest(glm_from_lpm, vcov = sandwich::vcovHC(glm_from_lpm, type = "HC1"))
  robust_logit_df <- data.frame(
    term = rownames(robust_logit),
    estimate = robust_logit[,1],
    robust_se = robust_logit[,2],
    z = robust_logit[,3],
    p_value = robust_logit[,4],
    row.names = NULL
  )
  write.csv(robust_logit_df, "outputs/robust_se_glm_from_lpm.csv", row.names = FALSE)
}

# -----------------------
# 11) Save test metrics (new)
# -----------------------
baseline_acc <- mean(test$Response == 0)

test_metrics <- data.frame(
  model = c("Baseline (always 0)", "LPM @0.5", "Logit @0.5"),
  accuracy = c(baseline_acc, m_lpm$acc, m_logit$acc),
  recall = c(0, m_lpm$recall, m_logit$recall),
  precision = c(NA, m_lpm$precision, m_logit$precision),
  f1 = c(NA, m_lpm$f1, m_logit$f1),
  balanced_accuracy = c(NA, m_lpm$bal_acc, m_logit$bal_acc),
  mcc = c(NA, m_lpm$mcc, m_logit$mcc),
  auc = c(NA, NA, auc_val),
  logloss = c(NA, NA, logloss)
)
write.csv(test_metrics, "outputs/test_metrics.csv", row.names = FALSE)
# -----------------------
# 12) Threshold optimisation (LOGIT, test set)
# -----------------------
sweep_thresholds <- function(y_true, p_hat, by = 0.01){
  ts <- seq(0, 1, by = by)
  out <- lapply(ts, function(th){
    m <- metrics_from_probs(y_true, p_hat, thr = th)
    data.frame(
      threshold = th,
      accuracy = m$acc,
      recall = m$recall,
      precision = m$precision,
      specificity = m$specificity,
      f1 = m$f1,
      balanced_accuracy = m$bal_acc,
      youden_j = m$recall + m$specificity - 1
    )
  })
  bind_rows(out)
}

thr_tbl <- sweep_thresholds(test$Response, pred_logit, by = 0.01)

best_f1 <- thr_tbl %>% filter(!is.na(f1)) %>% slice_max(order_by = f1, n = 1, with_ties = FALSE)
best_youden <- thr_tbl %>% filter(!is.na(youden_j)) %>% slice_max(order_by = youden_j, n = 1, with_ties = FALSE)

write.csv(thr_tbl, "outputs/threshold_optimization_logit.csv", row.names = FALSE)
write.csv(best_f1, "outputs/threshold_best_f1.csv", row.names = FALSE)
write.csv(best_youden, "outputs/threshold_best_youden.csv", row.names = FALSE)

# Plot: Precision/Recall vs threshold
p_pr_thr <- ggplot(thr_tbl, aes(x = threshold)) +
  geom_line(aes(y = precision)) +
  geom_line(aes(y = recall)) +
  labs(title = "Logit (test): Precision & Recall vs Threshold", y = "Metric value") +
  theme_minimal()
ggsave("outputs/precision_recall_vs_threshold.png", p_pr_thr, width = 9, height = 5, dpi = 200)

# Plot: F1 vs threshold
p_f1_thr <- ggplot(thr_tbl, aes(x = threshold, y = f1)) +
  geom_line() +
  labs(title = "Logit (test): F1 vs Threshold", y = "F1 score") +
  theme_minimal()
ggsave("outputs/f1_vs_threshold.png", p_f1_thr, width = 9, height = 5, dpi = 200)

# -----------------------
# 13) Lift / Gains (LOGIT, test set)
# -----------------------
lift_table <- function(y_true, p_hat, k = 10){
  df <- data.frame(y = y_true, p = p_hat) %>%
    arrange(desc(p)) %>%
    mutate(rank = row_number(),
           bucket = ceiling(rank / (n()/k)))
  overall_rate <- mean(df$y)

  tbl <- df %>%
    group_by(bucket) %>%
    summarise(
      n = n(),
      responders = sum(y),
      response_rate = mean(y),
      avg_score = mean(p),
      .groups = "drop"
    ) %>%
    mutate(
      cum_n = cumsum(n),
      cum_responders = cumsum(responders),
      cum_capture = cum_responders / sum(responders),
      lift = response_rate / overall_rate
    )

  list(tbl = tbl, overall_rate = overall_rate)
}

lt <- lift_table(test$Response, pred_logit, k = 10)
lift_df <- lt$tbl
write.csv(lift_df, "outputs/lift_deciles_logit_test.csv", row.names = FALSE)

gains_df <- lift_df %>% mutate(pct_targeted = cum_n / sum(n))

# Gains chart
p_gains <- ggplot(gains_df, aes(x = pct_targeted, y = cum_capture)) +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = 2) +
  scale_x_continuous(labels = function(x) paste0(round(100*x), "%")) +
  scale_y_continuous(labels = function(x) paste0(round(100*x), "%")) +
  labs(title = "Gains chart (test): cumulative responders captured",
       x = "% of customers targeted (highest scores first)",
       y = "% of responders captured") +
  theme_minimal()
ggsave("outputs/gains_chart_logit_test.png", p_gains, width = 9, height = 5, dpi = 200)

# Lift chart
p_lift <- ggplot(lift_df, aes(x = bucket, y = lift)) +
  geom_col() +
  labs(title = "Lift chart (test): response lift by decile",
       x = "Decile (1 = highest predicted probability)",
       y = "Lift (decile response rate / overall response rate)") +
  theme_minimal()
ggsave("outputs/lift_chart_logit_test.png", p_lift, width = 9, height = 5, dpi = 200)

# also save confusion matrices as text
sink("outputs/test_confusion_matrices.txt")
cat("=== TEST SET SIZE ===\n")
cat("n_test =", nrow(test), "\n")
cat("Response rate (test) =", mean(test$Response), "\n\n")
cat("=== LPM (thr=0.5) confusion ===\n"); print(m_lpm$tab); cat("\n")
cat("=== LOGIT (thr=0.5) confusion ===\n"); print(m_logit$tab); cat("\n")
sink()

# -----------------------
# 12) Copy figures to Images/ (optional, for LaTeX paths)
# -----------------------
dir.create("Images", showWarnings = FALSE)
file.copy(list.files("outputs", pattern="\\.png$", full.names=TRUE), "Images", overwrite=TRUE)

cat("\nDONE. Outputs in outputs/ and (optionally) Images/.\n")


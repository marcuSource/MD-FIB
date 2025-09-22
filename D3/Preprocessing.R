 # Load required libraries
library(dplyr)
library(class)

# 0. Load raw dataset
ifood <- read.csv("ifood_base.csv", sep=",", header=TRUE, stringsAsFactors = FALSE)

# 1. Remove irrelevant columns
ifood <- ifood[, !names(ifood) %in% c("ID", "Z_CostContact", "Z_Revenue")]

# 2. Transform date-related variables
ifood$Age <- 2020 - ifood$Year_Birth
ifood <- ifood[, !names(ifood) %in% c("Year_Birth")]

reference_date <- as.Date("2020-12-31")
ifood$CustDays <- as.numeric(reference_date - as.Date(ifood$Dt_Customer, format="%Y-%m-%d"))
ifood <- ifood[, !names(ifood) %in% c("Dt_Customer")]

# 3. Rename columns for easier access
colnames(ifood) <- gsub("NumDealsPurchases", "DealsPurc", colnames(ifood))
colnames(ifood) <- gsub("NumWebPurchases", "WebPurc", colnames(ifood))
colnames(ifood) <- gsub("NumStorePurchases", "StorePurc", colnames(ifood))
colnames(ifood) <- gsub("NumWebVisitsMonth", "WebVisits", colnames(ifood))
colnames(ifood) <- gsub("AcceptedCmpOverall", "CmpOverall", colnames(ifood))
colnames(ifood) <- gsub("MntWines", "WineExp", colnames(ifood))
colnames(ifood) <- gsub("MntFruits", "FruitExp", colnames(ifood))
colnames(ifood) <- gsub("MntMeatProducts", "MeatExp", colnames(ifood))
colnames(ifood) <- gsub("MntFishProducts", "FishExp", colnames(ifood))
colnames(ifood) <- gsub("MntSweetProducts", "SweetExp", colnames(ifood))
colnames(ifood) <- gsub("MntGoldProds", "GoldExp", colnames(ifood))
colnames(ifood) <- gsub("Marital_Status", "MaritalSts", colnames(ifood))
colnames(ifood) <- gsub("NumCatalogPurchases", "CatalogPurc", colnames(ifood))
colnames(ifood) <- gsub("AcceptedCmp1", "AccCmp1", colnames(ifood))
colnames(ifood) <- gsub("AcceptedCmp2", "AccCmp2", colnames(ifood))
colnames(ifood) <- gsub("AcceptedCmp3", "AccCmp3", colnames(ifood))
colnames(ifood) <- gsub("AcceptedCmp4", "AccCmp4", colnames(ifood))
colnames(ifood) <- gsub("AcceptedCmp5", "AccCmp5", colnames(ifood))

# 4. Handle outliers
ifood$Age <- ifelse(ifood$Age > 80, 80, ifood$Age)

# 5. Handle missing values
ifood <- ifood[!ifood$MaritalSts %in% c("YOLO", "Absurd"),]
ifood$MaritalSts[ifood$MaritalSts == "Alone"] <- "Single"

# 6. Impute missing Income using KNN
ifood$Income <- ifelse(ifood$Income < 12500, NA, ifood$Income)

num_vars <- sapply(ifood, is.numeric)
complete_vars <- colnames(ifood)[num_vars]
missing_threshold <- 0.2 * nrow(ifood)
complete_vars <- complete_vars[colSums(is.na(ifood[, complete_vars])) < missing_threshold]
aux <- ifood[, complete_vars]

var <- "Income"
aux1 <- aux[!is.na(ifood[[var]]), , drop = FALSE]
aux2 <- aux[is.na(ifood[[var]]), , drop = FALSE]

cols_na <- colnames(aux2)[colSums(is.na(aux2)) > 0]
if (length(cols_na) > 0) {
  aux1 <- aux1[, !(colnames(aux1) %in% cols_na), drop = FALSE]
  aux2 <- aux2[, !(colnames(aux2) %in% cols_na), drop = FALSE]
}

knn_impute <- knn(aux1, aux2, ifood[[var]][!is.na(ifood[[var]])], k = 1)
ifood[[var]][is.na(ifood[[var]])] <- as.numeric(as.character(knn_impute))

# 7. Correct calculation of TotAccCmp
ifood$TotAccCmp <- ifood$AccCmp1 + ifood$AccCmp2 + ifood$AccCmp3 + ifood$AccCmp4 + ifood$AccCmp5

# 8. Remove duplicate records
ifood <- ifood %>% arrange(desc(Response)) %>% distinct_at(vars(-Response), .keep_all = TRUE)

# 9. Create `TotalExp` before using it
ifood$TotalExp <- rowSums(ifood[, c("WineExp", "FruitExp", "MeatExp", "FishExp", "SweetExp", "GoldExp")], na.rm = TRUE)

# 10. Save cleaned dataset
write.csv(ifood, "ifood_cleaned.csv", row.names = FALSE)

# -----------------------------------
# Variable Creation
# -----------------------------------

# SECOND-GENERATON:
# -----------------------------------
# Total Purchases
ifood$TotalPurchases <- ifood$DealsPurc + ifood$WebPurc + ifood$CatalogPurc + ifood$StorePurc

# Purchase Frequency
ifood$PurchaseFrequency <- ifelse(ifood$CustDays > 0, ifood$TotalPurchases / (ifood$CustDays / 30), 0)

# Preferred Product Category
product_categories <- c("WineExp", "FruitExp", "MeatExp", "FishExp", "SweetExp", "GoldExp")
max_index <- apply(ifood[ , product_categories], 1, which.max)
ifood$PreferredProductCategory <- product_categories[max_index]
ifood$PreferredProductCategory <- as.factor(ifood$PreferredProductCategory)

# Preferred Purchase Channel
channels <- c("DealsPurc", "WebPurc", "CatalogPurc", "StorePurc")
max_ch_index <- apply(ifood[ , channels], 1, which.max)
ifood$PreferredChannel <- channels[max_ch_index]
ifood$PreferredChannel <- as.factor(ifood$PreferredChannel)

# Average Spend Per Purchase
ifood$AvgSpendPerPurchase <- ifelse(ifood$TotalPurchases > 0, ifood$TotalExp / ifood$TotalPurchases, 0)

# HasChildren
ifood$HasChildren <- ifelse(ifood$Kidhome + ifood$Teenhome > 0, 1, 0)

# IncomeSegment
income_quantiles <- quantile(ifood$Income, probs = c(0.33, 0.66), na.rm = TRUE)
ifood$IncomeSegment <- cut(ifood$Income, breaks = c(-Inf, income_quantiles[1], income_quantiles[2], Inf),
                           labels = c("Low", "Medium", "High"))
# CustomerTenure
ifood$CustomerTenure <- ifood$CustDays / 365

# CampaignAcceptanceRate
ifood$CampaignAcceptanceRate <- ifelse(ifood$TotAccCmp > 0, ifood$TotAccCmp / 5, 0)

# THIRD GENERATION
# ---------------------------------

# Customer Segmentation using k-means clustering
cluster_data <- ifood %>% select(Recency, TotalPurchases, TotalExp)
cluster_data_scaled <- scale(cluster_data)
set.seed(123)
k3 <- kmeans(cluster_data_scaled, centers = 3, nstart = 25)
ifood$CustomerSegment <- as.factor(k3$cluster)

# Propensity Score
propensity_model <- glm(Response ~ Income + Recency + TotalExp + TotalPurchases + TotAccCmp + Age + MaritalSts,
                        data = ifood, family = binomial)
ifood$PropensityScore <- predict(propensity_model, ifood, type = "response")

# Engagement Index
recency_norm <- (max(ifood$Recency) - ifood$Recency) / max(ifood$Recency)
frequency_norm <- ifood$TotalPurchases / max(ifood$TotalPurchases)
monetary_norm <- ifood$TotalExp / max(ifood$TotalExp)
campaign_norm <- (ifood$TotAccCmp + ifood$Response) / 6
webvisit_norm <- ifood$WebVisits / max(ifood$WebVisits)
ifood$EngagementIndex <- (recency_norm + frequency_norm + monetary_norm + campaign_norm + webvisit_norm) / 5 * 100

# Save enriched dataset
write.csv(ifood, "ifood_enriched.csv", row.names = FALSE)


# 1. Load the packages ---------------------------------------------------------
library(arules)
library(arulesViz)  
library(rCBA)

setwd("C:/Users/Michel/Desktop/UPC/Q5/MD/Pruebas")
# 2. Load the databases --------------------------------------------------------
## 2.1 GROCERIES ---------------------------------------------------------------
data(Groceries)
transaccionesG <- Groceries

## 2.2 IFOOD_ENRICHED ----------------------------------------------------------
dd <- read.table("ifood_enriched.csv",header = T, stringsAsFactors = TRUE, sep = ",")

# Aseguramos que las variables son categoricas
dd$IncomeSegment   <- as.factor(dd$IncomeSegment)
dd$PreferredChannel <- as.factor(dd$PreferredChannel)
dd$PreferredProductCategory <- as.factor(dd$PreferredProductCategory)
dd$Teenhome <- as.factor(dd$Teenhome)

#Añadimos variables discretizadas
dd$Response <- factor(dd$Response, levels = c(0, 1), labels = c("No", "Yes"))
dd$HasChildren <- factor(dd$HasChildren, levels = c(0, 1), labels = c("No", "Yes"))
dd$WineSegment <- cut(dd$WineExp, 
                      breaks = quantile(dd$WineExp, probs = c(0, 0.33, 0.66, 1), na.rm = TRUE),
                      labels = c("Wine_Low", "Wine_Medium", "Wine_High"), 
                      include.lowest = TRUE)
dd$AgeGroup <- cut(dd$Age, 
                   breaks = c(-Inf, 35, 55, Inf), 
                   labels = c("Age_Young", "Age_Adult", "Age_Senior"))

### Seleccion de las variables categoricas
dcat <- dd[,sapply(dd, is.factor)]
### transformamos a transacciones
dtrans <- as(dcat, "transactions")
# 3. Data Preprocessing  -------------------------------------------------------
## 3.1 GROCERIES ---------------------------------------------------------------
### Find the products in the data set
products <- itemLabels(transaccionesG)
### View the unique products
products
### Look the stats for transactions 
summary(transaccionesG)
cat("Transaction number:", nrow(transaccionesG), "\n")
size <- size(transaccionesG)
cat("Average size of basket", mean(size), "\n")
### This graph shows the frequencies of each product.
itemFrequencyPlot(transaccionesG, support = 0.05, cex.names = 0.8, col = "pink")
### and now visualize the top 
itemFrequencyPlot(transaccionesG, topN = 30, col = "pink")
### Print the list of items order by relative proportions to itemset
sort(itemFrequency(transaccionesG, type = "relative"), decreasing = T) 
### Remove the transactions which proportion is lower thant 0.005 
transaccionesG <- transaccionesG[, itemFrequency(transaccionesG) > 0.005]
sort(itemFrequency(transaccionesG, type = "relative")) 

## 3.2 IFOOD ---------------------------------------------------------------

### tendra tantas columnas como categorias
foo <- function(x){length(levels(x))}
sum(sapply(dcat, foo)); dtrans
### Find the products in the data set
(modalidades <- itemLabels(dtrans))
### Look the stats for transactions 
summary(dtrans)
cat("Transaction number:", nrow(dtrans), "\n")
size <- size(dtrans)
cat("Average size of basket", mean(size), "\n")
### This graph shows the frequencies of each product.
itemFrequencyPlot(dtrans, support = 0.05, cex.names = 0.8, col = "pink")
### and now visualize the top 
itemFrequencyPlot(dtrans, topN = 30, col = "pink")
### Print the list of items order by relative proportions to itemset
sort(itemFrequency(dtrans, type = "relative"), decreasing = T) 
### Remove the transactions which proportion is lower thant 0.005 
dtrans <- dtrans[, itemFrequency(dtrans) > 0.005]
sort(itemFrequency(dtrans, type = "relative")) 

# 4. Algorithm -----------------------------------------------------------------
db_transaciones <- dtrans

## 4.1 Apriori Algorithm -------------------------------------------------------
### Apply the function (esta tiene demasiadas association rules)
rulesApriori <- apriori(dtrans, 
                        parameter = list(supp = 0.01,
                                         conf = 0.3,
                                         minlen = 2), 
                        appearance = list(rhs = "Response=Yes", default="lhs"))

### Visualitzation the rules
summary(rulesApriori)
attributes(rulesApriori)

### Print the rules 
inspect(rulesApriori)
inspect(sort(rulesApriori, by = "lift"))
### Display rules
inspectDT(rulesApriori)

rulesFiltered <- head(sort(rulesApriori, by = "lift"), 10)
#rulesFiltered <- sort(rulesApriori, by = "lift")
### Display the rules
inspect(sort(rulesFiltered, by = "lift"))
inspectDT(sort(rulesFiltered, by = "lift"))

## Visualitzation the rules 
plot(rulesFiltered, measure = c("support", "lift"), shading = "confidence")
plot(rulesFiltered, method = "two-key plot")
## plot(rulesFiltered, method = "grouped")
plot(rulesFiltered, method = "paracoord")
plot(rulesFiltered, method = "paracoord", measure = "confidence", 
     control = list(reorder = TRUE))
plot(rulesFiltered, method = "graph")


## Visualitzation
plot(rulesFiltered, method = "graph", measure = "lift", shading = "confidence", 
     engine = "htmlwidget", network = TRUE, itemCol = "pink", max = 200)



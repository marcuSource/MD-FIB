# --- CONFIGURACIÓN DE RUTAS Y DATOS ---

# Establece el directorio de trabajo
setwd("C:/Users/Michel/Desktop/UPC/Q5/MD/MD-D4")

# Lee los datos
dd <- read.table("ifood_enriched.csv", header = T, sep = ",", dec = '.')
names(dd)
attach(dd)


# --- FUNCIONES DE ANÁLISIS ---

# Calcula el valor test de la variable Xnum para todas las modalidades del factor P
ValorTestXnum <- function(Xnum, P) {
  # freq dis of fac
  nk <- as.vector(table(P))
  n <- sum(nk)
  # mitjanes x grups
  xk <- tapply(Xnum, P, mean)
  # valors test
  txk <- (xk - mean(Xnum)) / (sd(Xnum) * sqrt((n - nk) / (n * nk)))
  # p-values
  pxk <- pt(txk, n - 1, lower.tail = F)
  for (c in 1:length(levels(as.factor(P)))) {
    if (pxk[c] > 0.5) {
      pxk[c] <- 1 - pxk[c]
    }
  }
  return(pxk)
}


# Calcula el valor test de la variable Xquali para el factor P
ValorTestXquali <- function(P, Xquali) {
  taula <- table(P, Xquali)
  n <- sum(taula)
  pk <- apply(taula, 1, sum) / n
  pj <- apply(taula, 2, sum) / n
  pf <- taula / (n * pk)
  pjm <- matrix(data = pj, nrow = dim(pf)[1], ncol = dim(pf)[2], byrow = TRUE)
  dpf <- pf - pjm
  dvt <- sqrt(((1 - pk) / (n * pk)) %*% t(pj * (1 - pj)))
  
  zkj <- dpf
  zkj[dpf != 0] <- dpf[dpf != 0] / dvt[dpf != 0]
  pzkj <- pnorm(zkj, lower.tail = F)
  for (c in 1:length(levels(as.factor(P)))) {
    for (s in 1:length(levels(Xquali))) {
      if (pzkj[c, s] > 0.5) {
        pzkj[c, s] <- 1 - pzkj[c, s]
      }
    }
  }
  return(list(rowpf = pf, vtest = zkj, pval = pzkj))
}

# Función auxiliar para generar nombres de archivo seguros
make_filename <- function(prefix, var_name, suffix = "") {
  # Reemplaza caracteres no alfanuméricos por guiones bajos
  safe_name <- gsub("[^[:alnum:]_]", "_", var_name)
  # Añade sufijo si se proporciona
  if (suffix != "") {
    suffix <- paste0("_", suffix)
  }
  return(file.path(output_dir, paste0(prefix, "_", safe_name, suffix, ".png")))
}


# --- ANÁLISIS Y GENERACIÓN DE GRÁFICAS ---

dades <- dd
K <- dim(dades)[2]
# Se elimina par(ask=TRUE) para permitir el guardado automático de archivos

# P must contain the class variable
P <- dd$Response
nameP <- "Response"
nc <- length(levels(factor(P)))
pvalk <- matrix(data = 0, nrow = nc, ncol = K, dimnames = list(levels(P), names(dades)))
nameP <- "Response" # El nombre de la clase utilizado en los títulos de las gráficas
n <- dim(dades)[1]

# Define y crea la carpeta de salida para las gráficas, usando el valor final de nameP
output_dir <- paste0("Graficas_", nameP)
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
  cat(paste("Carpeta de salida creada:", output_dir, "\n"))
}

for (k in 1:K) {
  var_name <- names(dades)[k]
  
  if (is.numeric(dades[, k])) {
    # ------------------
    # VARIABLES NUMÉRICAS
    # ------------------
    print(paste("Anàlisi per classes de la Variable:", var_name))
    
    # Gráfica 1: Boxplot
    png(make_filename("boxplot", var_name), width = 800, height = 600)
    boxplot(dades[, k] ~ P, main = paste("Boxplot of", var_name, "vs", nameP), horizontal = TRUE)
    dev.off()
    
    # Gráfica 2: Barplot de Medias
    png(make_filename("mean_barplot", var_name), width = 800, height = 600)
    barplot(tapply(dades[[k]], P, mean), main = paste("Means of", var_name, "by", nameP))
    abline(h = mean(dades[[k]]))
    legend("topright", "global mean", bty = "n")
    dev.off()
    
    # Otros cálculos...
    print("Estadístics per groups:")
    for (s in levels(as.factor(P))) {
      print(summary(dades[P == s, k]))
    }
    o <- oneway.test(dades[, k] ~ P)
    print(paste("p-valueANOVA:", o$p.value))
    kw <- kruskal.test(dades[, k] ~ P)
    print(paste("p-value Kruskal-Wallis:", kw$p.value))
    pvalk[, k] <- ValorTestXnum(dades[, k], P)
    print("p-values ValorsTest: ")
    print(pvalk[, k])
    
  } else {
    if (class(dd[, k]) == "Date") {
      # ------------------
      # VARIABLES TIPO DATE
      # ------------------
      print(summary(dd[, k]))
      print(sd(dd[, k]))
      
      # Gráfica 3: Histograma de Fechas
      png(make_filename("date_hist", var_name), width = 800, height = 600)
      hist(dd[, k], breaks = "weeks", main = paste("Histogram of", var_name, "by weeks"))
      dev.off()
      
    } else {
      # ------------------
      # VARIABLES CUALITATIVAS
      # ------------------
      print(paste("Variable", var_name))
      table <- table(P, dades[, k])
      rowperc <- prop.table(table, 1)
      colperc <- prop.table(table, 2)
      
      # Asegura que sea factor antes de usar levels/attach
      dades[, k] <- as.factor(dades[, k])
      
      marg <- table(as.factor(P)) / n
      print(append("Categories=", levels(as.factor(dades[, k]))))
      
      # [ORIGINAL 1]: Proporciones colperc (condicionadas a la variable de columna dades[,k])
      # Variable de clase (P) en eje de abscisas, líneas para categorías de dades[,k]
      png(make_filename("prop_class_cond_col", var_name, "1"), width = 800, height = 600)
      plot(marg, type = "l", ylim = c(0, 1), main = paste("Prop. of pos & neg by", names(dades)[k]))
      paleta <- rainbow(length(levels(dades[, k])))
      for (c in 1:length(levels(dades[, k]))) {
        lines(colperc[, c], col = paleta[c])
      }
      dev.off()
      
      # [ORIGINAL 2]: Proporciones colperc (con leyenda)
      png(make_filename("prop_class_cond_col", var_name, "2_legend"), width = 800, height = 600)
      plot(marg, type = "l", ylim = c(0, 1), main = paste("Prop. of pos & neg by", names(dades)[k]))
      paleta <- rainbow(length(levels(dades[, k])))
      for (c in 1:length(levels(dades[, k]))) {
        lines(colperc[, c], col = paleta[c])
      }
      legend("topright", levels(dades[, k]), col = paleta, lty = 2, cex = 0.6)
      dev.off()
      
      # [ORIGINAL 3]: Proporciones rowperc (condicionadas a clases P) - Eje de Clase
      png(make_filename("prop_cond_class", var_name, "3"), width = 800, height = 600)
      print(append("Categories=", levels(dades[, k])))
      plot(marg, type = "n", ylim = c(0, 1), main = paste("Prop. of pos & neg by", names(dades)[k]))
      paleta <- rainbow(length(levels(dades[, k])))
      for (c in 1:length(levels(dades[, k]))) {
        lines(rowperc[, c], col = paleta[c])
      }
      dev.off()
      
      # [ORIGINAL 4]: Proporciones rowperc (con leyenda)
      png(make_filename("prop_cond_class", var_name, "4_legend"), width = 800, height = 600)
      plot(marg, type = "n", ylim = c(0, 1), main = paste("Prop. of pos & neg by", names(dades)[k]))
      paleta <- rainbow(length(levels(dades[, k])))
      for (c in 1:length(levels(dades[, k]))) {
        lines(rowperc[, c], col = paleta[c])
      }
      legend("topright", levels(dades[, k]), col = paleta, lty = 2, cex = 0.6)
      dev.off()
      
      # [ORIGINAL 5]: Proporciones rowperc - Eje de Categoría (dades[,k])
      marg_x <- table(dades[, k]) / n
      png(make_filename("prop_cond_class_var_x", var_name, "5"), width = 800, height = 600)
      plot(marg_x, type = "l", ylim = c(0, 1), main = paste("Prop. of pos & neg by", names(dades)[k]), las = 3)
      paleta <- rainbow(length(levels(as.factor(P))))
      for (c in 1:length(levels(as.factor(P)))) {
        lines(rowperc[c, ], col = paleta[c])
      }
      dev.off()
      
      # [ORIGINAL 6]: Proporciones rowperc - Eje de Categoría (con leyenda)
      png(make_filename("prop_cond_class_var_x", var_name, "6_legend"), width = 800, height = 600)
      plot(marg_x, type = "l", ylim = c(0, 1), main = paste("Prop. of pos & neg by", names(dades)[k]), las = 3)
      for (c in 1:length(levels(as.factor(P)))) {
        lines(rowperc[c, ], col = paleta[c])
      }
      legend("topright", levels(as.factor(P)), col = paleta, lty = 2, cex = 0.6)
      dev.off()
      
      # [ORIGINAL 7]: Proporciones colperc (Condicionadas a Columna) - Eje de Categoría (dades[,k])
      png(make_filename("prop_cond_col_var_x", var_name, "7"), width = 800, height = 600)
      plot(marg_x, type = "n", ylim = c(0, 1), main = paste("Prop. of pos & neg by", names(dades)[k]), las = 3)
      paleta <- rainbow(length(levels(as.factor(P))))
      for (c in 1:length(levels(as.factor(P)))) {
        lines(colperc[c, ], col = paleta[c])
      }
      dev.off()
      
      # [ORIGINAL 8]: Proporciones colperc (con leyenda) - Eje de Categoría (dades[,k])
      png(make_filename("prop_cond_col_var_x", var_name, "8_legend"), width = 800, height = 600)
      plot(marg_x, type = "n", ylim = c(0, 1), main = paste("Prop. of pos & neg by", names(dades)[k]), las = 3)
      for (c in 1:length(levels(as.factor(P)))) {
        lines(colperc[c, ], col = paleta[c])
      }
      legend("topright", levels(as.factor(P)), col = paleta, lty = 2, cex = 0.6)
      dev.off()
      
      # Se imprime la tabla (sin gráficas)
      table <- table(dades[, k], P)
      print("Cross Table:")
      print(table)
      print("Distribucions condicionades a columnes:")
      print(colperc)
      
      # [ORIGINAL 9]: Diagramas de barras apiladas (recuentos)
      png(make_filename("stacked_barplot_counts", var_name, "9"), width = 800, height = 600)
      paleta <- rainbow(length(levels(dades[, k])))
      barplot(table(dades[, k], as.factor(P)), beside = FALSE, col = paleta)
      dev.off()
      
      # [ORIGINAL 10]: Diagramas de barras apiladas (con leyenda)
      png(make_filename("stacked_barplot_counts", var_name, "10_legend"), width = 800, height = 600)
      barplot(table(dades[, k], as.factor(P)), beside = FALSE, col = paleta)
      legend("topright", levels(as.factor(dades[, k])), pch = 1, cex = 0.5, col = paleta)
      dev.off()
      
      # [ORIGINAL 11]: Diagramas de barras adosadas (recuentos)
      png(make_filename("grouped_barplot_counts", var_name, "11"), width = 800, height = 600)
      barplot(table(dades[, k], as.factor(P)), beside = TRUE, col = paleta)
      dev.off()
      
      # [ORIGINAL 12]: Diagramas de barras adosadas (con leyenda)
      png(make_filename("grouped_barplot_counts", var_name, "12_legend"), width = 800, height = 600)
      barplot(table(dades[, k], as.factor(P)), beside = TRUE, col = paleta)
      legend("topright", levels(as.factor(dades[, k])), pch = 1, cex = 0.5, col = paleta)
      dev.off()
      
      print("Test de Fisher: ")
      print(fisher.test(dades[, k], as.factor(P), simulate.p.value = TRUE, B = 2000, workspace = 2000000))
      
      print("valorsTest:")
      print(ValorTestXquali(P, dades[, k]))
      # calcular els pvalues de les quali
    }
  }
} # endfor

# Finalización y resumen
for (c in 1:length(levels(as.factor(P)))) {
  if (!is.na(levels(as.factor(P))[c])) {
    print(paste("P.values per class:", levels(as.factor(P))[c]))
    print(sort(pvalk[c, ]), digits = 3)
  }
}

cat(paste("\n¡Proceso completado! Todas las gráficas han sido guardadas en la carpeta:", output_dir, "\n"))

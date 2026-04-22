# =========================
# K-MEANS (k = 3)
# =========================

library(dplyr)
library(tidyverse)
library(ggplot2)

df <- read.csv("/Users/nastya/Desktop/Spring_2026/Stats Learning/Insurance Risk Algorithm/medical_insurance.csv")

set.seed(123) # set seed for repetition

df <- df %>%
  filter(age > 18) # filter for only adults



# Data frame with 16 key features instead of 54
# clust_df_og <- df %>%
#   dplyr::select(
#     age,
#     income,
#     bmi,
#     visits_last_year,
#     hospitalizations_last_3yrs,
#     days_hospitalized_last_3yrs,
#     medication_count,
#     systolic_bp,
#     diastolic_bp,
#     ldl,
#     hba1c,
#     annual_medical_cost,
#     claims_count,
#     avg_claim_amount,
#     total_claims_paid,
#     chronic_count
#   )

# Less correlation btw variables: kept important vars on health status, utilization and cost
clust_df <- df %>%
  dplyr::select(
    age,
    income,
    bmi,
    visits_last_year,
    days_hospitalized_last_3yrs,
    avg_claim_amount,
    chronic_count
  )

clust_scaled <- scale(clust_df) # scale the data frame

#Could change all variables to numeric to find proxies for cat vars
cor_matrix <- cor(clust_scaled)
library(corrplot)
corrplot(cor_matrix, method = "circle")

# corr_signif <- as.data.frame(cor_matrix) %>%
#   filter(if_any(everything(), ~ abs(.) >= 0.4 & abs(.) < 1))
# 
# View(corr_signif)

# Elbow method to select k
wss <- numeric(10)


for (k in 1:10) {
  wss[k] <- kmeans(clust_scaled, centers = k, nstart = 25)$tot.withinss
}

plot(
  1:10, wss, type = "b", pch = 19,
  xlab = "Number of Clusters (k)",
  ylab = "Total Within-Cluster Sum of Squares",
  main = "Elbow Method"
)

# run k-means with k = 3
kmeans_result <- kmeans(clust_scaled, centers = 3, nstart = 25)

# Add clusters
clust_df$cluster <- as.factor(kmeans_result$cluster)

# Cluster Sizes
table(clust_df$cluster)

# Cluster summary and means
cluster_summary <- clust_df %>%
  group_by(cluster) %>%
  summarise(across(everything(), mean))

print(cluster_summary)

# Unscale Centers
centers_unscaled <- sweep(
  kmeans_result$centers,
  2,
  attr(clust_scaled, "scaled:scale"),
  "*"
)

centers_unscaled <- sweep(
  centers_unscaled,
  2,
  attr(clust_scaled, "scaled:center"),
  "+"
)

centers_unscaled <- as.data.frame(centers_unscaled)
centers_unscaled$cluster <- rownames(centers_unscaled)

print(centers_unscaled)


# Run PCA
pca <- prcomp(clust_scaled)

pca_df <- data.frame(
  PC1 = pca$x[, 1],
  PC2 = pca$x[, 2],
  cluster = as.factor(kmeans_result$cluster)
)

summary(pca)


library(ggplot2)

# Extract % variance explained
var_explained <- pca$sdev^2 / sum(pca$sdev^2) * 100

scree_df <- data.frame(
  PC    = factor(paste0("PC", seq_along(var_explained)), 
                 levels = paste0("PC", seq_along(var_explained))),
  var   = var_explained
)

# Extract % variance explained
var_explained <- pca$sdev^2 / sum(pca$sdev^2) * 100

scree_df$cumvar <- cumsum(var_explained)

ggplot(scree_df, aes(x = PC, y = cumvar, group = 1)) +
  geom_line(color = "steelblue", linewidth = 0.8) +
  geom_point(size = 2, color = "steelblue") +
  geom_hline(yintercept = 80, linetype = "dashed", color = "red") + # 80% threshold
  labs(title = "Cumulative Variance Explained",
       x = "Principal Component",
       y = "Cumulative % Variance") +
  theme_minimal()

# Keep first 5 PCs
pc_scores <- as.data.frame(pca$x[, 1:5])

# Then fit
kmeans_pc5 <- kmeans(pc_scores, centers = 3, nstart = 25)
pc_scores$cluster <- as.factor(kmeans_pc5$cluster)

library(ggplot2)

ggplot(pc_scores, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(alpha = 0.7, size = 3) +
  labs(title = "K-Means Clusters on PC1 vs PC2") +
  theme_minimal()

seed(123)

# Or with fviz_cluster — pass only the 5 PCs (no cluster column)
fviz_cluster(kmeans_result, data = pc_scores[, 1:5],
             geom = "point", ellipse.type = "convex",
             palette = "jco", ggtheme = theme_minimal())

# Loadings tell you which original variables drive each PC
loadings <- as.data.frame(pca$rotation[, 1:5])
print(loadings)

# Easier to read — show top contributing variables per PC
library(factoextra)
fviz_contrib(pca, choice = "var", axes = 1)  # PC1
fviz_contrib(pca, choice = "var", axes = 2)  # PC2
fviz_contrib(pca, choice = "var", axes = 1:5) # across all 5


# Profile each cluster using original variables
cluster_summary_pca5 = clust_df |>
  group_by(cluster) |>
  summarise(across(where(is.numeric), mean))
# 
# # plot pca
# ggplot(pca_df, aes(x = PC1, y = PC2, color = cluster)) +
#   geom_point(alpha = 0.6) +
#   labs(title = "K-means Clusters (k = 3, PCA Projection)") +
#   theme_minimal()
# 
# # # Plot Boxplot for Annual Medical Cost by Clusters
# # ggplot(clust_df, aes(cluster, annual_medical_cost, fill = cluster)) +
# #   geom_boxplot() +
# #   labs(title = "Annual Medical Cost by Cluster") +
# #   theme_minimal()
# 
# # Plot Boxplot for Avg Claim Amount by Clusters
# ggplot(clust_df, aes(cluster, avg_claim_amount, fill = cluster)) +
#   geom_boxplot() +
#   labs(title = "Avg Claim Amount by Cluster") +
#   theme_minimal()
# 
# 
# # Plot Boxplot for Chronic Conditions by Cluster
# ggplot(clust_df, aes(cluster, chronic_count, fill = cluster)) +
#   geom_boxplot() +
#   labs(title = "Chronic Conditions by Cluster") +
#   theme_minimal()
# 
# # Evaluate Clusters with the is high risk variable
# if ("is_high_risk" %in% names(clust_df)) {
#   clust_df %>%
#     group_by(cluster) %>%
#     summarise(high_risk_rate = mean(is_high_risk)) %>%
#     print()
# }

# =========================
# Data Setup
# =========================

library(dplyr)
library(tidyverse)
library(ggplot2)
library(corrplot)
library(e1071)

set.seed(123) # set seed for repetition

df <- read.csv("/Users/nastya/Desktop/Spring_2026/Stats Learning/Insurance Risk Algorithm/medical_insurance.csv")

df <- df %>%
  filter(age > 18) # filter for only adults

# Data frame with 16 key features instead of 54
clust_df <- df %>%
  dplyr::select(
    age,
    income,
    bmi,
    visits_last_year,
    hospitalizations_last_3yrs,
    days_hospitalized_last_3yrs,
    medication_count,
    systolic_bp,
    diastolic_bp,
    ldl,
    hba1c,
    annual_medical_cost,
    claims_count,
    avg_claim_amount,
    total_claims_paid,
    chronic_count
  ) 

# Take log of heavily skewed vars
clust_df_log <- clust_df %>% 
  mutate(across(c(income,
                  visits_last_year, 
                  hospitalizations_last_3yrs,
                  days_hospitalized_last_3yrs,
                  hba1c,
                  annual_medical_cost,
                  claims_count,
                  avg_claim_amount,
                  total_claims_paid,
                  ), log1p)) %>% 
  rename(hosp_last3yrs = hospitalizations_last_3yrs, days_hosp_last3yrs = days_hospitalized_last_3yrs)

clust_scaled <- scale(clust_df_log) # scale the data frame

# =========================
# Elbow Method
# =========================

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

# =========================
# K-MEANS (k = 3)
# =========================

# Run k-means with k = 3
kmeans_result <- kmeans(clust_scaled, centers = 3, nstart = 25)

# Attach clusters back to ORIGINAL (untransformed) data for profiling
clust_df$cluster <- kmeans_result$cluster

clust_df$is_high_risk <-df$is_high_risk
cluster_summary <- clust_df %>% 
  group_by(cluster) %>% 
  summarise(across(everything(), mean)) %>% 
  mutate(across(where(is.numeric), \(x) round(x, 2)))
View(cluster_summary)

# Cluster Sizes
table(clust_df$cluster)

# =========================
# Run PCA K-Means and Graphs
# =========================

# Run PCA
pca <- prcomp(clust_scaled)

pca_df <- data.frame(
  PC1 = pca$x[, 1],
  PC2 = pca$x[, 2],
  cluster = as.factor(kmeans_result$cluster)
)

summary(pca)

# plot pca 1 and 2
ggplot(pca_df, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(alpha = 0.6) +
  labs(title = "K-means Clusters (k = 3, PCA Projection)") +
  theme_minimal()

##SHOULD PROBABLY NOT INCLUDE THESE: HARD TO SEE DIFFERENCES
# Plot Boxplot for Avg Claim Amount by Clusters
ggplot(clust_df, aes(cluster, avg_claim_amount, fill = cluster)) +
  geom_boxplot() +
  labs(title = "Avg Claim Amount by Cluster") +
  theme_minimal()

# Plot Boxplot for Chronic Conditions by Cluster
ggplot(clust_df, aes(cluster, chronic_count, fill = cluster)) +
  geom_boxplot() +
  labs(title = "Chronic Conditions by Cluster") +
  theme_minimal()

# Assuming clust_df has your cluster assignments and your original df 
result <- clust_df %>%
  group_by(cluster) %>%
  summarise(high_risk_rate = mean(is_high_risk)) %>% 
  ungroup()
print(result)

# =========================
# PCA Variable Contribution
# =========================

# Loadings tell you which original variables drive each PC
loadings <- as.data.frame(pca$rotation[, 1:5])
print(loadings)

# Easier to read — show top contributing variables per PC
library(factoextra)
fviz_contrib(pca, choice = "var", axes = 1)  # PC1: claims amounts and visit count most influential
fviz_contrib(pca, choice = "var", axes = 2)  # PC2: time spent in hospital
fviz_contrib(pca, choice = "var", axes = 1:5) # across all 5


# =========================
# Fuzzy K-MEANS (k = 3)
# =========================

set.seed(123)
# fuzzy k-means fitting
fuzzy_result <- cmeans(
  clust_scaled,
  centers = 3,     # same k as k-means
  m = 1.3,           # according to burt paper, 1.2 worked well for insurance
  iter.max = 100
)

# Needs membership matrix
membership <- fuzzy_result$membership
initial_centers <- fuzzy_result$centers
final_centers <- t(fuzzy_result$centers)

# Each row sums to 1
head(membership)

# Assign points to highest membership clusters
fuzzy_cluster <- apply(membership, 1, which.max)

# Add to dataset
clust_df$fuzzy_cluster <- as.factor(fuzzy_cluster)

# How many points in each cluster
table(clust_df$fuzzy_cluster)

# Cluster Means
fuzzy_summary <- clust_df %>%
  select(-c(cluster)) %>% 
  group_by(fuzzy_cluster) %>%
  summarise(across(where(is.numeric), mean)) %>% 
  mutate(across(where(is.numeric), \(x) round(x, 2)))

print(fuzzy_summary)

# =========================
# Run PCA Fuzzy and Plot
# =========================

pca_fuzzy <- prcomp(clust_scaled)

pca_fuzzy_df <- data.frame(
  PC1 = pca_fuzzy$x[,1],
  PC2 = pca_fuzzy$x[,2],
  cluster = as.factor(fuzzy_cluster)
)

ggplot(pca_fuzzy_df, aes(PC1, PC2, color = cluster)) +
  geom_point(alpha = 0.6) +
  labs(title = "Fuzzy K-means Clusters (PCA Projection)") +
  theme_minimal()


# Get how ambiguous each point is (all nearly 1/k probability)
max_membership <- apply(membership, 1, max)

summary(max_membership)

# Get total uncertain points
uncertain_points <- sum(max_membership < 0.7)
uncertain_points

# Make a histogram of maximum membership probability
hist(max_membership, breaks = 30,
     main = "Cluster Membership Confidence",
     xlab = "Max Membership Probability")

# Print Risk
result2 <- clust_df %>%
  group_by(fuzzy_cluster) %>%
  summarise(high_risk_rate = mean(is_high_risk)) %>% 
  ungroup()
print(result2)

# =============================
# DATA Exploration: Find best M
# =============================

install.packages("clusterCrit")
library(clusterCrit)

m_values <- seq(1.1, 3, by = 0.1)

results <- lapply(m_values, function(m) {
  fit <- cmeans(clust_scaled, centers = 3, m = m, iter.max = 100)
  
  # Partition coefficient — higher is better (more crisp)
  pc <- fit$membership^2 |> sum() / nrow(clust_scaled)
  
  # Partition entropy — lower is better (less fuzzy)
  pe <- -sum(fit$membership * log(fit$membership)) / nrow(clust_scaled)
  
  data.frame(m = m, partition_coef = pc, partition_entropy = pe)
})

results_df <- do.call(rbind, results)
print(results_df)

# 1.2 is the better m according to pc > 0.7 and pe < log(k)/2 (median membership around 0.9)
# However 1.3 is fuzzier and has lower mean/median max membership of around 0.7

results_df |>
  pivot_longer(-m, names_to = "metric", values_to = "value") |>
  ggplot(aes(x = m, y = value, color = metric)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_wrap(~metric, scales = "free_y") +
  labs(title = "Choosing m for Fuzzy C-Means",
       x = "Fuzziness parameter (m)") +
  theme_minimal()

install.packages("bluster")
library(bluster)
sil_approx <- approxSilhouette(pca$x[, 1:5], kmeans_result$cluster)
mean(sil_approx$width)

# =============================
# Graph differences btw Methods
# =============================

# Add method column to each summary
cluster_summary$method <- "K-Means"
cluster_summary$cluster  <- as.character(cluster_summary$cluster)
fuzzy_summary$method <- "Fuzzy C-Means"

# Standardize cluster column name
fuzzy_summary$cluster  <- as.character(fuzzy_summary$fuzzy_cluster)
fuzzy_summary$fuzzy_cluster <- NULL
library(ggplot2)
library(tidyr)

bind_rows(cluster_summary, fuzzy_summary) |>
  select(-any_of("fuzzy_cluster")) |>  # drop it if it exists
  mutate(cluster = paste0("Cluster ", cluster)) |>
  pivot_longer(cols      = -c(cluster, method),
               names_to  = "variable",
               values_to = "value") |>
  group_by(variable) |>
  mutate(value_scaled = scale(value)) |>  # normalize per variable so all are comparable
  ungroup() |>
  ggplot(aes(x = cluster, y = variable, fill = value_scaled)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = round(value, 1)), size = 3) +  # show original values as labels
  facet_wrap(~method) +
  scale_fill_gradient2(low  = "steelblue",
                       mid  = "white",
                       high = "coral",
                       midpoint = 0) +
  labs(title = "Cluster Profiles: K-Means vs Fuzzy C-Means",
       x     = "Cluster",
       y     = NULL,
       fill  = "Scaled Value") +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title      = element_text(face = "bold", hjust = 0.5))

# =========================
# Other Code
# =========================

# skewness(clust_df$chronic_count)
# 
# # 1) Find Correlations and Subset
# cor_matrix_cont <- cor(clust_scaled_cont) #Could change all variables to numeric to find proxies for cat vars
# corrplot(cor_matrix_cont)
# 
# install.packages("caret")
# library(caret)
# high_cor <- findCorrelation(cor_matrix, cutoff = 0.75)
# colnames(clust_df)[high_cor]  # variables to consider removing

# Less correlation btw variables: kept important vars on health status, utilization and cost
# clust_df <- clust_df_cont %>%
#   dplyr::select(
#     age,
#     income,
#     bmi,
#     visits_last_year,
#     days_hospitalized_last_3yrs,
#     avg_claim_amount,
#     chronic_count
#   ) %>%
#   mutate(across(c(income,
#                   visits_last_year,
#                   days_hospitalized_last_3yrs,
#                   avg_claim_amount,
#   ), log1p))
# 
# library(moments)
# 
# 
# # Correlation after subsetting
# clust_scaled <- scale(clust_df) # scale the data frame
# cor_matrix <- cor(clust_scaled) #Could change all variables to numeric to find proxies for cat vars
# corrplot(cor_matrix)

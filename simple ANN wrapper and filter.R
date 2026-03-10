# ============================================================
#  SINGLE-GENE ANN (Keras 3) + FILTER + WRAPPER
#  - Runs on your Windows file:
#      C:/Graham/Test2/GSE8479_series_age_for analysis.csv
#  - Binary class column: "Class" (Disease vs healthy)
#  - One gene at a time
#  - 5x Monte Carlo cross-validations (stratified random splits)
#  - Correct scaling (fit on TRAIN only; apply to TEST)
#  - Writes report CSVs to the WORKING DIRECTORY
#
#  REQUIREMENTS:
#    install.packages(c("readr","dplyr","tibble","pROC","keras","tensorflow"))
# ============================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tibble)
  library(pROC)
  library(keras)
  library(tensorflow)
})

# ---------------------------
# CONFIG (WINDOWS PATH)
# ---------------------------
INFILE    <- "C:/Graham/Test2/GSE8479_series_age_for analysis.csv"
CLASS_COL <- "Class"
POS_LABEL <- "Disease"

# Monte Carlo CV settings
N_MCCV     <- 5
TRAIN_FRAC <- 0.70
SEED       <- 123

# Filter step (optional): keep only the top N genes by univariate AUC on full dataset
USE_FILTER <- TRUE
TOP_N      <- 2000  # (if you have fewer genes than this, it keeps them all)

# Keras training settings (small net because 1D input)
EPOCHS     <- 60
BATCH_SIZE <- 16
PATIENCE   <- 8
VERBOSE    <- 0

set.seed(SEED)

# ---------------------------
# HELPERS
# ---------------------------
safe_auc <- function(y_true01, y_prob) {
  y_true01 <- as.numeric(y_true01)
  y_prob   <- as.numeric(y_prob)
  if (length(unique(y_true01)) < 2) return(NA_real_)
  suppressWarnings(as.numeric(pROC::auc(pROC::roc(y_true01, y_prob, quiet = TRUE))))
}

safe_accuracy <- function(y_true01, y_prob, thr = 0.5) {
  y_true01 <- as.numeric(y_true01)
  y_prob   <- as.numeric(y_prob)
  if (length(y_true01) == 0) return(NA_real_)
  y_hat <- ifelse(y_prob >= thr, 1, 0)
  mean(y_hat == y_true01)
}

scale_train_apply <- function(x_train, x_test) {
  mu <- mean(x_train, na.rm = TRUE)
  sdv <- sd(x_train, na.rm = TRUE)
  if (is.na(sdv) || sdv == 0) sdv <- 1
  list(
    train = (x_train - mu) / sdv,
    test  = (x_test  - mu) / sdv,
    mu = mu,
    sd = sdv
  )
}

# Keras 3 safe: avoid layer_input()/Input(); define input_shape only once
build_single_gene_model <- function() {
  model <- keras_model_sequential() |>
    layer_dense(units = 8, activation = "relu", input_shape = c(1)) |>
    layer_dropout(rate = 0.2) |>
    layer_dense(units = 4, activation = "relu") |>
    layer_dense(units = 1, activation = "sigmoid")
  
  model |>
    compile(
      optimizer = optimizer_adam(learning_rate = 1e-3),
      loss = "binary_crossentropy",
      metrics = c("accuracy")
    )
  model
}

mc_split_stratified <- function(y01, train_frac = 0.7) {
  idx_pos <- which(y01 == 1)
  idx_neg <- which(y01 == 0)
  
  n_pos_train <- max(1L, floor(length(idx_pos) * train_frac))
  n_neg_train <- max(1L, floor(length(idx_neg) * train_frac))
  
  train_idx <- c(sample(idx_pos, n_pos_train), sample(idx_neg, n_neg_train))
  test_idx  <- setdiff(seq_along(y01), train_idx)
  list(train = train_idx, test = test_idx)
}

# ---------------------------
# LOAD DATA
# ---------------------------
if (!file.exists(INFILE)) {
  stop(
    sprintf(
      "Input file not found:\n  %s\n\nCheck the exact filename with:\n  list.files('C:/Graham/Test2')",
      INFILE
    )
  )
}

df <- readr::read_csv(INFILE, show_col_types = FALSE) |> as.data.frame()
if (!(CLASS_COL %in% names(df))) stop(sprintf("CLASS_COL '%s' not found in file.", CLASS_COL))

# Convert class to 0/1
y_raw <- as.character(df[[CLASS_COL]])
y01 <- ifelse(y_raw == POS_LABEL, 1, 0)
if (length(unique(y01)) < 2) stop("Class column must contain both classes.")

# Select numeric feature columns only (genes); excludes non-numeric like 'case'
gene_candidates <- setdiff(names(df), CLASS_COL)
is_num <- vapply(df[gene_candidates], is.numeric, logical(1))
genes_all <- gene_candidates[is_num]
if (length(genes_all) == 0) stop("No numeric gene columns detected (check your CSV).")

# QC: remove high-NA and zero-variance genes
na_frac <- vapply(df[genes_all], function(x) mean(is.na(x)), numeric(1))
var0 <- vapply(df[genes_all], function(x) {
  x2 <- x[!is.na(x)]
  if (length(x2) < 2) TRUE else (sd(x2) == 0)
}, logical(1))

genes_all <- genes_all[na_frac <= 0.20 & !var0]
if (length(genes_all) == 0) stop("All genes removed by NA/zero-variance filters. Relax thresholds.")

# ---------------------------
# FILTER STEP (univariate AUC)
# ---------------------------
filter_tbl <- tibble(Gene = genes_all, FilterAUC = NA_real_)

if (USE_FILTER) {
  message("Running univariate AUC filter on full dataset...")
  for (i in seq_along(genes_all)) {
    g <- genes_all[i]
    x <- df[[g]]
    ok <- !is.na(x)
    if (sum(ok) < 10) next
    
    x_ok <- x[ok]
    xz <- (x_ok - mean(x_ok)) / (sd(x_ok) + 1e-8)
    score01 <- (xz - min(xz)) / (max(xz) - min(xz) + 1e-8)
    
    filter_tbl$FilterAUC[i] <- safe_auc(y01[ok], score01)
  }
  filter_tbl <- filter_tbl |> arrange(desc(FilterAUC))
  genes <- head(filter_tbl$Gene, TOP_N)
} else {
  genes <- genes_all
}

write_csv(filter_tbl, file.path(getwd(), "FILTER_univariate_auc.csv"))

# ---------------------------
# WRAPPER STEP (single-gene ANN over MCCV)
# ---------------------------
perf_rows <- vector("list", length(genes) * N_MCCV)
pred_rows <- vector("list", length(genes) * N_MCCV)
k <- 1L

message(sprintf("Running wrapper ANN: %d genes x %d MCCV = %d fits",
                length(genes), N_MCCV, length(genes) * N_MCCV))

for (g in genes) {
  x_full <- df[[g]]
  
  for (iter in seq_len(N_MCCV)) {
    
    split <- mc_split_stratified(y01, TRAIN_FRAC)
    train_idx <- split$train
    test_idx  <- split$test
    
    # Drop NAs per split
    train_ok <- train_idx[!is.na(x_full[train_idx])]
    test_ok  <- test_idx[!is.na(x_full[test_idx])]
    
    # Must have both classes in train and test
    if (length(train_ok) < 10 ||
        length(test_ok) < 5 ||
        length(unique(y01[train_ok])) < 2 ||
        length(unique(y01[test_ok])) < 2) {
      
      perf_rows[[k]] <- tibble(
        Gene = g, Iteration = iter,
        N_train = length(train_ok), N_test = length(test_ok),
        TrainAUC = NA_real_, TestAUC = NA_real_,
        TrainAcc = NA_real_, TestAcc = NA_real_,
        ScaleMean = NA_real_, ScaleSD = NA_real_,
        Notes = "Skipped: insufficient data or single-class after NA removal"
      )
      pred_rows[[k]] <- tibble()
      k <- k + 1L
      next
    }
    
    # Scale train only -> apply to test
    scaled <- scale_train_apply(x_full[train_ok], x_full[test_ok])
    x_train <- matrix(scaled$train, ncol = 1)
    y_train <- matrix(y01[train_ok], ncol = 1)
    x_test  <- matrix(scaled$test,  ncol = 1)
    y_test  <- matrix(y01[test_ok], ncol = 1)
    
    keras::k_clear_session()
    model <- build_single_gene_model()
    
    model |> fit(
      x = x_train, y = y_train,
      validation_split = 0.2,
      epochs = EPOCHS,
      batch_size = BATCH_SIZE,
      verbose = VERBOSE,
      callbacks = list(
        callback_early_stopping(monitor = "val_loss", patience = PATIENCE, restore_best_weights = TRUE)
      )
    )
    
    # Predict
    p_train <- as.numeric(model |> predict(x_train, verbose = 0))
    p_test  <- as.numeric(model |> predict(x_test,  verbose = 0))
    
    # Metrics
    perf_rows[[k]] <- tibble(
      Gene = g, Iteration = iter,
      N_train = nrow(x_train), N_test = nrow(x_test),
      TrainAUC = safe_auc(y_train, p_train),
      TestAUC  = safe_auc(y_test,  p_test),
      TrainAcc = safe_accuracy(y_train, p_train, 0.5),
      TestAcc  = safe_accuracy(y_test,  p_test,  0.5),
      ScaleMean = scaled$mu,
      ScaleSD   = scaled$sd,
      Notes = ""
    )
    
    pred_rows[[k]] <- tibble(
      Gene = g, Iteration = iter,
      SampleIndex = c(train_ok, test_ok),
      Split = c(rep("Train", length(train_ok)), rep("Test", length(test_ok))),
      y_true = c(as.numeric(y_train), as.numeric(y_test)),
      y_prob = c(p_train, p_test)
    )
    
    k <- k + 1L
  }
}

perf <- bind_rows(perf_rows)
pred <- bind_rows(pred_rows)

summary_per_gene <- perf |>
  group_by(Gene) |>
  summarise(
    N_runs = sum(!is.na(TestAUC)),
    Median_TestAUC = median(TestAUC, na.rm = TRUE),
    Mean_TestAUC   = mean(TestAUC, na.rm = TRUE),
    SD_TestAUC     = sd(TestAUC, na.rm = TRUE),
    Median_TestAcc = median(TestAcc, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(desc(Median_TestAUC), desc(Median_TestAcc))

# ---------------------------
# WRITE REPORTS (to working directory)
# ---------------------------
write_csv(perf, file.path(getwd(), "WRAPPER_single_gene_ANN_MCCV_performance.csv"))
write_csv(summary_per_gene, file.path(getwd(), "WRAPPER_single_gene_ANN_MCCV_summary_per_gene.csv"))
write_csv(pred, file.path(getwd(), "WRAPPER_single_gene_ANN_MCCV_predictions.csv"))

message("Done. Wrote to working directory:")
message("  FILTER_univariate_auc.csv")
message("  WRAPPER_single_gene_ANN_MCCV_performance.csv")
message("  WRAPPER_single_gene_ANN_MCCV_summary_per_gene.csv")
message("  WRAPPER_single_gene_ANN_MCCV_predictions.csv")

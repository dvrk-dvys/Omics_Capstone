# ---- Load Required Libraries ----
if (!requireNamespace("TCGAbiolinks", quietly = TRUE)) {
  install.packages("BiocManager")
  BiocManager::install("TCGAbiolinks")
}
library(TCGAbiolinks)

# ---- Setup Paths ----
manifest_path <- "C:/Graham/Healthy tissues/Liver/gdc_manifest.2025-06-07.111044.txt"
output_dir <- "C:/Graham/Healthy tissues/Liver"
gdc_client_path <- "C:/GDC/gdc-client.exe"

if (!file.exists(gdc_client_path)) {
  stop("ERROR: gdc-client.exe not found at the specified path. Please verify 'gdc_client_path'.")
}

# ---- Define a function to check for incomplete downloads ----
check_incomplete_files <- function(dir) {
  incomplete_files <- list.files(dir, pattern = "\\.incomplete$", recursive = TRUE, full.names = TRUE)
  return(incomplete_files)
}

max_retries <- 5
retry_count <- 0
incomplete_files <- NULL

repeat {
  cat("\nAttempting to download files (try ", retry_count + 1, " of ", max_retries, ")...\n", sep = "")
  
  download_command <- paste0('"', gdc_client_path, '" download -m "', manifest_path, '" -d "', output_dir, '"')
  exit_code <- system(download_command)
  
  # Check for incomplete files
  incomplete_files <- check_incomplete_files(output_dir)
  
  if (length(incomplete_files) == 0) {
    cat("\n✅ All files downloaded successfully.\n")
    break
  } else {
    cat("\n⚠️ Found ", length(incomplete_files), " incomplete files. Retrying download for missing files...\n", sep = "")
    retry_count <- retry_count + 1
  }
  
  if (retry_count >= max_retries) {
    cat("\n❌ Maximum retries reached. Some files may not have been downloaded correctly.\n")
    cat("Incomplete files:\n")
    print(incomplete_files)
    break
  }
}
# ---- Load Required Libraries ----
if (!requireNamespace("TCGAbiolinks", quietly = TRUE)) {
  BiocManager::install("TCGAbiolinks")
}
library(TCGAbiolinks)

# ---- Setup ----
# Path to your GDC manifest file
manifest_path <- "C:/Graham/Healthy tissues/Liver/gdc_manifest.2025-06-07.111044.txt"

# Output directory
output_dir <- "C:/Graham/Healthy tissues/Liver"

gdc_client_path <- "C:/GDC/gdc-client.exe"
system(paste0('"', gdc_client_path, '" download -m "', manifest_path, '" -d "', output_dir, '"')) 
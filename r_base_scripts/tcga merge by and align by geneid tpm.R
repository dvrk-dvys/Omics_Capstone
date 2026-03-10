library(dplyr)
library(readr)
library(purrr)

# Directory path
dir_path <- "C:/Graham/Healthy tissues/Liver/raw data"  # <-- Change to your folder
file_list <- list.files(path = dir_path, pattern = "\\.tsv$", full.names = TRUE)

# Function to process each file
process_file <- function(file) {
  # Read raw lines
  raw <- readLines(file, warn = FALSE)
  
  # Remove lines 1, 3, 4, 5, 6 if they exist
  skip_lines <- c(1, 3, 4, 5, 6)
  skip_lines <- skip_lines[skip_lines <= length(raw)]
  keep_rows <- raw[-skip_lines]
  
  # Parse the kept lines
  df <- tryCatch({
    read.delim(text = paste(keep_rows, collapse = "\n"), stringsAsFactors = FALSE)
  }, error = function(e) {
    warning(paste("Error reading", file, ":", e$message))
    return(NULL)
  })
  
  # Check for required columns
  if (is.null(df) || !all(c("gene_id", "gene_name", "tpm_unstranded") %in% names(df))) {
    warning(paste("Missing columns in", file))
    return(NULL)
  }
  
  # Sort and rename
  base <- tools::file_path_sans_ext(basename(file))
  df <- df %>%
    select(gene_id, gene_name, tpm_unstranded) %>%
    arrange(gene_id) %>%
    rename_with(~ paste0(., "_", base), .cols = -gene_id)
  
  return(df)
}

# Process all files
processed_list <- lapply(file_list, process_file)
processed_list <- Filter(Negate(is.null), processed_list)

# Merge all on gene_id using reduce + full_join
merged_df <- reduce(processed_list, full_join, by = "gene_id")

# OPTIONAL: Insert blank columns between datasets for clarity
insert_blanks <- function(df) {
  cols <- colnames(df)
  result <- df["gene_id"]
  groups <- split(cols[-1], ceiling(seq_along(cols[-1]) / 2))  # Group gene_name + tpm
  for (g in groups) {
    result <- cbind(result, df[g], blank = NA)
  }
  return(result)
}
merged_df <- insert_blanks(merged_df)

# Write output
write.csv(merged_df, "C:/Graham/Healthy tissues/Liver/raw data/merged_output.csv", row.names = FALSE)
# Load necessary library
library(readr)

# Select your file
file_path <- file.choose()  # Manually select the CSV file

# Read the data
data <- read_csv(file_path)

# Choose a directory to save the output files
output_dir <- choose.dir()  # Use choose.dir() on Windows

# Check that output directory is selected
if (is.na(output_dir)) {
  stop("No output directory selected. Exiting.")
}

# How many middle columns
num_files <- ncol(data) - 2

# Loop through each middle column
for (i in 2:(ncol(data) - 1)) {
  # Extract first, current, and last columns
  subset_data <- data[, c(1, i, ncol(data))]
  
  # Get column name
  column_name <- names(data)[i]
  
  # Make filename safe
  safe_column_name <- gsub("[^A-Za-z0-9_]", "_", column_name)
  
  # Create a padded number (e.g., 01, 02, 03, etc.)
  file_number <- sprintf("%02d", i - 1)  # "-1" because i starts at 2
  
  # Final filename
  output_filename <- file.path(output_dir, paste0(file_number, "_", safe_column_name, ".csv"))
  
  # Write the subset to a CSV file
  write_csv(subset_data, output_filename)
  
  cat("Saved:", output_filename, "\n")
}

cat("All files have been saved to:", output_dir, "\n") 
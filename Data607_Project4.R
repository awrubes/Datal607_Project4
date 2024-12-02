library(tidyverse)
library(tidytext)
library(readtext)
library(dplyr)
library(stringr)
library(rvest)
library(tm)
library(e1071)

repofolder <- "/Users/alliewrubel/Documents/GitHub/Datal607_Project4"
ham_folder <- paste0(repofolder, "/easy_ham/")
spam_folder <- paste0(repofolder, "/spam/")

# Get list of files
ham_files <- list.files(ham_folder, full.names = TRUE)
spam_files <- list.files(spam_folder, full.names = TRUE)


read_emails_to_df <- function(file_paths, label) {
  email_list <- lapply(file_paths, function(file) {
    email_content <- readLines(file, warn = FALSE)
    
    # Identify the first blank line and extract the content after it
    body_start <- which(email_content == "")[1] + 1
    if (!is.na(body_start)) {
      email_body <- email_content[body_start:length(email_content)] 
      email_text <- paste(email_body, collapse = " ")  
    } else {
      email_text <- NA  # No body found
    }
    
    return(email_text)
  })
  
  # Create dataframe
  email_df <- data.frame(
    text = unlist(email_list),  # Convert list to character vector
    spam = label,
    stringsAsFactors = FALSE
  )
  
  return(email_df)
}

ham_df <- read_emails_to_df(ham_files, label = 0)
spam_df <- read_emails_to_df(spam_files, label = 1)
email_data_raw <- rbind(ham_df, spam_df)


#function to clean the text portion of the email
clean_text <- function(email_text) {
  email_text <- str_squish(email_text)
  email_text <- str_replace_all(email_text, "http\\S+|www\\S+", "<URL>")
  
  email_text <- str_remove_all(email_text, "(?i)^(From|To|Subject|Date|Received|Return-Path|Delivered-To|Message-ID|X-.*|Content-.*|Mime-Version|Thread-Index|Precedence|List-Id|Errors-To):.*?(\\n\\s.*?)*")
  
  email_text <- str_remove_all(email_text, "[^[:alnum:][:punct:]\\s]")
  
  email_text <- str_replace_all(email_text, "\\b\\d+\\b", "<NUM>")
  
  email_text <- tolower(email_text)
  
  email_text <- str_squish(email_text)
  return(email_text)
}


email_data_cleaned <- email_data_raw %>%
  mutate(cleaned_text = clean_text(text))

head(email_data_cleaned)

email_data_cleaned$text <- iconv(email_data_cleaned$text, from = "latin1", to = "UTF-8", sub = "")

email_data_cleaned <- email_data_cleaned %>%
  mutate(doc_id = row_number())

# Create a corpus
corpus <- VCorpus(VectorSource(email_data_cleaned$text))
names(corpus) <- email_data_cleaned$doc_id
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)

# Create a Document-Term Matrix
dtm <- DocumentTermMatrix(corpus)
print(dim(dtm))

# Reduce sparsity
dtm_reduced <- removeSparseTerms(dtm, 0.90)
dtm_matrix <- as.matrix(dtm_reduced)

# Align labels with the reduced matrix using doc_id
labels <- email_data_cleaned %>%
  filter(doc_id %in% rownames(dtm_matrix)) %>%
  pull(spam)

set.seed(123)  # For reproducibility

# Create training and testing indices
train_indices <- sample(1:nrow(dtm_matrix), 0.8 * nrow(dtm_matrix))
test_indices <- setdiff(1:nrow(dtm_matrix), train_indices)

# Split the data
train_data <- dtm_matrix[train_indices, ]
test_data <- dtm_matrix[test_indices, ]
train_labels <- labels[train_indices]
test_labels <- labels[test_indices]

cat("Spam in Training:", sum(train_labels == 1), "\n")
cat("Ham in Training:", sum(train_labels == 0), "\n")

print(train_data)
print(train_labels)
# Train a Naive Bayes classifier
nb_model <- naiveBayes(train_data, as.factor(train_labels))

# Naive Bayes
nb_predictions <- predict(nb_model, test_data)

# Calculate accuracy
accuracy <- sum(nb_predictions == test_labels) / length(test_labels)
print(paste("Accuracy:", accuracy))


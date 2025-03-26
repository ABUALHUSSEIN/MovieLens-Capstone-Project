#Introduction

#This project is part of the HarvardX PH125.9x Data Science Capstone, where we aim to build a predictive model for movie ratings using the MovieLens dataset. 
#The MovieLens dataset is a widely used benchmark dataset in the field of recommendation systems. It contains user-movie interactions in the form of explicit ratings, ranging from 1 to 5 stars, where users express their preferences for movies. The dataset also includes metadata such as movie titles and genres (optional), which can be leveraged for content-based recommendations.
#The objective is to develop an algorithm that accurately predicts user ratings for movies while minimizing the Root Mean Square Error (RMSE). 
  

######## Step 1: Load and Prepare the Data #######
##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(knitr)
library(kableExtra)
library(dplyr)
library(ggplot2)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

######## Step 2: Exploratory Data Analysis (EDA) #######
# To see the basic information about the data, dimension,
# types and to see number of missing entries on each column.

#2.1 DataSet Structure
class(edx)
glimpse(edx)

colnames(edx)

# There are 9,000,055 obs of 6 variables "userID", "movieID", "rating", "timestamp", "title", and "genres".

###The edx dataset contains the following columns:

# quantitative features
#-userId : discrete, Unique ID for the user.
#-movieId: discrete, Unique ID for the movie.
#-timestamp : discrete , Date and time the rating was given.

#qualitative features
#-title: nominal , movie title (not unique)
#-genres: nominal, genres associated with the movie.

#outcome,y
#-rating : continuous, a rating between 0 and 5 for the movie.



edx %>%
  head(5) %>%
  kable(booktabs = TRUE, caption = "\\textcolor{blue}{Head Rows}", align = "c") %>%
  kable_styling(position = "center", latex_options = c("striped", "scale_down"))

#2.2 Check basic statistics

summary_edx <- summary(edx)

kable(summary_edx, booktabs = TRUE, caption = "Summary Statistics") %>%
  kable_styling(position = "center", latex_options = c("scale_down", "striped"))

# Count total missing values in the dataset
total_missing <- sum(is.na(edx))
colSums(is.na(edx))
cat("Total missing values in the dataset:", total_missing, "\n")

# The dataset is clean with no missing values.

# Count unique values for the "users", "movies" and "genre" variables.

num_users <- n_distinct(edx$userId)
num_movies <- n_distinct(edx$movieId)
num_genres<- n_distinct(edx$genres)
cat("Number of users:", num_users, "\n")
cat("Number of movies:", num_movies, "\n")
cat("Number of genres:", num_genres, "\n")

# The data contains 9000055 ratings applied to 10677 different movies of 797
#different unique or combination of genre from 69878 single users. 

# Following is a bar chart showing the distribution of rating.

edx %>% 
  ggplot(aes(rating)) + 
  geom_histogram(binwidth=0.2, color="darkblue", fill="lightblue") + 
  ggtitle("Rating Distribution (Training)")

summary(edx$rating)
edx %>% group_by(rating) %>% summarize(number_rows=n())

#create a dataframe "explore_edx_ratings" which contains half star and whole star ratings from the edx
group <- ifelse((edx$rating == 1 |edx$rating == 2 |edx$rating == 3 |
                   edx$rating == 4 | edx$rating == 5),
                "whole_star","half_star")
explore_edx_ratings <- data.frame(edx$rating, group)

#Distribution of Ratings Values

ggplot(explore_edx_ratings, aes(x= edx.rating, fill = group)) +
  geom_histogram( binwidth = 0.2) +
  scale_x_continuous(breaks=seq(0, 5, by= 0.5)) +
  scale_fill_manual(values = c("half_star"="red", "whole_star"="blue")) +
  labs(x="rating", y="number of ratings", caption = "source data: edx set") +
  ggtitle("Distribution of Ratings Values ") 

#geom_line showing rating
edx %>%group_by(rating) %>%summarize(count = n()) %>%ggplot(aes(x = rating, y = count)) +geom_line(color="red")+
  labs(x="rating", y="number of ratings", caption = "source data: edx set") +ggtitle("geomplot : number of ratings per count") 

#summary of five rating count
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5, count) %>%arrange(desc(count))

# Exploring ratings of the edx dataset, it is observed that the average user's activity reveals that no user gives 0 as rating, the top 5 ratings from most to least are : 4, 3, 5, 3.5 and 2. 
# The Distribution of Ratings Values shows that the halfstar ratings are less common than whole star ratings.

###qualitative features: genres, title

#Now, we are going to explore the features "genres" and "title" of our edx set.

##########Genres#########
# Using str_detect for selected genres
drama <- edx %>% filter(str_detect(genres,"Drama"))
comedy <- edx %>% filter(str_detect(genres,"Comedy"))
thriller <- edx %>% filter(str_detect(genres,"Thriller"))
romance <- edx %>% filter(str_detect(genres,"Romance"))
cat('Drama has', nrow(drama), 'movies\n')
cat('Comedy has', nrow(comedy), 'movies\n')
cat('Thriller has', nrow(thriller), 'movies\n')
cat(' Romance has', nrow( romance), 'movies\n')

rm(drama, comedy, thriller, romance)

# Most Rated Movies

# Find the top 10 most rated movies:

top_movies <- edx %>%
  group_by(title) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  head(10)

print(top_movies)

# bar chart of top_title

ggplot(top_movies, aes(x = reorder(title, count), y = count)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip(y=c(0, 40000)) +labs(x="", y="Number of ratings") +
  geom_text(aes(label= count), hjust=-0.1, size=3) +
  labs(title = "Top 10 Most Rated Movies", x = "Movie", y = "Number of Ratings") +
  theme_minimal()

#top ten title and genres

edx %>%
  group_by(title, genres) %>%
  summarize(count = n(), .groups = "drop") %>%  
  slice_max(order_by = count, n = 10) %>%  
  arrange(desc(count))

kable(head(edx %>%
             group_by(title, genres) %>%
             summarize(count = n(), .groups = "drop") %>%  # Explicitly drop grouping
             top_n(10, count) %>%
             arrange(desc(count)), 5)) %>%
  kable_styling(bootstrap_options = "bordered", full_width = F, position = "center") %>%
  column_spec(1, bold = T) %>%
  column_spec(2, bold = T) %>%
  column_spec(3, bold = T)



#The movies which have the highest number of ratings are in the top genres categories : movies like Pulp fiction (1994), ForrestGump(1994) or Jurrasic Park(1993) which are in the top 5 of movie's ratings number , are part of theDrama, Comedy or Action genres.

#3. Number of Ratings per User
# Check how many ratings each user has given:

# Ratings per user
user_ratings <- edx %>%
  group_by(userId) %>%
  summarise(count = n())

# Histogram of user ratings count

ggplot(user_ratings, aes(x = count)) +
  geom_histogram(bins = 50, fill = "purple", color = "black") +
  scale_x_log10() +  # Log scale for better visualization
  labs(title = "Distribution of Ratings per User", x = "Number of Ratings", y = "Count of Users") +
  theme_minimal()

# Ratings per movieId 
movieId_ratings <- edx %>%
  group_by( movieId ) %>%
  summarise(count = n())

# Histogram of movieId ratings count

ggplot(movieId_ratings, aes(x = count)) +
  geom_histogram(bins = 50, fill = "blue", color = "black") +
  scale_x_log10() +  # Log scale for better visualization
  labs(title = "Distribution of Ratings per movieId", x = "Number of Ratings", y = "Count of movieId") +
  theme_minimal()

#The distribution of the number of ratings by movieId and by userId shows the following relationships :
#some movies get rated more than others, and some users are more active than others at rating movies. 
#This explains the presence of movies and users effect. 

#4. Ratings Over Time
# Ratings change over time:

# Convert timestamp to date format
edx <- edx %>%
  mutate(date = as.Date(as.POSIXct(timestamp, origin = "1970-01-01")))

# Plot ratings over time

ggplot(edx, aes(x = date)) +
  geom_histogram(binwidth = 30, fill = "skyblue", color = "black") +
  labs(title = "Ratings Over Time", x = "Date", y = "Number of Ratings") +
  theme_minimal()

# This analysis helps us understand if time-based effects influence movie ratings. If we notice strong time trends, we might consider adding a time-based feature to our model later.
# "The Ratings over time", shows that time has a weak effect on the ratings. 
#########################
# 1. Yearly Trends in Ratings
# Extract year from the date column
edx <- edx %>%
  mutate(year = format(date, "%Y"))

# Aggregate ratings per year
yearly_ratings <- edx %>%
  group_by(year) %>%
  summarise(count = n()) %>%
  arrange(year)

ggplot(yearly_ratings, aes(x = as.numeric(year), y = count)) +
  geom_line(color = "blue", linewidth = 1) +  
  geom_point(color = "red") +
  labs(title = "Number of Ratings Per Year", x = "Year", y = "Number of Ratings") +
  theme_minimal()

# Despite the number of ratings fluctuating, the rating distribution has not changed significantly over time.


######## Step 3: Modelling and Regularization ########

#The goal is to predict the rating column for unseen data in the final_holdout_test set.
#Following the EDA we conducted, we will focus on the the effects  in relation with the "users" and "movies".


# Define RMSE function
#Root Mean Squared Error (RMSE) is our evaluation metric.

# Define RMSE function
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Split edx into training and validation set
set.seed(1, sample.kind="Rounding")  # Ensures reproducibility
test_index <- createDataPartition(edx$rating, p = 0.1, list = FALSE)
train_set <- edx[-test_index, ]
validation_set <- edx[test_index, ]

# Compute the mean rating
mu <- mean(train_set$rating)

# Baseline Model: Predicting all ratings as mean
baseline_predictions <- rep(mu, nrow(validation_set))
baseline_rmse <- RMSE(validation_set$rating, baseline_predictions)
cat("Baseline RMSE:", baseline_rmse, "\n")

#The RMSE for this model is 1.060056, which is unsurprisingly high.
#This baseline model is too simple. Next, we will account for movie effects (some movies are rated higher or lower than average.

# Movie Effect Model
movie_effects <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

movie_predictions <- validation_set %>%
  left_join(movie_effects, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

movie_predictions[is.na(movie_predictions)] <- mu
movie_effect_rmse <- RMSE(validation_set$rating, movie_predictions)
cat("Movie_effect RMSE:", movie_effect_rmse, "\n")

#The RMSE for this model is 0.9429666, which is quite an improvement from our Baseline model but still far from our initial goal.

## Movie + User Effect Model
#we can further improve our model by adding user effects.
# Compute user effect (b_u)
user_effects <- train_set %>%
  left_join(movie_effects, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

user_predictions <- validation_set %>%
  left_join(movie_effects, by = "movieId") %>%
  left_join(user_effects, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

user_predictions[is.na(user_predictions)] <- mu
user_effect_rmse <- RMSE(validation_set$rating, user_predictions)

cat("User_effect_RMSE:", user_effect_rmse, "\n")
#The RMSE for this model is 0.8646915. Quite an improvement from our latest model but we are not there yet!


#If RMSE is still high, we might have overfitting due to movies/users with very few ratings.
#The next step is to regularize our model to improve generalization.
# Instead of just computing the average deviations (b_i and b_u), we use penalized estimates by shrinking extreme values towards zero.

# Regularization - Choose the Best Lambda
## remembering that lambda (??) is a tuning parameter. 
# Tune lambda using cross-validation
lambda_values <- seq(0, 10, 0.25)
rmse_results <- sapply(lambda_values, function(lambda) {
  movie_effects <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (lambda + n()))
  
  user_effects <- train_set %>%
    left_join(movie_effects, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i) / (lambda + n()))
  
  predictions <- validation_set %>%
    left_join(movie_effects, by = "movieId") %>%
    left_join(user_effects, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  predictions[is.na(predictions)] <- mu
  return(RMSE(validation_set$rating, predictions))
})

best_lambda <- lambda_values[which.min(rmse_results)]
cat("Best Lambda:", best_lambda, "\n")

# Plot RMSE vs Lambda
# Create a data frame for plotting
lambda_df <- data.frame(lambda_values = lambda_values, rmse_results = rmse_results)

# Plot RMSE vs Lambda

ggplot(lambda_df, aes(x = lambda_values, y = rmse_results)) +
  geom_point(color = "blue") +  # Scatter plot points
  geom_line(color = "red", linewidth = 1) +  # Connect points with a red line
  geom_vline(xintercept = best_lambda, color = "green", linetype = "dashed", linewidth = 1) +  # Best lambda line
  annotate(
    "text",
    x = best_lambda,
    y = min(rmse_results),
    label = paste("Best ?? =", round(best_lambda, 2)),
    color = "black",
    hjust = -0.1,
    vjust = -0.5
  ) +  # Label for best lambda
  labs(
    title = "Lambda Selection for Regularization",
    x = "Lambda",
    y = "RMSE"
  ) +
  theme_minimal()


# Apply best lambda on full edx data
movie_effects <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (best_lambda + n()))
    
user_effects <- edx %>%
  left_join(movie_effects, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (best_lambda + n()))

predictions <- final_holdout_test %>%
  left_join(movie_effects, by = "movieId") %>%
  left_join(user_effects, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

predictions[is.na(predictions)] <- mu
Regularised_Movie_User_Biases_rmse <- RMSE(final_holdout_test$rating, predictions)
cat("Regularised_Movie_User Biases_rmse:", Regularised_Movie_User_Biases_rmse, "\n")

#RMSE comparison table


rating_RMSE3 <- data.frame (Method = c("RMSE Target", "Regularised Movie & User Biases"), RMSE_test_set= c(0.8649,Regularised_Movie_User_Biases_rmse ))

kable(rating_RMSE3) %>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  column_spec(1,bold = T ) %>%
  row_spec(1,bold =T ,color = "blue") %>%
  row_spec(2,bold =T ,color = "white" , background ="#D7261E") 

#Results

#The "Regularised Movie and User Biases" Model being the one with the lowest RMSE,
#With a RMSE at 0.8648177, the objective of the exercise has been achieved !
#The final model achieved an RMSE of  0.8648177 on the final_holdout_test dataset, demonstrating its predictive effectiveness.

#Conclusion

#In this project, we successfully built a movie recommendation system using the MovieLens dataset , achieving a final Root Mean Squared Error (RMSE) of 0.8648 on the holdout test set. Our approach involved several stages:

#Exploratory Data Analysis (EDA) : Understanding the dataset's structure and identifying patterns such as rating distributions and sparsity.

#Baseline Models : Establishing performance benchmarks with simple models like global mean prediction and movie-specific biases.

#Advanced Modeling : Leveraging regularization techniques  to capture latent user-movie interactions and improve predictive accuracy.

#Final Evaluation : Training the best-performing model on the full edx dataset and evaluating its performance on the unseen holdout test set.

#The results demonstrate that our recommendation system is both effective and competitive, achieving an RMSE that aligns with industry standards for similar datasets. By incorporating regularization and advanced modeling techniques, we significantly improved upon baseline predictions, showcasing the importance of systematic refinement in recommendation systems.

#The limitations were primarily associated with the size of the dataset and memory constraints. Additionally, advanced models like matrix factorization require significant computational resources, making real-time implementation challenging.

#Future work includes developing more complex models, such as deep learning approaches. For instance, neural network-based models like deep matrix factorization or autoencoders could be explored to capture more intricate patterns in the data.
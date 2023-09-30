import matplotlib
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Read in the data
airbnb_listings = pd.read_csv('/Users/josephjo/Documents/Vanguard/airbnb_data/Airbnb_Listings.csv')

# Viewing mean, std, min, count, ... of airbnb_listings
airbnb_listings.describe()

# Null values analysis
airbnb_listings.isnull().sum()

# Null columns
null_columns = []
for f in airbnb_listings.columns:
    if airbnb_listings[f].isnull().sum() > 0:
        null_columns.append(f)
sns.heatmap(airbnb_listings.isnull(),yticklabels=False, cbar = False, cmap = 'Blues')

# View the data types
df = airbnb_listings.dtypes

# Distribution of cities in Bay Area
listing_cities = airbnb_listings.neighbourhood_cleansed.value_counts().index
listing_cities_cnt = airbnb_listings.neighbourhood_cleansed.value_counts().values
plt.pie(listing_cities_cnt, labels = listing_cities,autopct='%1.1f%%')
#plt.show()
# largest count of city: San Jose , Palo Alto, Sunnyvale

# Removing $ from prices to make the column into a float
airbnb_listings['price'] = airbnb_listings['price'].str.replace('[\$,]', '', regex=True).astype(float)

# Understanding price
price_summary = airbnb_listings['price'].describe()
# Create a boxplot for the 'price' column
plt.figure(figsize=(20, 16))  # Adjust the figure size as needed
sns.boxplot(y=airbnb_listings['price'])
# Set labels and title
plt.ylabel('Price', fontsize = 25)
plt.title('Boxplot of Airbnb Listing Prices', fontsize = 25)
plt.xticks(fontsize=25)
plt.ylim(0,2000)
plt.yticks(fontsize=25)

"""Exploratory Data Analysis"""
# Neighbourhood vs Price
plt.figure(figsize=(12, 8))
sns.barplot(x='neighbourhood_cleansed', y='price', data=airbnb_listings, estimator=np.mean)
plt.xlabel('Neighborhood')
plt.ylabel('Mean Price')
plt.title('Mean Price by Neighborhood')
plt.xticks(rotation=90)
#plt.show()
# Los Altos Hills & Saratoga have the daily highest avg prices

# Rating Scores vs Price
review_score = airbnb_listings['review_scores_rating'].describe()
#print(review_score)
plt.figure(figsize=(12, 6))
sns.barplot(x='neighbourhood_cleansed', y='review_scores_rating', data=airbnb_listings, estimator=np.mean)
plt.xlabel('Neighborhood')
plt.ylabel('Review Score')
plt.title('Mean Price by Review Scores')
plt.xticks(rotation=90)
#plt.show()
# Neighborhood vs Review score

# print(airbnb_listings['review_scores_rating'].isnull().sum())
# 1367 null reviews out of 7221 from Review Scores

# Super-host stats
superhost_stat = airbnb_listings['host_is_superhost'].describe()
# print(superhost_stat)
# 55% are not superhost 45% are superhost

"""Viewing Relationship between predictors and target variable"""
plt.scatter(airbnb_listings['review_scores_rating'], airbnb_listings['price'])
plt.xlabel('Review Scores')
plt.ylabel('Price')
plt.title('Review Score vs Price')
#plt.show()
#Findings:
#1. Review scores does not particularly show a linear relationship
#2. Reviews per month does not show any linear relationship with price
#3. # of Bedrooms seem to show a -x^2 relationship

# Duplicate Analysis
num_duplicates = airbnb_listings.duplicated().sum()
# No dups

# Description into Number of Words in Description
def count_words(text):
    if isinstance(text, str):  # Check if the input is a string
        words = text.split()
        return len(words)
    else:
        return 0

# List of Amenities {TV, Air Condition, ...} -> 12
def count_amenities(row):
    amenities_list = row.split(',')
    return len(amenities_list)

# Summary description of airbnb -> Count of numbers of summary (Maybe more description = higher price?)
airbnb_listings['summary_word_count'] = airbnb_listings['summary'].apply(count_words)

# Number of amenities
airbnb_listings['amenity_count'] = airbnb_listings['amenities'].apply(count_amenities)

# Initial Feature Selection
airbnb_listings.drop(['id','square_feet','weekly_price','monthly_price','amenities','listing_url','scrape_id','summary','last_scraped','name','space','description','experiences_offered','neighborhood_overview','notes','transit','access','interaction','house_rules','thumbnail_url','medium_url','picture_url','xl_picture_url','host_id','host_url','host_name','host_location','host_about','host_thumbnail_url','host_picture_url','host_neighbourhood','host_listings_count','host_total_listings_count','host_verifications','street','neighbourhood','neighbourhood_group_cleansed','city','state','zipcode','market','smart_location','country_code','country','is_location_exact','security_deposit','calendar_updated','maximum_nights_avg_ntm','minimum_nights_avg_ntm','maximum_maximum_nights','minimum_maximum_nights','maximum_minimum_nights','minimum_minimum_nights','has_availability','availability_30','availability_60','availability_90','availability_365','calendar_last_scraped','number_of_reviews_ltm','requires_license','license','jurisdiction_names','is_business_travel_ready','instant_bookable','require_guest_profile_picture','require_guest_phone_verification','calculated_host_listings_count','calculated_host_listings_count_entire_homes','calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms'],axis = 1, inplace = True)

"""Missing Values Data Prep"""
airbnb_listings['host_since'] = pd.to_datetime(airbnb_listings['host_since'])
numeric_timestamps = airbnb_listings['host_since'].view('int64')
non_null_timestamps = numeric_timestamps.dropna()  # Exclude null timestamps
mean_timestamp = non_null_timestamps.mean()
mean_date = pd.to_datetime(mean_timestamp)
airbnb_listings['host_since'].fillna(mean_date, inplace=True)
airbnb_listings['host_since'] = airbnb_listings['host_since'].dt.strftime('%Y-%m-%d')

#print(airbnb_listings['host_response_time'].describe())
airbnb_listings['host_response_time'].fillna('within an hour', inplace = True)

# Host Response Rate fillnas with mean
airbnb_listings['host_acceptance_rate'] = airbnb_listings['host_acceptance_rate'].str.rstrip('%').astype(float)
airbnb_listings['host_response_rate'] = airbnb_listings['host_response_rate'].str.rstrip('%').astype(float)
mean_response_rate = airbnb_listings['host_response_rate'].mean()
airbnb_listings['host_response_rate'].fillna(mean_response_rate, inplace = True)

# Removing $ signs
airbnb_listings['cleaning_fee'] = airbnb_listings['cleaning_fee'].str.replace('[^\d.]', '', regex=True).astype(float)
airbnb_listings['extra_people'] = airbnb_listings['extra_people'].str.replace('[^\d.]', '', regex=True).astype(float)

# Converting date to days since
date_columns = ['host_since', 'first_review', 'last_review']
def convert_to_days_since_min_date(column):
    min_date = column.min()
    return (column - min_date).dt.days

# Apply the function to each date column and create new columns with days since min date
for column in date_columns:
    airbnb_listings[column] = pd.to_datetime(airbnb_listings[column])  # Convert to datetime
    new_column_name = f'{column}_days'
    min_date = airbnb_listings[column].min()
    airbnb_listings[new_column_name] = (airbnb_listings[column] - min_date).dt.days
    airbnb_listings.drop(columns=[column], inplace=True)

airbnb_listings = airbnb_listings.drop(columns='neighbourhood_cleansed')

# Review scores fillnas with mean
columns_to_fill = [
    'host_acceptance_rate',
    'bathrooms',
    'bedrooms',
    'beds',
    'cleaning_fee',
    'reviews_per_month',
    'first_review_days',
    'last_review_days'
]
for column in columns_to_fill:
    column_median = airbnb_listings[column].median()
    airbnb_listings[column].fillna(column_median, inplace=True)

# Dropping Null values
columns_to_fill_with_zero = ['review_scores_rating',
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_checkin',
    'review_scores_communication',
    'review_scores_location',
    'review_scores_value']
#airbnb_listings[columns_to_fill_with_zero] = airbnb_listings[columns_to_fill_with_zero].fillna(0)
# Decided to drop rows where review scores were blank
airbnb_listings = airbnb_listings.dropna(subset=columns_to_fill_with_zero, how='all')

# One-hot encoding
categorical_cols = ['review_scores_rating','host_response_time', 'host_is_superhost','host_has_profile_pic','host_identity_verified','property_type','room_type','bed_type','cancellation_policy']  # List of categorical column names
airbnb_listings_encoded = pd.get_dummies(airbnb_listings, columns=categorical_cols, drop_first=True)

# Feature selection
columns_to_drop = ['cleaning_fee','property_type_Townhouse','property_type_Bungalow','property_type_Boutique hotel','property_type_Villa','property_type_Other','host_has_profile_pic_t','first_review_days','host_acceptance_rate','last_review_days','host_response_rate','property_type_Train','property_type_Loft','property_type_Camper/RV','property_type_Treehouse', 'property_type_Campsite','property_type_Lighthouse','cancellation_policy_super_strict_60','room_type_Hotel room','property_type_Barn','property_type_Cabin','property_type_Farm stay','bed_type_Pull-out Sofa','property_type_Tent','property_type_Tiny house','property_type_Earth house','bed_type_Couch','property_type_Bed and breakfast','review_scores_accuracy','review_scores_communication','review_scores_location','review_scores_checkin','review_scores_value','review_scores_cleanliness','property_type_Cottage','property_type_Yurt','bed_type_Futon','bed_type_Real Bed']
airbnb_listings_encoded = airbnb_listings_encoded.drop(columns=columns_to_drop)

# Outlier control
z_scores = stats.zscore(airbnb_listings_encoded['price'])
threshold = 3
airbnb_listings_final = airbnb_listings_encoded[np.abs(z_scores) <= threshold]
airbnb_listings_final.reset_index(drop=True, inplace=True)


# Convert the DataFrame to a CSV file
csv_filename = 'airbnb.csv'
airbnb_listings.to_csv(csv_filename, index=False)

csv_filename = 'airbnb_encoded.csv'
airbnb_listings_final.to_csv(csv_filename, index=False)


"""-------------------Random Forest Build---------------------"""

# Split the data into features (X) and target variable (y)
X = airbnb_listings_final.drop(columns=['price'])  # Use the encoded DataFrame
y = airbnb_listings_final['price']


# Number of folds for cross-validation
num_folds = 5

# Create a KFold cross-validation method
cv = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize an empty list to store metrics for each fold
rmse_scores = []
r2_scores = []
mae_scores = []
fold_actual_values = []
fold_predicted_values = []

# Perform cross-validation and calculate MAE, RMSE, and R^2 for each fold
for i, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Create a Random Forest Regressor model
    rf_model = RandomForestRegressor(n_estimators=500, random_state=42)

    # Fit the model on the training data
    rf_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = rf_model.predict(X_test)

    fold_actual = pd.Series(y_test.reset_index(drop=True), name=f'Actual Fold {i}')
    fold_predicted = pd.Series(y_pred, name=f'Predicted Fold {i}')

    fold_actual_values.append(fold_actual)
    fold_predicted_values.append(fold_predicted)

    # Calculate MAE for the fold
    mae_fold = mean_absolute_error(y_test, y_pred)
    mae_scores.append(mae_fold)

    # Calculate RMSE for the fold
    rmse_fold = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse_fold)

    # Calculate R^2 for the fold
    r2_fold = r2_score(y_test, y_pred)
    r2_scores.append(r2_fold)

    # Print MAE, RMSE, and R^2 for the fold
    print(f"Fold {i} MAE: {mae_fold:.2f}, RMSE: {rmse_fold:.2f}, R^2: {r2_fold:.2f}")

# Calculate and print the mean MAE across all folds
mean_mae = np.mean(mae_scores)
print(f"Mean MAE: {mean_mae:.2f}")

# Calculate and print the mean RMSE across all folds
mean_rmse = np.mean(rmse_scores)
print(f"Mean RMSE: {mean_rmse:.2f}")

# Calculate and print the mean R^2 score across all folds
mean_r2_score = np.mean(r2_scores)
print(f"Mean R^2 Score: {mean_r2_score:.2f}")

# Fit the model on the entire dataset
rf_model.fit(X, y)

N = 20 #top 20 features

# Get feature importances
feature_importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Show top features
top_features = feature_importance_df.sort_values(by='Importance', ascending=True)
print("Top Features:")
print(top_features.head(N))



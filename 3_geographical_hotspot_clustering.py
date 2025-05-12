import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

import pandas as pd
import plotly.graph_objects as go

# Load your data
df = pd.read_csv("2023_dataset.csv")
df['end_lat'].isna().sum()

print("Top 3 start_lat values:")
print(df['start_lat'].value_counts().head(3))

print("\nTop 3 start_lng values:")
print(df['start_lng'].value_counts().head(3))

print("\nTop 3 end_lat values:")
print(df['end_lat'].value_counts().head(3))

print("\nTop 3 end_lng values:")
print(df['end_lng'].value_counts().head(3))

df[df['start_lat'] == 41.892278][['start_station_name']].drop_duplicates()

print("Top 3 start_lat values:")
print(df['start_station_name'].value_counts().head(3))

print("Top 3 start_lat values:")
print(df['end_station_name'].value_counts().head(3))

df[df['start_lng'] == -87.612043][['end_station_name']].drop_duplicates()

df[df['end_lng'] == -87.650000][['end_station_name']].drop_duplicates()

# Manually convert the time column
df['started_at'] = pd.to_datetime(df['started_at'])
df['ended_at'] = pd.to_datetime(df['ended_at'])

df['duration_min'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60

avg_duration = df['duration_min'].mean()
print(f"Average ride duration: {avg_duration:.2f} minutes")

# Calculate the average ride distance
avg_distance = df['distance_miles'].mean()
print(f"Average ride distance: {avg_distance:.2f} miles")

# Count the number of members and casual users
user_counts = df['member_casual'].value_counts()

# Calculate the percentages
user_percentages = df['member_casual'].value_counts(normalize=True) * 100
print("User counts:")
print(user_counts)
print("\nUser percentages:")
print(user_percentages)

# Plot pie chart
labels = user_percentages.index
sizes = user_percentages.values

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#ff9999'])
plt.title('User Composition: Member vs Casual')
plt.axis('equal')
plt.savefig('/Users/pengxinyi/Library/CloudStorage/OneDrive-TheUniversityofManchester/p20/member.png', dpi=300)
plt.show()

# Extract hour from the 'started_at' column
df['hour'] = df['started_at'].dt.hour

# Calculate the number of rides per hour
hourly_ride_count = df.groupby('hour').size()

# Plot hourly ride trend
plt.figure(figsize=(10, 6))
plt.plot(hourly_ride_count.index, hourly_ride_count.values, marker='o', color='b')
plt.title("Hourly Ride Trend")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Rides")
plt.xticks(range(24))  # Ensure all hours (0-23) are labeled
plt.grid(True)
plt.savefig('/Users/pengxinyi/Library/CloudStorage/OneDrive-TheUniversityofManchester/p20/daytime.png', dpi=300)
plt.show()

# Extract the weekday (0 = Monday, 6 = Sunday)
df['weekday'] = df['started_at'].dt.weekday

# Create a new column to classify as weekday or weekend
df['day_type'] = df['weekday'].apply(lambda x: 'Weekday' if x < 5 else 'Weekend')

# Calculate the total number of rides for weekdays and weekends
weekday_ride_count = df[df['day_type'] == 'Weekday'].shape[0]
weekend_ride_count = df[df['day_type'] == 'Weekend'].shape[0]

print(f"Total rides on weekdays: {weekday_ride_count:,}")
print(f"Total rides on weekends: {weekend_ride_count:,}")

import matplotlib.pyplot as plt

# Group by day_type (weekday vs weekend) and count the number of rides
day_type_ride_count = df['day_type'].value_counts()

# Plot the number of rides for weekdays and weekends
plt.figure(figsize=(8, 5))
day_type_ride_count.plot(kind='bar', color=['#66b3ff', '#ff9999'])
plt.title("Ride Count: Weekdays vs Weekends")
plt.xlabel("Day Type")
plt.ylabel("Number of Rides")
plt.xticks(rotation=0)  # Keep the x-axis labels horizontal
plt.savefig('/Users/pengxinyi/Library/CloudStorage/OneDrive-TheUniversityofManchester/p20/ridecount.png', dpi=300)
plt.show()

# Group by the weekday and calculate the total number of rides per day
day_of_week_ride_count = df.groupby('weekday').size()

# Plot the number of rides per day of the week
plt.figure(figsize=(10, 6))
day_of_week_ride_count.plot(kind='bar', color='lightblue')
plt.title("Number of Rides Per Day of the Week")
plt.xlabel("Day of the Week (0=Monday, 6=Sunday)")
plt.ylabel("Number of Rides")
plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
plt.savefig('/Users/pengxinyi/Library/CloudStorage/OneDrive-TheUniversityofManchester/p20/ridecountperday.png', dpi=300)
plt.show()

# Calculate the proportion of rides on weekdays vs weekends
total_rides = df.shape[0]
weekday_proportion = weekday_ride_count / total_rides * 100
weekend_proportion = weekend_ride_count / total_rides * 100

print(f"Proportion of rides on weekdays: {weekday_proportion:.2f}%")
print(f"Proportion of rides on weekends: {weekend_proportion:.2f}%")

# Extract year and month from the 'started_at' column
df['year_month'] = df['started_at'].dt.to_period('M')
df['month'] = df['started_at'].dt.month
df['year'] = df['started_at'].dt.year

# Calculate the total number of rides per month (grouped by year and month)
monthly_ride_count = df.groupby('year_month').size()

# Alternatively, you can group by just the 'month' to see trends per month across all years
monthly_ride_count_by_month = df.groupby('month').size()

import matplotlib.pyplot as plt

# Plot the trend of rides per month
plt.figure(figsize=(12, 6))
monthly_ride_count.plot(kind='line', marker='o', color='b')
plt.title("Monthly Ride Trend (with Potential Seasonality)")
plt.xlabel("Month")
plt.ylabel("Number of Rides")
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.grid(True)
plt.savefig('/Users/pengxinyi/Library/CloudStorage/OneDrive-TheUniversityofManchester/p20/ridecountpermonth.png', dpi=300)
plt.show()

# Calculate the total number of rides per month across all years
monthly_ride_count_by_month = df.groupby('month').size()

# Plot rides per month (ignoring the year)
plt.figure(figsize=(10, 6))
monthly_ride_count_by_month.plot(kind='bar', color='lightcoral')
plt.title("Total Number of Rides Per Month (Across All Years)")
plt.xlabel("Month")
plt.ylabel("Number of Rides")
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=0)
plt.show()

# Group by 'rideable_type' and count the number of rides for each type
rideable_type_counts = df['rideable_type'].value_counts()

# Display the result
print(rideable_type_counts)

plt.figure(figsize=(8, 5))
rideable_type_counts.plot(kind='bar', color='skyblue')
plt.title("Frequency of Different Rideable Types")
plt.xlabel("Rideable Type")
plt.ylabel("Number of Rides")
plt.xticks(rotation=15)  # Rotate x-axis labels by 15 degrees
plt.savefig('/Users/pengxinyi/Library/CloudStorage/OneDrive-TheUniversityofManchester/p20/tools.png', dpi=300)
plt.show()

# Group by 'user_type' and 'rideable_type' and count the number of rides
user_rideable_preference = df.groupby(['member_casual', 'rideable_type']).size().unstack()

# Display the result
print(user_rideable_preference)

# Group by 'rideable_type' and calculate the average 'duration_min' and 'distance_miles'
average_duration_distance = df.groupby('rideable_type')[['duration_min', 'distance_miles']].mean()

# Display the result
print(average_duration_distance)
# Plot the average duration for each rideable type
plt.figure(figsize=(10, 6))
average_duration_distance['duration_min'].plot(kind='bar', color='lightblue')
plt.title("Average Duration for Each Rideable Type")
plt.xlabel("Rideable Type")
plt.ylabel("Average Duration (Minutes)")
plt.xticks(rotation=45)
plt.show()
# Plot the average distance for each rideable type
plt.figure(figsize=(10, 6))
average_duration_distance['distance_miles'].plot(kind='bar', color='salmon')
plt.title("Average Distance for Each Rideable Type")
plt.xlabel("Rideable Type")
plt.ylabel("Average Distance (Miles)")
plt.xticks(rotation=45)
plt.show()

# Create a figure and axis
fig, ax1 = plt.subplots(figsize=(12, 7))

# Set the width for the bars
bar_width = 0.35

# Set the positions for the bars
indices = np.arange(len(average_duration_distance))

# Plot the average duration on the left y-axis
bars1 = ax1.bar(indices - bar_width/2, average_duration_distance['duration_min'], bar_width, color='lightblue', label='Duration (Minutes)')

# Create a second y-axis to plot the average distance
ax2 = ax1.twinx()
bars2 = ax2.bar(indices + bar_width/2, average_duration_distance['distance_miles'], bar_width, color='salmon', label='Distance (Miles)')

# Set labels and title
ax1.set_xlabel("Rideable Type")
ax1.set_ylabel("Average Duration (Minutes)", color='lightblue')
ax2.set_ylabel("Average Distance (Miles)", color='salmon')
ax1.set_title("Average Duration and Distance for Each Rideable Type")

# Set the x-ticks to be in the middle of the bars
ax1.set_xticks(indices)
ax1.set_xticklabels(average_duration_distance.index, rotation=45)

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Display the plot
plt.show()
#Spatial Analysis）
# Count the occurrences of each start and end station
start_station_counts = df['start_station_name'].value_counts()
end_station_counts = df['end_station_name'].value_counts()

# Get the top 10 most popular stations for both start and end
top_start_stations = start_station_counts.head(10)
top_end_stations = end_station_counts.head(10)

# Display the top stations
print("Top 10 Most Popular Start Stations:")
print(top_start_stations)

print("\nTop 10 Most Popular End Stations:")
print(top_end_stations)

# Plot the top 10 most popular start stations
plt.figure(figsize=(10, 6))
top_start_stations.plot(kind='bar', color='lightblue')
plt.title("Top 10 Most Popular Start Stations")
plt.xlabel("Start Station")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

# Plot the top 10 most popular end stations
plt.figure(figsize=(10, 6))
top_end_stations.plot(kind='bar', color='salmon')
plt.title("Top 10 Most Popular End Stations")
plt.xlabel("End Station")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

# Group the data by 'start_station_name' and 'end_station_name' and count occurrences
route_counts = df.groupby(['start_station_name', 'end_station_name']).size().reset_index(name='ride_count')

# Sort the routes by ride count in descending order to get the high-frequency routes
high_frequency_routes = route_counts.sort_values(by='ride_count', ascending=False).head(10)  # Top 10 most frequent routes

# Display the top high-frequency routes
print(high_frequency_routes)

# Create a bar chart to visualize the top 10 high-frequency routes
plt.figure(figsize=(12, 8))
plt.barh(high_frequency_routes['start_station_name'] + ' to ' + high_frequency_routes['end_station_name'],
         high_frequency_routes['ride_count'], color='lightblue')

# Add labels and title
plt.xlabel('Number of Rides')
plt.ylabel('Start-End Station Pair')
plt.title('Top 10 High-Frequency Cycling Routes')
plt.gca().invert_yaxis()  # To display the highest frequency at the top
plt.show()

# Count the number of rides at each start station
start_station_counts = df['start_station_name'].value_counts().reset_index(name='ride_count')
start_station_counts.columns = ['station_name', 'ride_count']

# Count the number of rides at each end station
end_station_counts = df['end_station_name'].value_counts().reset_index(name='ride_count')
end_station_counts.columns = ['station_name', 'ride_count']

# Combine start and end station counts
station_counts = pd.concat([start_station_counts, end_station_counts], ignore_index=True)
station_counts = station_counts.groupby('station_name').agg({'ride_count': 'sum'}).reset_index()

# Sort by the highest number of rides
station_counts = station_counts.sort_values(by='ride_count', ascending=False)
# Display the top 10 most popular stations
print(station_counts.head(10))
# Check for NaN values specifically in 'end_lat' and 'end_lng'
print(df_cleaned[['end_lat', 'end_lng']].isna().sum())
# Drop rows where either 'end_lat' or 'end_lng' is NaN
df_cleaned = df.dropna(subset=['end_lat', 'end_lng'])
# Create a new column combining start and end stations to identify routes
df_cleaned['route'] = df_cleaned['start_station_name'] + " to " + df_cleaned['end_station_name']

# Find the most frequent routes
popular_routes = df_cleaned['route'].value_counts().head(10)
print("Most popular routes:")
print(popular_routes)

# Count the number of rides for each station (both start and end)
station_usage = pd.concat([
    df_cleaned['start_station_name'].value_counts(),
    df_cleaned['end_station_name'].value_counts()
], axis=1).fillna(0)

# Rename columns for clarity
station_usage.columns = ['start_usage', 'end_usage']

# Calculate total usage for each station (both start and end)
station_usage['total_usage'] = station_usage['start_usage'] + station_usage['end_usage']

# Find stations with low usage (underutilized)
underutilized_stations = station_usage[station_usage['total_usage'] < 5]  # Customize the threshold as needed
print("Underutilized stations:")
print(underutilized_stations)
#Analysis of high-frequency areas of cycling paths
import folium
from folium.plugins import HeatMap
df_cleaned['route'] = df_cleaned['start_station_name'] + " to " + df_cleaned['end_station_name']

route_counts = df_cleaned.groupby('route').size().reset_index(name='count')
top_routes = route_counts.nlargest(10, 'count')
top_start_coords = []
top_end_coords = []
for _, row in top_routes.iterrows():
    route_data = df_cleaned[df_cleaned['route'] == row['route']]
    top_start_coords.extend(route_data[['start_lat', 'start_lng']].values)
    top_end_coords.extend(route_data[['end_lat', 'end_lng']].values)

map_center = [df_cleaned['start_lat'].mean(), df_cleaned['start_lng'].mean()]
bike_map = folium.Map(location=map_center, zoom_start=12)

heat_data = []
for start, end in zip(top_start_coords, top_end_coords):
    heat_data.append([start[0], start[1]])
    heat_data.append([end[0], end[1]])

HeatMap(heat_data).add_to(bike_map)
bike_map

from sklearn.cluster import KMeans
df_cleaned = df.dropna(subset=['start_lat', 'start_lng', 'end_lat', 'end_lng'])
df_cleaned.head()
start_coords = df_cleaned[['start_lat', 'start_lng']]
start_coords.columns = ['lat', 'lng']  # 统一列名
end_coords = df_cleaned[['end_lat', 'end_lng']]
end_coords.columns = ['lat', 'lng']  # 统一列名
coords = pd.concat([start_coords, end_coords], ignore_index=True)

print(coords.isna().sum())

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
coords['cluster'] = kmeans.fit_predict(coords[['lat', 'lng']])

plt.figure(figsize=(12,8))

sns.scatterplot(data=coords, x='lng', y='lat', hue='cluster', palette='tab10', s=20)

plt.title('Bike Start and End Points Clustering (K-Means)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Cluster')
plt.show()

centers = kmeans.cluster_centers_

center_df = pd.DataFrame(centers, columns=['lat', 'lng'])
print(center_df)
plt.figure(figsize=(12, 8))

sns.scatterplot(data=coords, x='lng', y='lat', hue='cluster', palette='tab10', s=20, alpha=0.6)

plt.scatter(centers[:, 1], centers[:, 0], 
            c='black', marker='X', s=200, label='Cluster Centers')

plt.title('Bike Start and End Points Clustering with Cluster Centers')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Cluster')
plt.grid(True)

plt.savefig('/Users/pengxinyi/Library/CloudStorage/OneDrive-TheUniversityofManchester/p20/cluster.png', dpi=300)
plt.show()

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

k_range = range(1, 16)

sse = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(coords[['lat', 'lng']])
    sse.append(kmeans.inertia_)  # inertia_ SSE

plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.xticks(k_range)
plt.grid(True)

plt.savefig('/Users/pengxinyi/Library/CloudStorage/OneDrive-TheUniversityofManchester/p20/elbow_method.png', dpi=300)
plt.show()
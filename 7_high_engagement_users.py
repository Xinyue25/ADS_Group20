import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('2023_dataset.csv')

# 修复错误的年份格式，如"0023" -> "2023"
def fix_date(date_str):
    if isinstance(date_str, str) and date_str.startswith('0023'):
        return date_str.replace('0023', '2023', 1)
    return date_str

df['started_at'] = pd.to_datetime(df['started_at'].apply(fix_date), errors='coerce')
df['ended_at'] = pd.to_datetime(df['ended_at'].apply(fix_date), errors='coerce')

# 删除无效时间记录
df = df.dropna(subset=['started_at', 'ended_at'])

# 重新生成时间衍生字段
df['hour'] = df['started_at'].dt.hour
df['weekday_name'] = df['started_at'].dt.day_name()
df['month_name'] = df['started_at'].dt.month_name()

# 清理站点名字段：统一处理 Unknown
df['start_station_name'] = df['start_station_name'].replace('Unknown', np.nan)
df['end_station_name'] = df['end_station_name'].replace('Unknown', np.nan)
df['start_station_id'] = df['start_station_id'].replace('Unknown', np.nan)
df['end_station_id'] = df['end_station_id'].replace('Unknown', np.nan)

# 如果没有 user_id，这里模拟创建（假设一个用户多次用相同设备/卡号）
# 示例逻辑：我们假设每个 rideable_type + member_casual + 起始站点组合作为伪用户标识
df['user_id'] = (
    df['rideable_type'].astype(str) + "_" +
    df['member_casual'].astype(str) + "_" +
    df['start_station_id'].astype(str)
)

# 按用户统计骑行次数
user_counts = df.groupby('user_id').size().reset_index(name='trip_count')
threshold = user_counts['trip_count'].quantile(0.95)

# 前5%的用户
top_user_ids = user_counts[user_counts['trip_count'] >= threshold]['user_id']
df_top = df[df['user_id'].isin(top_user_ids)]

# 总用户数（基于 ride_id 推测的用户总数）
total_users = len(df)
high_engagement_users = len(df_top)


# 创建透视表
heatmap_data = df_top.pivot_table(
    index='weekday_name',
    columns='hour',
    values='ride_id',
    aggfunc='count'
).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# 绘图
plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', linewidths=0.2, linecolor='gray')
plt.title('Hourly Usage Heatmap – Top 5% Users')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')
plt.tight_layout()
plt.show()

from scipy.stats import entropy

entropy_list = []
for user, group in df_top.groupby('user_id'):
    hour_dist = group['hour'].value_counts(normalize=True).sort_index()
    user_entropy = entropy(hour_dist)
    entropy_list.append({'user_id': user, 'entropy': user_entropy})

entropy_df = pd.DataFrame(entropy_list)
entropy_df = entropy_df.merge(df[['user_id', 'member_casual']].drop_duplicates(), on='user_id')

plt.figure(figsize=(10, 6))
sns.boxplot(x='member_casual', y='entropy', data=entropy_df, palette='Set2')
sns.stripplot(x='member_casual', y='entropy', data=entropy_df,
              jitter=True, color='black', size=3, alpha=0.4)
plt.title('Distribution of Time Entropy by User Type')
plt.xlabel('User Type')
plt.ylabel('Time Entropy')
plt.tight_layout()
plt.show()

# 假设你已经通过用户行为识别出了 top_users（用户ID列表）
all_user_ids = df['user_id'].dropna().unique()
high_engagement_user_ids = entropy_df['user_id'].unique()

high_count = len(high_engagement_user_ids)
other_count = len(set(all_user_ids) - set(high_engagement_user_ids))

plt.figure(figsize=(6,6))
plt.pie([high_count, other_count],
        labels=['High-engagement Users', 'Other Users'],
        autopct='%1.1f%%',
        colors = ['#f4a582', '#92c5de'],
        startangle=140)
plt.title('User Proportion by Engagement Level')
plt.axis('equal')
plt.show()


# 处理时间（如果尚未处理）
df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')

# 创建用户骑行次数统计
user_ride_counts = df.groupby('user_id')['ride_id'].count().reset_index(name='ride_count')

# 按照骑行次数排序，获取 top 5%
cutoff = user_ride_counts['ride_count'].quantile(0.95)
high_users = user_ride_counts[user_ride_counts['ride_count'] >= cutoff]

# 合并标签
df['engagement_level'] = df['user_id'].apply(
    lambda x: 'High Engagement Users' if x in high_users['user_id'].values else 'Other Users'
)

# 计算总骑行次数中各类用户的贡献
ride_counts_by_group = df['engagement_level'].value_counts()

# 绘图
plt.figure(figsize=(6,6))
colors = ['#1f77b4', '#ff7f0e']
ride_counts_by_group.plot.pie(
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    labels=ride_counts_by_group.index,
    wedgeprops={'edgecolor': 'black'}
)
plt.title('Ride Volume Share by Engagement Level')
plt.ylabel('')  # 去除 y 轴标签
plt.tight_layout()
plt.show()


# 按星期统计骑行次数
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_counts = df_top.groupby(['weekday_name', 'member_casual']).size().reset_index(name='ride_count')
weekday_counts['weekday_name'] = pd.Categorical(weekday_counts['weekday_name'], categories=weekday_order, ordered=True)
weekday_counts = weekday_counts.sort_values('weekday_name')

# 绘图
plt.figure(figsize=(10,6))
sns.barplot(data=weekday_counts, x='weekday_name', y='ride_count', hue='member_casual')
plt.title('Top 5% Users: Weekly Ride Count by User Type')
plt.xlabel('Day of Week')
plt.ylabel('Number of Rides')
plt.legend(title='User Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 按月份统计骑行次数
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_counts = df_top.groupby(['month_name', 'member_casual']).size().reset_index(name='ride_count')
monthly_counts['month_name'] = pd.Categorical(monthly_counts['month_name'], categories=month_order, ordered=True)
monthly_counts = monthly_counts.sort_values('month_name')

# 绘图
plt.figure(figsize=(10,6))
sns.lineplot(data=monthly_counts, x='month_name', y='ride_count', hue='member_casual', marker='o')
plt.title('Top 5% Users: Monthly Ride Trends by User Type')
plt.xlabel('Month')
plt.ylabel('Number of Rides')
plt.legend(title='User Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
















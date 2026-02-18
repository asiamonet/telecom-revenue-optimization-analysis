# Loading all the libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats as st

# Load the data files into different DataFrames
calls = pd.read_csv('/datasets/megaline_calls.csv')
internet = pd.read_csv('/datasets/megaline_internet.csv')
messages = pd.read_csv('/datasets/megaline_messages.csv')
plans = pd.read_csv('/datasets/megaline_plans.csv')
users = pd.read_csv('/datasets/megaline_users.csv')

# Print the general/summary information about the plans' DataFrame
plans.info()

# Print a sample of data for plans
plans

#converted MB to GB for easier calculation later
plans['gb_per_month_included']=plans['mb_per_month_included']/1024

plans
#confirmed column name and data for GB was added.

#In the plans table, the column is called plan_name. 
#In the users table, the column is called plan. Renaming the column to prevent confusion later
plans = plans.rename(columns={'plan_name': 'plan'})

#confirm change
print(plans.columns)

# Print the general/summary information about the users' DataFrame
users.info()# Print the general/summary information about the users' DataFrame
users.info()

# Print a sample of data for users
users

#Fixing the d-type for reg_date
users['reg_date'] = pd.to_datetime(users['reg_date'])

#will also fix d-type of churn_date because currently Python treats this data like words/names.
users['churn_date']=pd.to_datetime(users['churn_date'])

users.info()

"""
Note on Data Enrichment: I have chosen not to add derived features here as the current columns
provide all necessary dimensions for this analysis. My approach relies on aggregating raw usage
data and merging it with user plans, which is sufficient to calculate revenue and perform the
required hypothesis testing without adding extra complexity to the dataset.
"""
# Print the general/summary information about the calls' DataFrame
calls.info()

# Print a sample of data for calls
calls

"""
I observed no missing data, large table with 137,735 entries. There are issues with call_date
dtype, must be converted to datetime64. There are issues with duration dtype, must be rounded
up and converted to an integer since Megaline rounds up to the nearest minute for every call.
"""

#Fix the data
#Convert call_date to datetime
calls['call_date'] = pd.to_datetime(calls['call_date'])

#Check for 0-duration calls before rounding
print(f"Number of 0-minute calls before: {(calls['duration'] == 0).sum()}")

#Round up duration and convert to integer
# his turns 8.52 into 9 and 0.37 into 1. np.ceil(0) remains 0.
calls['duration'] = np.ceil(calls['duration']).astype(int)

#Confirm they are still 0 after rounding
print(f"Number of 0-minute calls after: {(calls['duration'] == 0).sum()}")

calls.info()
calls.head()

"""
I have chosen not to add a dedicated month column to the calls dataframe at this stage. To keep
the workflow lean and avoid redundant columns, I perform the monthly extraction directly within
the aggregation phase using dt.month during the groupby process. This ensures that the final
revenue calculations are accurate while maintaining a more memory-efficient dataset during the 
preprocessing stage.
"""
# Print the general/summary information about the messages' DataFrame
messages.info()

# Print a sample of data for messages
messages

"""
There are no missing values. I will need to fix message_date dtype. Important to note that each row in the table represents a single message.
"""
# Convert message_date to datetime
messages['message_date']=pd.to_datetime(messages['message_date'])
messages.info()
messages.head()

# Print the general/summary information about the internet DataFrame
internet.info()

# Print a sample of data for the internet traffic
internet

"""
I observed no missing values. The session_date is an object, must be converted to datetime64. I
also noticed sessions with 0.0 MB usage; I've decided to keep them as they represent network
connection attempts.
"""

#Change session_date to datetime dtype. 
internet['session_date']=pd.to_datetime(internet['session_date'])

# Check for 0-MB sessions
zero_sessions = internet[internet['mb_used'] == 0]
print(f"Number of 0-MB sessions: {len(zero_sessions)}")
print(f"Percentage of 0-MB sessions: {len(zero_sessions) / len(internet):.2%}")

internet.info()

# Print out the plan conditions and make sure they are clear for you

plan_rules = """
GENERAL RULES:
- Rounding: Megaline always rounds UP for billing.

CALL RULES:
- Each call is rounded up to the nearest minute.
unded up to the nearest minute.
- Even a 1-second call counts as 1 minute.
- 0.0 duration calls = Missed calls (no charge).

INTERNET RULES:
- Individual sessions are NOT rounded.
- Total MB for the month is summed first, then rounded UP to the nearest GB.
- Conversion: 1 GB = 1024 MB.
  Example: 1025 MB / 1024 = 1.0009 -> Round up to 2 GB.

REVENUE FORMULA:
- (Total usage - Plan limit) * Calling plan value + Monthly Fee
"""

display(plans)

print(plan_rules)

"""
Clarification on Missed Calls & Rounding: As noted in the Data Preprocessing section for calls, I have verified that 0.0-duration calls (missed calls) remain 0 after the np.ceil() rounding step. This
aligns with Megaline's policy that only connected calls are billed. When calculating monthly revenue, these zero-duration calls add 0 to the total minutes and therefore do not trigger
overage charges or affect the final billing.
"""
# Calculate the number of calls made by each user per month. Save the result.
#create a month column 
calls['month']=calls['call_date'].dt.month
#Group by user, month and count the call IDs
calls_per_month = calls.groupby(['user_id', 'month'])['id'].count().reset_index()
#Rename columns
calls_per_month.columns = ['user_id', 'month', 'calls_count']
#see result
calls_per_month.head()

# Calculate the amount of minutes spent by each user per month. Save the result.
#Groupby user and month, and add number of minutes(duration)
minutes_per_month=calls.groupby(['user_id','month'])['duration'].sum().reset_index()
#rename columns 
minutes_per_month.columns=['user_id', 'month', 'minutes_sum']
minutes_per_month.head()

# Calculate the number of messages sent by each user per month. Save the result.
#Create a month column 
messages['month']=messages['message_date'].dt.month
#Groupby user_id and month & count id. 
messages_per_month=messages.groupby(['user_id','month'])['id'].count().reset_index()
#rename columns
messages_per_month.columns=['user_id','month','messages_count']
messages_per_month.head()

# Calculate the volume of internet traffic used by each user per month. Save the result.
#Create a month column 
internet['month']=internet['session_date'].dt.month
#Groupby user_id & month and count mb_used
internet_per_month=internet.groupby(['user_id','month'])['mb_used'].sum().reset_index()
#Rename columnsn for clarity
internet_per_month.columns=['user_id','month', 'mb_used_total']

#Convert total MB to GB and round UP according to Megaline policy
#Note: 1 GB = 1024 MB
internet_per_month['gb_used_billed'] = np.ceil(internet_per_month['mb_used_total'] / 1024).astype(int)

internet_per_month.head()
internet_per_month.head()

#Merge the data for calls, minutes, messages, internet based on user_id and month
#Merge calls and minutes (since they have the same number of rows)
df_usage = pd.merge(calls_per_month, minutes_per_month, on=['user_id', 'month'], how='outer')
#Add the messages count
df_usage = pd.merge(df_usage, messages_per_month, on=['user_id', 'month'], how='outer')
#Add internet traffic 
df_usage = pd.merge(df_usage, internet_per_month, on=['user_id', 'month'], how='outer')
# Fill in NaN's. After an outer merge, if someone didn't use a service, it shows 'NaN'.
df_usage = df_usage.fillna(0)
df_usage.head()

# Add the plan information
df_merged = df_usage.merge(users[['user_id', 'plan', 'city']], on='user_id', how='left')
# add the plan details (costs and limits)
df_merged = df_merged.merge(plans, on='plan', how='left')
#Fill in any NaN's
df_merged=df_merged.fillna(0)
df_merged

#Calculate the monthly revenue for each user
#Update the Revenue Function to be more precise
def calculate_revenue(row):
    #Start with base monthly fee
    total_bill = row['usd_monthly_pay']
    
    #Overage for minutes 
    extra_minutes = max(0, row['minutes_sum'] - row['minutes_included'])
    total_bill += extra_minutes * row['usd_per_minute']
    
    #Overages for messages
    extra_messages = max(0, row['messages_count'] - row['messages_included'])
    total_bill += extra_messages * row['usd_per_message']
    
    #Overage for internet (THE BILLING RULE)
    #Rule: Sum MB -> Subtract limit -> If over, convert overage to GB and round UP
    extra_mb = max(0, row['mb_used_total'] - row['mb_per_month_included'])
    
    if extra_mb > 0:
        # Convert only the overage to GB and round up
        extra_gb = np.ceil(extra_mb / 1024)
        total_bill += extra_gb * row['usd_per_gb']
        
    return total_bill

#Apply the function
df_merged['revenue'] = df_merged.apply(calculate_revenue, axis=1)

#Clean up display
df_merged[['user_id', 'month', 'plan', 'revenue']].head()

# Compare average duration of calls per each plan per each distinct month. Plot a bar plat to visualize it.
# Group by plan and month, then calculate the average duration
avg_calls = df_merged.groupby(['plan', 'month'])['minutes_sum'].mean()

# Plot as a bar chart
avg_calls.plot(kind='bar', figsize=(12, 6), color='skyblue')

# Add labels so the chart is readable
plt.title('Average Call Duration per Plan and Month')
plt.xlabel('Plan, Month')
plt.ylabel('Average Duration (minutes)')
plt.show()

"""
Observations for Average Call Duration Per Plan and Month
1. While monthly averages appear to increase toward year-end, this is likely influenced by the
growing user base. Because more customers joined as the year progressed, this "trend"
reflects a changing sample size rather than a shift in individual behavior.
2. Users on both Surf and Ultimate plans exhibit nearly identical habits, consistently averaging
between 400 and 500 minutes.
3. Average Surf users stay within their 500-minute limit, suggesting call overages are not the
primary revenue driver for that plan.
"""
# Compare the number of minutes users of each plan require each month. Plot a histogram.
# Filter the data into two groups
surf_minutes = df_merged.query('plan == "surf"')['minutes_sum']
ultimate_minutes = df_merged.query('plan == "ultimate"')['minutes_sum']

# Plot both histograms
plt.figure(figsize=(10, 6))
plt.hist(surf_minutes, bins=30, alpha=0.5, label='Surf', color='green')
plt.hist(ultimate_minutes, bins=30, alpha=0.5, label='Ultimate', color='red')

# Add labels and a legend
plt.title('Distribution of Monthly Minutes by Plan')
plt.xlabel('Minutes')
plt.ylabel('Number of Users')
plt.legend(loc='upper right')
plt.show()

"""
Observations about Monthly Minutes by Plan
1. The average Surf user consumes approximately 429 minutes, which is well within the 500-minute allowance.
2.While the average user does not exceed the limit, the distribution shows a significant portion of user-months 
surpassing the 500-minute mark. These specific instances are what drive the overage revenue for the Surf plan.
"""
# Calculate the mean, variance and std of the monthly call duration
# Calculate for Surf
surf_mean = df_merged[df_merged['plan'] == 'surf']['minutes_sum'].mean()
surf_variance = df_merged[df_merged['plan'] == 'surf']['minutes_sum'].var()
surf_std = df_merged[df_merged['plan'] == 'surf']['minutes_sum'].std()
# Calculate for Ultimate
ultimate_mean = df_merged[df_merged['plan'] == 'ultimate']['minutes_sum'].mean()
ultimate_variance = df_merged[df_merged['plan'] == 'ultimate']['minutes_sum'].var()
ultimate_std = df_merged[df_merged['plan'] == 'ultimate']['minutes_sum'].std()
# Print the results clearly
print(f"Surf Plan: Mean = {surf_mean:.2f}, Variance = {surf_variance:.2f},Std Dev = {surf_std:.2f}")
print(f"Ultimate Plan: Mean = {ultimate_mean:.2f}, Variance = {ultimate_variance:.2f},Std Dev = {ultimate_std:.2f}")

# Plot a boxplot to visualize the distribution of the monthly call duration
# Prepare the data piles
surf_minutes = df_merged[df_merged['plan'] == 'surf']['minutes_sum']
ultimate_minutes = df_merged[df_merged['plan'] == 'ultimate']['minutes_sum']

# Plot
plt.figure(figsize=(10, 6))
plt.boxplot([surf_minutes, ultimate_minutes], labels=['Surf', 'Ultimate'])

# Add labels
plt.title('Monthly Call Duration Distribution by Plan')
plt.ylabel('Minutes')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

"""
Conclusion about users and calling (comparing plans)
The data confirms there is no significant difference in calling habits between plans, as both groups average around 430 minutes. However, the value proposition differs greatly:
Approximately 35.98% of Surf user-months exceed the 500-minute limit, generating overage revenue.
In contrast, 0% of Ultimate user-months ever reach the 3,000-minute threshold.
This confirms that Ultimate users are paying a premium for a massive volume of minutes they do not use, while Megaline's additional revenue is primarily driven by Surf users exceeding their much lower limits.
"""
# Calculate the percentage of user-months that exceed the plan limits
surf_over_limit = len(df_merged[(df_merged['plan'] == 'surf') & (df_merged['minutes_sum'] > 500)]) / len(df_merged[df_merged['plan'] == 'surf'])
ultimate_over_limit = len(df_merged[(df_merged['plan'] == 'ultimate') & (df_merged['minutes_sum'] > 3000)]) / len(df_merged[df_merged['plan'] == 'ultimate'])

print(f"Percentage of Surf user-months exceeding 500 min: {surf_over_limit:.2%}")
print(f"Percentage of Ultimate user-months exceeding 3000 min: {ultimate_over_limit:.2%}")

# Compare the number of messages users of each plan tend to send each month. Plot a bar plat to visualize it
avg_messages = df_merged.groupby(['plan', 'month'])['messages_count'].mean()

# 2. Plot as a bar chart
avg_messages.plot(kind='bar', figsize=(12, 6), color='lightgreen')

# Add labels for clarity
plt.title('Average Monthly Messages per Plan')
plt.xlabel('Plan and Month (1=Jan, 12=Dec)')
plt.ylabel('Average Number of Messages')

# Rotate the x-labels so they don't overlap
plt.xticks(rotation=45)
plt.show()

"""
Observations for Average Monthly Messages per Plan
1.There is a clear upward trend in texting for both plans as the year progresses, peaking in December.
2. Most users on both plans stay well below their limits
3. Users on both plans send a similar number of messages, with monthly averages mostly staying between 20 and 40.
"""

# Compare the amount of internet traffic consumed by users per plan. Plot a histogram.
#Creating a histogram to compare message usage 
# Separate the message counts for each plan
surf_messages = df_merged[df_merged['plan'] == 'surf']['messages_count']
ultimate_messages = df_merged[df_merged['plan'] == 'ultimate']['messages_count']

# Plot the histograms
plt.figure(figsize=(10, 6))
plt.hist(surf_messages, bins=30, alpha=0.5, label='Surf', color='lightgreen')
plt.hist(ultimate_messages, bins=30, alpha=0.5, label='Ultimate', color='blue')

# Add details
plt.title('Distribution of Monthly Messages by Plan')
plt.xlabel('Number of Messages')
plt.ylabel('Number of Users')
plt.legend()
plt.show()

"""
Observations for Monthly Messages by Plan
1. Many users—regardless of their plan—hardly use SMS at all in the age of internet messaging.
2. Right skewed
3. For messages, Ultimate users are essentially paying for an "infinite" resource they barely touch.
"""
#calculate mean and variance for messages
surf_msg_mean = df_merged[df_merged['plan'] == 'surf']['messages_count'].mean()
surf_msg_var = df_merged[df_merged['plan'] == 'surf']['messages_count'].var()

ultimate_msg_mean = df_merged[df_merged['plan'] == 'ultimate']['messages_count'].mean()
ultimate_msg_var = df_merged[df_merged['plan'] == 'ultimate']['messages_count'].var()

print(f"Surf Messages: Mean = {surf_msg_mean:.2f}, Variance = {surf_msg_var:.2f}")
print(f"Ultimate Messages: Mean = {ultimate_msg_mean:.2f}, Variance = {ultimate_msg_var:.2f}")

#Plot a boxplot
plt.figure(figsize=(10, 6))
plt.boxplot([df_merged[df_merged['plan'] == 'surf']['messages_count'], 
             df_merged[df_merged['plan'] == 'ultimate']['messages_count']], 
            labels=['Surf', 'Ultimate'])

plt.title('Distribution of Monthly Messages')
plt.ylabel('Number of Messages')
plt.show()

"""
Conclusion on how users behave in terms of messaging: Users do not change their texting habits based on their plan.
The Surf plan is more likely to generate extra revenue from "extreme texters", while the Ultimate plan provides a massive surplus of messages that the average user never touches.
"""
# Create the billed GB column in the main dataframe
df_merged['gb_used_billing'] = np.ceil(df_merged['mb_used_total'] / 1024).astype(int)
avg_internet = df_merged.groupby(['plan', 'month'])['gb_used_billing'].mean()
# Plot
avg_internet.plot(kind='bar', figsize=(12, 6), color='orange')
plt.title('Average Monthly GB Used per Plan')
plt.ylabel('Average GB')
plt.xticks(rotation=45)
plt.show()

"""
Observations for Avg Monthly GB used per plan
1. There is an upward trend in average internet consumption from January through December. This suggests that as users spend more time with 
the service the total volume of data on the network increases.
2. While the Ultimate plan often shows slightly higher average data usage than the Surf plan, the difference is not as large 
as the price gap suggests. Users on both plans seem to follow the same seasonal usage patterns.
"""
#plot histogram 
surf_net = df_merged[df_merged['plan'] == 'surf']['gb_used_billing']
ultimate_net = df_merged[df_merged['plan'] == 'ultimate']['gb_used_billing']

plt.figure(figsize=(10, 6))
plt.hist(surf_net, bins=30, alpha=0.5, label='Surf', color='orange')
plt.hist(ultimate_net, bins=30, alpha=0.5, label='Ultimate', color='blue')
plt.title('Distribution of Internet Usage')
plt.xlabel('GB')
plt.ylabel('Number of Users')
plt.legend()
plt.show()

"""
Observation for histogram
1. The internet usage for both plans typically follows a Normal Distribution. Most users cluster around a central average, with very few users
consuming 0 GB or 40+ GB. 2.The majority of the Ultimate distribution remains well to the left of its 30 GB limit. 
Even though Ultimate users have double the data of Surf users, they do not dramatically increase their consumption. 
Most Ultimate users are paying for data they never touch.
"""
#calculate variance and mean 
print(f"Surf Internet: Mean = {surf_net.mean():.2f}, Variance = {surf_net.var():.2f}")
print(f"Ultimate Internet: Mean = {ultimate_net.mean():.2f}, Variance = {ultimate_net.var():.2f}")

#create a boxplot
plt.figure(figsize=(10, 6))
plt.boxplot([surf_net, ultimate_net], labels=['Surf', 'Ultimate'])
plt.title('Internet Usage Distribution (GB)')
plt.ylabel('GB Used')
plt.show()

"""
Conclusion on Internet Usage Distribution
Internet consumption is much more consistent. The distributions follow a Normal (Bell) Curve, meaning there is a very clear "typical" user behavior for both plans. For the Surf plan, the
average usage often sits right at or slightly above the 15 GB limit. This means the Surf plan is designed in a way that the "average" user is practically guaranteed to pay overage fees at least a
few months out of the year. Also, Surf users are high-value for Megaline because they frequently cross the 15 GB threshold, triggering extra 10 dollar charges per GB. Additionally, Ultimate users
are high-value because they pay a 70 dollar flat fee for a 30 GB allowance that the histogram shows they almost never fully use.
"""
#Bar plot Average Revenue per Month
#Group by plan and month to see the trend
avg_revenue = df_merged.groupby(['plan', 'month'])['revenue'].mean().unstack(level=0)

# Plotting
avg_revenue.plot(kind='bar', figsize=(12, 6), color=['orange', 'red'])
plt.title('Average Monthly Revenue: Surf vs. Ultimate')
plt.xlabel('Month')
plt.ylabel('Average Revenue ($)')
plt.legend(['Surf', 'Ultimate'])
plt.xticks(rotation=0)
plt.show()

"""
Observations
1. It appears that Ultimate brings in more revenue for Megaline.
2. Ultimates revenue is consistent over the year.
3. Surfs most consistent month is December.
"""
#Histogram Revenue Distribution
surf_rev = df_merged[df_merged['plan'] == 'surf']['revenue']
ultimate_rev = df_merged[df_merged['plan'] == 'ultimate']['revenue']

plt.figure(figsize=(10, 6))
plt.hist(surf_rev, bins=30, alpha=0.5, label='Surf', color='orange')
plt.hist(ultimate_rev, bins=30, alpha=0.5, label='Ultimate', color='skyblue')

plt.title('Distribution of Monthly Revenue per User')
plt.xlabel('Revenue ($)')
plt.ylabel('Number of Users')
plt.legend()
plt.show()

"""
Observations
1. The Ultimate plan data appears skinny because almost everyone on the Ultimate plan pays exactly $70 (or close to that).
2. Surf plan data is wider comapred to Ultimate because users have very different bills.
3. Surf plan data aslo has a positive skew which implies most Surf users are clustered on the left side. These are the people who manage to stay close to their limits.
"""
surf_rev = df_merged[df_merged['plan'] == 'surf']['revenue']
ultimate_rev = df_merged[df_merged['plan'] == 'ultimate']['revenue']

# Now the print statements will work
print(f"Surf Revenue: Mean = {surf_rev.mean():.2f}, Variance = {surf_rev.var():.2f}, Std Dev = {surf_rev.std():.2f}")
print(f"Ultimate Revenue: Mean = {ultimate_rev.mean():.2f}, Variance = {ultimate_rev.var():.2f}, Std Dev = {ultimate_rev.std():.2f}")

#Box Plot to visualize the "Profit Zone"
plt.figure(figsize=(10, 6))
plt.boxplot([surf_rev, ultimate_rev], labels=['Surf', 'Ultimate'])
plt.title('Revenue Range and Outliers by Plan')
plt.ylabel('Revenue ($)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

"""
Conclusion: While the Ultimate plan provides a higher guaranteed monthly income of $70 per
user, the Surf plan's right-skewed revenue distribution shows it has higher growth potential
through overage fees. Users on the Surf plan frequently exceed their 15 GB data limit, narrowing
the revenue gap between the two plans. From a business perspective, Surf users are often
paying a higher price-per-GB than Ultimate users
"""
"""
Hypothesis Testing: Surf vs. Ultimate Revenue

Null Hypothesis: The average revenue from users of the Ultimate and Surf calling plans is equal.

Alternative Hypothesis: The average revenue from users of the Ultimate and Surf calling plans differs.

Alpha Level: 0.05

Test Selection: I am using an Independent Two-Sample T-test because I am comparing the means of two distinct groups (different users on different plans).

Justification for Welch’s T-test (equal_var=False): As seen in my descriptive statistics, the variance for the Surf plan is significantly higher than that of the Ultimate plan. Since the groups
have unequal variances and different sample sizes, Welch's T-test is the appropriate choice to ensure an accurate p-value.
"""
# Test the hypotheses to see if average revenues between Surf and Ultimate plans are different

#Set the alpha
alpha = 0.05

#Perform the test
results = st.ttest_ind(surf_rev, ultimate_rev, equal_var=False)

print('p-value:', results.pvalue)

#Compare p-value to alpha
if results.pvalue < alpha:
    print("We reject the null hypothesis: The average revenues are different.")
else:
    print("We cannot reject the null hypothesis: There is no significant difference in revenue.")

"""
Hypothesis Testing: NY-NJ Area vs. Other Regions

Null Hypothesis: The average revenue from users in the NY-NJ area is equal to the average revenue from users in other regions.

Alternative Hypothesis: The average revenue from users in the NY-NJ area differs from the average revenue from users in other regions.

Alpha Level: 0.05.

Test Selection: I am using an Independent Two-Sample T-test to compare the means of these two geographical groups.

Justification for Welch’s T-test: I have used equal_var=False because the sample sizes and variances between the NY-NJ metropolitan area 
 and the rest of the country are unlikely to be equal, making Welch's T-test a more robust choice.
 """
#Test the hypothesis to see if revenue in NY-NJ is different from other regions.
#Filter the two groups
nynj_revenue = df_merged[df_merged['city'].str.contains('NY-NJ', case=False)]['revenue']
other_revenue = df_merged[~df_merged['city'].str.contains('NY-NJ', case=False)]['revenue']

#Set the alpha
alpha = 0.05

#Perform the test
results = st.ttest_ind(nynj_revenue, other_revenue, equal_var=False)

print('p-value:', results.pvalue)

#Compare p-value to alpha
if results.pvalue < alpha:
    print("We reject the null hypothesis: Revenue in NY-NJ is different from other regions.")
else:
    print("We cannot reject the null hypothesis: There is no significant difference in revenue.")

"""
General Conclusion
The analysis of the Surf and Ultimate plans reveals that data consumption, rather than call minutes or messages, is the primary driver of revenue. During the data preprocessing stage, all
call durations were rounded up to the nearest minute according to company policy. For internet usage, monthly totals were converted from megabytes to gigabytes (using 1024 MB per GB) and
rounded up to the nearest whole gigabyte for billing purposes.

While users on both plans illustrate similar usage habits—averaging between 400 and 500 call minutes, their impact on revenue differs greatly. Surf users are significantly more likely to exceed
their 15 GB limit than Ultimate users are to reach their 30 GB threshold. Statistical testing confirmed that the average revenue between the two plans differs significantly. The Surf plan
features a heavily right-skewed revenue distribution, with a high standard deviation driven by frequent, high-margin overage charges.

Also, the geographical analysis also showed a statistically significant difference in revenue between the NY-NJ area and other regions (p≈0.033). This indicates that while plan structure is
the dominant factor in profit, location also plays a measurable role in user spending behavior.

Based on the data, the Surf plan is the superior driver for Megaline's revenue growth. Although the Ultimate plan offers a higher base price of 70 dollars, the Surf plan’s lower 15 GB threshold
creates a consistent and lucrative revenue stream through overage fees. Many Surf "power users" end up paying total monthly bills approaching the cost of the Ultimate plan, but for a
lower volume of included services. The Surf plan provides a low-cost entry point for new customers while capturing additional profit through its 10 dollar per GB overage fee.
"""


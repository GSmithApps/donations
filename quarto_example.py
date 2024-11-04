#%% [markdown]
# ---
# title: "My Python Script"
# format:
#   html:
#     code-fold: true
# ---


#%% [markdown]
# ## Donations
# here is some stuff
# here is some other stuff
# and some more

# here is another set of things

# %%

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd

@dataclass
class Donation:
    date: date
    amount: float

# Sample data
first_donation = Donation(date(2024, 2, 15), 5181.30)
second_donation = Donation(date(2024, 9, 16), 7318.04)
third_donation = Donation(date(2024, 10, 10), 15774.05)
donations = [first_donation, second_donation, third_donation]

# Convert dates to numerical values for KDE
dates_numeric = [datetime.combine(d.date, datetime.min.time()).timestamp() for d in donations]

# Add padding to date range (30 days before and after)
min_date = min(dates_numeric) - (40 * 24 * 60 * 60)  # 30 days in seconds
max_date = max(dates_numeric) + (40 * 24 * 60 * 60)  # 30 days in seconds

amounts = [d.amount for d in donations]

# Create a range of dates for smooth KDE with extended range
date_range = np.linspace(min_date, max_date, 200)

# Calculate KDE
kde = gaussian_kde(dates_numeric, weights=amounts, bw_method=0.073)
density = kde(date_range)

# Create the visualization
plt.figure(figsize=(12, 6))

# Plot the KDE
ax1 = plt.gca()
ax1.plot(
    [datetime.fromtimestamp(x) for x in date_range],
    density,
    'b-',
    label='Donation Density',
    alpha=0.6
)
ax1.fill_between(
    [datetime.fromtimestamp(x) for x in date_range],
    density,
    alpha=0.3
)

# Plot the scatter points for actual donations
ax2 = ax1.twinx()
ax2.scatter(
    [datetime.fromtimestamp(x) for x in dates_numeric],
    amounts,
    color='red',
    s=100,
    label='Donations',
    zorder=5
)

# Formatting
ax1.set_xlabel('Date')
ax1.set_ylabel('Density')
ax2.set_ylabel('Amount ($)')

# Add title and adjust layout
plt.title('Donation Amount Distribution Over Time')
plt.tight_layout()

# Show plot
plt.show()


# %%

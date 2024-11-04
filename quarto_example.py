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

@dataclass
class Donation:
    date: date
    amount: float

sigma = 8  # controls spread of the diffusion (in days)
padding_days = 30

# Sample data
first_donation = Donation(date(2024, 2, 15), 5181.30)
second_donation = Donation(date(2024, 9, 16), 7318.04)
third_donation = Donation(date(2024, 10, 10), 15774.05)
donations = [first_donation, second_donation, third_donation]

# Convert dates to numerical values (days since earliest donation)
base_date = min(d.date for d in donations)
dates_numeric = [(d.date - base_date).days for d in donations]
amounts = [d.amount for d in donations]
total_amount = sum(amounts)

# Create time grid with exactly one point per day
x_grid = np.arange(
    min(dates_numeric) - padding_days,
    max(dates_numeric) + padding_days + 1
)  # integers for days

# Function to create a Gaussian kernel
def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Sum up the diffused donations with relative weights
total_diffusion = np.zeros_like(x_grid, dtype=float)
for date_num, amount in zip(dates_numeric, amounts):
    total_diffusion += amount * gaussian(x_grid, date_num, sigma)

# Normalize so sum equals total donations (since width=1, sum=area)
total_diffusion *= total_amount / sum(total_diffusion)

# Create the visualization
plt.figure(figsize=(12, 6))

# Plot the diffusion
plt.plot(
    [base_date + timedelta(days=int(x)) for x in x_grid],
    total_diffusion,
    'b-',
    alpha=0.6
)
plt.fill_between(
    [base_date + timedelta(days=int(x)) for x in x_grid],
    total_diffusion,
    alpha=0.3
)

# Formatting
plt.xlabel('Date')
plt.ylabel('Dollars per Day')
plt.title('Donations Diffused Over Time')

# Format y-axis as currency
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.show()

# Verify that sum equals total
print(f"Total donations: ${total_amount:,.2f}")
print(f"Sum of daily amounts: ${sum(total_diffusion):.2f}")
# %%

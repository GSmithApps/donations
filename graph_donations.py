from dataclasses import dataclass
from datetime import date, datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Donation:
    date: date
    amount: float

def graph_donations(
    sigma: float,
    padding_days: int,
    donations: list[Donation]
):
    # Create figure with transparent background
    fig = plt.figure(facecolor='none')
    ax = plt.gca()
    ax.set_facecolor('none')  # Make plot background transparent
    
    # Convert dates to numerical values (days since earliest donation)
    base_date = min(d.date for d in donations)
    dates_numeric = [(d.date - base_date).days for d in donations]
    amounts = [d.amount for d in donations]
    total_amount = sum(amounts)
    
    # Create time grid with exactly one point per day
    x_grid = np.arange(
        min(dates_numeric) - padding_days,
        max(dates_numeric) + padding_days + 1
    ) # integers for days
    
    # Function to create a Gaussian kernel
    def gaussian(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    # Sum up the diffused donations with relative weights
    total_diffusion = np.zeros_like(x_grid, dtype=float)
    for date_num, amount in zip(dates_numeric, amounts):
        total_diffusion += amount * gaussian(x_grid, date_num, sigma)
    
    # Normalize so sum equals total donations (since width=1, sum=area)
    total_diffusion *= total_amount / sum(total_diffusion)
    
    # Use a medium-toned color that works well in both light and dark modes
    line_color = '#6b85b5'  # medium steel blue
    fill_color = '#6b85b5'  # same color for consistency
    
    # Plot the diffusion
    plt.plot(
        [base_date + timedelta(days=int(x)) for x in x_grid],
        total_diffusion,
        color=line_color,
        alpha=0.8,
        linewidth=2
    )
    
    plt.fill_between(
        [base_date + timedelta(days=int(x)) for x in x_grid],
        total_diffusion,
        color=fill_color,
        alpha=0.3
    )
    
    # Formatting
    plt.xlabel('Date', fontsize=10, labelpad=10)
    plt.ylabel('Dollars per Day', fontsize=10, labelpad=10)
    plt.title('Donations Diffused Over Time', fontsize=12)
    
    # Format y-axis as currency
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Adjust tick parameters for better visibility
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.xticks(rotation=45, ha='right')
    
    # Make sure spines (borders) are also transparent
    for spine in ax.spines.values():
        spine.set_edgecolor(line_color)
        spine.set_alpha(0.3)
        
    # Add grid with medium opacity
    ax.grid(True, alpha=0.2, color=line_color)
    
    plt.tight_layout()
    plt.savefig('temp.png', transparent=True)  # Save with transparency
    plt.show()
    
    return total_diffusion
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from termcolor import colored

# Configuration settings
pd.options.mode.copy_on_write = True
sns.set_style('darkgrid')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
sns.set_palette("cool_r")

print(colored('\nAll libraries configured successfully.', 'green'))

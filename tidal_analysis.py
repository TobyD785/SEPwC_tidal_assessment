#!/usr/bin/env python3

# import the modules you need here
import argparse
import numpy as np
import pandas as pd
import os
import glob
import pytz
import datetime
from scipy.stats import linregress
import matplotlib.dates as mdates

TIDAL_CONSTITUENT_FREQS = {
    'M2': 1.9322736,
    'S2': 2.0000000,
    'N2': 1.89598186,
    'K1': 0.99856143,
    'O1': 0.9295357,
    # Add more as needed
}

def read_tidal_data(filename):
    try:
        # Find the line number where data starts (line starting with "  1)")
        with open(filename, 'r') as file:
            lines = file.readlines()
            start_idx = next(i for i, line in enumerate(lines) if line.strip().startswith('1)'))

        # Define column specs based on fixed-width format
        colspecs = [(0, 6), (8, 18), (18, 26), (30, 45), (42, 54)]
        column_names = ['Cycle', 'Date', 'Time', 'Sea Level', 'Residual']

        # Read the data section
        df = pd.read_fwf(filename, colspecs=colspecs, names=column_names, skiprows=start_idx)

        # Combine Date and Time into a single datetime column
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y/%m/%d %H:%M:%S', errors='coerce')

        # Process 'Sea Level'
        if isinstance(df['Sea Level'].iloc[0], str):
            df['Sea Level'] = df['Sea Level'].str.strip().replace(r'[MNT]', np.nan, regex=True)
        df['Sea Level'] = pd.to_numeric(df['Sea Level'], errors='coerce')

        # Process 'Residual'
        if isinstance(df['Residual'].iloc[0], str):
            df['Residual'] = df['Residual'].str.strip().replace(r'[MNT]', np.nan, regex=True)
        df['Residual'] = pd.to_numeric(df['Residual'], errors='coerce')
       
        # Drop original text columns
        df = df.drop(columns=['Cycle', 'Date'])

        # Set datetime as index
        df = df.set_index('Datetime')

        # Save to CSV in same directory with '_cleaned' suffix
        output_filename = os.path.splitext(filename)[0] + '_cleaned.csv'
        df.to_csv(output_filename)

        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filename}")


def process_directory(dirname):
    txt_files = glob.glob(os.path.join(dirname, "*.txt"))

    if not txt_files:
        print(f"No .txt files found in '{dirname}'.")
        return None

    all_data = []
    for file in txt_files:
        try:
            if args.verbose:
                print(f"Processing {file}...")
            df = read_tidal_data(file)
            if df is not None: # Only append if processing was successful
                all_data.append(df)
            if args.verbose:
                print(f"Successfully processed {file}.")
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if all_data:
        combined_data = pd.concat(all_data).sort_index()
        print("All successfully processed .txt files combined.")
        return combined_data
    else:
        return None
    
def extract_single_year_remove_mean(year, data):
    year = int(year)

    # Filter data for the given year
    data_year = data[data.index.year == year]

    # Return empty DataFrame if year not found
    if data_year.empty:
        return data_year

    # Drop NaNs before mean subtraction
    data_year = data_year.copy()
    valid = data_year['Sea Level'].notna()
    mean_sea_level = data_year.loc[valid, 'Sea Level'].mean()

    # Subtract mean only where Sea Level is valid
    data_year.loc[valid, 'Sea Level'] -= mean_sea_level

    return data_year

def extract_section_remove_mean(start, end, data):
    try:
        start_dt = pd.to_datetime(start, format="%Y%m%d")
        end_dt = pd.to_datetime(end, format="%Y%m%d")
    except ValueError:
        return pd.DataFrame(columns=data.columns)

    # Create complete hourly time index
    full_index = pd.date_range(start=start_dt, end=end_dt + pd.Timedelta(hours=23), freq='h')

    # Filter and reindex to full hourly range
    section = data.loc[data.index.intersection(full_index)].copy()
    section = section.reindex(full_index)

    # Subtract mean only from valid entries
    if 'Sea Level' in section.columns:
        valid = section['Sea Level'].notna()
        mean = section.loc[valid, 'Sea Level'].mean()
        section.loc[valid, 'Sea Level'] -= mean

    return section


def join_data(*data_frames):
    """
    Joins multiple dataframes. If any dataframe is None, it's skipped.
    Performs an outer join on all dataframes, aligning columns and 
    handling mismatches with NaN.

    Parameters:
    *data_frames (pd.DataFrame): Dataframes to be joined. Can handle multiple dataframes.

    Returns:
    pd.DataFrame: The resulting dataframe after joining all provided dataframes.
    """
    # Filter out None values from the input dataframes
    data_frames = [df for df in data_frames if df is not None]
    
    if not data_frames:
        print("No dataframes to join.")
        return None

    # Perform the join by concatenating all dataframes along rows (axis=0)
    joined_data = pd.concat(data_frames, axis=0, join='outer').sort_index()

    print(f"Dataframes joined successfully. {len(data_frames)} dataframes joined.")
    return joined_data


def get_longest_contiguous_data(data):
    """
    Finds the longest contiguous block of valid sea level data in the given dataframe.
    
    Parameters:
    data (pd.DataFrame): Dataframe containing sea level data.
    
    Returns:
    pd.DataFrame: The longest contiguous block of valid sea level data.
    """
    if data is None or 'Sea Level' not in data.columns:
        print("Invalid data or 'Sea Level' column missing.")
        return pd.DataFrame()

    # Only consider valid sea level entries (non-NaN values)
    valid = data['Sea Level'].notna()

    # Identify contiguous blocks of valid data
    group_id = (valid != valid.shift()).cumsum()

    # Filter only valid blocks (those without NaN in 'Sea Level')
    valid_blocks = data[valid].groupby(group_id)

    # Find the block with the maximum length
    longest_block = max(valid_blocks, key=lambda x: len(x[1]))[1]

    return longest_block



def sea_level_rise(data):
    # Use only the longest valid block of continuous hourly Sea Level data
    longest = get_longest_contiguous_data(data)

    if longest.empty:
        raise ValueError("No valid contiguous sea level data to analyze.")

    # Convert datetime index to numerical format (days since epoch)
    x = mdates.date2num(longest.index.to_pydatetime())
    y = longest['Sea Level'].values

    # Perform linear regression
    result = linregress(x, y)

    return result.slope, result.pvalue



def tidal_analysis(data, constituents, start_datetime):
    # Ensure datetime index and drop NaNs
    df = data.dropna(subset=['Sea Level'])
    if df.empty:
        return [], []

    # Make df index tz-aware if it's tz-naive
    if df.index.tz is None:
        df.index = df.index.tz_localize(start_datetime.tzinfo)

    # Convert datetime index to time in days since start
    time_hours = (df.index - start_datetime).total_seconds() / 3600.0
    time_days = time_hours / 24.0
    sea_level = df['Sea Level'].values

    # Build design matrix with sin/cos terms
    X = []
    for name in constituents:
        freq = TIDAL_CONSTITUENT_FREQS.get(name)
        if freq is None:
            raise ValueError(f"Unknown tidal constituent: {name}")
        omega = 2 * np.pi * freq
        X.append(np.cos(omega * time_days))
        X.append(np.sin(omega * time_days))

    X = np.column_stack(X)

    # Solve least squares
    coeffs, _, _, _ = np.linalg.lstsq(X, sea_level, rcond=None)

    amp = []
    pha = []
    for i in range(0, len(coeffs), 2):
        a, b = coeffs[i], coeffs[i + 1]
        amplitude = np.sqrt(a**2 + b**2)
        phase = np.arctan2(-b, a) * 180 / np.pi
        if phase < 0:
            phase += 360
        amp.append(amplitude)
        pha.append(phase)

    return amp, pha


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="UK Tidal analysis",
        description="Calculate tidal constituents and RSL from tide gauge data",
        epilog="Copyright 2024, Jon Hill"
    )

    parser.add_argument("directory",
                        help="the directory containing txt files with data")
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help="Print progress")

    args = parser.parse_args()
    dirname = args.directory

    process_directory(dirname)
"""This module provides functions for analyzing tidal data.

It includes functions for reading data from text files,
calculating sea level rise, and performing basic tidal analysis.
"""

#!/usr/bin/env python3

# Import necessary modules
import argparse
import os
import glob
from datetime import datetime
import pytz
from scipy.stats import linregress
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

TIDAL_CONSTITUENT_FREQS = {
    'M2': 1.9322736,
    'S2': 2.0000000,
    'N2': 1.89598186,
    'K1': 0.99856143,
    'O1': 0.9295357,
    # Add more as needed
}


def read_tidal_data(filename):
    """
    Reads tidal data from a given file and returns a cleaned dataframe.

    Args:
        filename (str): Path to the tidal data file.

    Returns:
        pd.DataFrame: Cleaned data with datetime index and numeric sea level and residual.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        # Find the line number where data starts (line starting with " 1)")
        with open(filename, 'r') as file:
            lines = file.readlines()
            start_idx = next(i for i, line in enumerate(lines) if line.strip().startswith('1)'))

        # Define column specs based on fixed-width format
        colspecs = [(0, 6), (8, 18), (18, 26), (30, 45), (42, 54)]
        column_names = ['Cycle', 'Date', 'Time', 'Sea Level', 'Residual']

        # Read the data section
        df = pd.read_fwf(filename, colspecs=colspecs, names=column_names, skiprows=start_idx)

        # Combine Date and Time into a single datetime column
        df['Datetime'] = pd.to_datetime(df['Date'] + ' '
                                        + df['Time'], format='%Y/%m/%d %H:%M:%S', errors='coerce')

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
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {filename}") from e


def process_directory(dirname):
    """
    Processes all tidal data files in the given directory and combines them into a single dataframe.

    Args:
        dirname (str): Path to the directory containing the tidal data files.

    Returns:
        pd.DataFrame: Combined dataframe with data from all files in the directory, 
        or None if no valid data is found.
    """
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
            if df is not None:  # Only append if processing was successful
                all_data.append(df)
            if args.verbose:
                print(f"Successfully processed {file}.")
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if all_data:
        c_data = pd.concat(all_data).sort_index()
        print("All successfully processed .txt files combined.")
        return c_data

    return None


def extract_single_year_remove_mean(year, data):
    """
    Extracts data for a specific year and removes the mean from the sea level values.

    Args:
        year (int): Year to filter.
        data (pd.DataFrame): Data containing sea level and datetime index.

    Returns:
        pd.DataFrame: Data for the specified year with the mean sea level removed.
    """
    year = int(year)

    # Filter data for the given year
    data_year = data[data.index.year == year].copy()

    # Return empty DataFrame if year not found
    if data_year.empty:
        return data_year

    # Drop NaNs before mean subtraction
    valid = data_year['Sea Level'].notna()
    mean_sea_level = data_year.loc[valid, 'Sea Level'].mean()

    # Subtract mean only where Sea Level is valid
    data_year.loc[valid, 'Sea Level'] -= mean_sea_level

    return data_year


def extract_section_remove_mean(start, end, data):
    """
    Extracts data for a specific date range and removes the mean from the sea level values.

    Args:
        start (str): Start date in YYYYMMDD format.
        end (str): End date in YYYYMMDD format.
        data (pd.DataFrame): Data containing sea level and datetime index.

    Returns:
        pd.DataFrame: Data for the specified date range with the mean sea level removed.
    """
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

    Args:
        *data_frames (pd.DataFrame): Dataframes to be joined.

    Returns:
        pd.DataFrame: The resulting dataframe after joining all provided dataframes.
    """
    # Filter out None values from the input dataframes
    valid_data_frames = [df for df in data_frames if df is not None]

    if not valid_data_frames:
        print("No dataframes to join.")
        return None

    # Perform the join by concatenating all dataframes along rows (axis=0)
    joined_data = pd.concat(valid_data_frames, axis=0, join='outer').sort_index()

    print(f"Dataframes joined successfully. {len(valid_data_frames)} dataframes joined.")
    return joined_data


def sea_level_rise(data):
    """
    Calculates the sea level rise rate using linear regression on all available sea level data.

    Args:
        data (pandas.DataFrame): DataFrame containing sea level data.

    Returns:
        tuple: (slope, p-value) of the linear regression, or (None, None) if no valid data.
    """
    valid_data = data.dropna(subset=['Sea Level'])

    if valid_data.empty:
        print("No valid sea level data to analyze.")
        return None, None  # Return None, None

    # Convert datetime index to numerical format (days since epoch)
    x = mdates.date2num(valid_data.index.to_pydatetime())
    y = valid_data['Sea Level'].values

    # Perform linear regression
    result = linregress(x, y)

    return result.slope, result.pvalue


def tidal_analysis(data, constituents, start_datetime):
    """
    Performs tidal analysis to calculate amplitudes and phases of specified tidal constituents.

    Args:
        data (pandas.DataFrame): DataFrame containing sea level data with a datetime index.
        constituents (list): List of tidal constituent names (e.g., ['M2', 'S2']).
        start_datetime (datetime.datetime): 
            The starting datetime used as a reference for time calculations.

    Returns:
        tuple: (amplitudes, phases) - lists of amplitudes and phases for the given constituents.
    """
    df = data.dropna(subset=['Sea Level'])
    if df.empty:
        print("No Sea Level data available for tidal analysis.")
        return [], []

    # Make df index tz-aware if it's tz-naive
    if df.index.tz is None and start_datetime.tzinfo is not None:
        df.index = df.index.tz_localize(start_datetime.tzinfo)
    elif df.index.tz is not None and start_datetime.tzinfo is None:
        start_datetime = start_datetime.replace(tzinfo=df.index.tz)
    elif df.index.tz is None and start_datetime.tzinfo is None:
        pass  # Both are naive, no localization needed

    # Convert datetime index to time in days since start
    time_hours = (df.index - start_datetime).total_seconds() / 3600.0
    time_days = time_hours / 24.0
    sea_level = df['Sea Level'].values

    # Build design matrix with sin/cos terms
    trig = []
    for name in constituents:
        freq = TIDAL_CONSTITUENT_FREQS.get(name)
        if freq is None:
            raise ValueError(f"Unknown tidal constituent: {name}")
        omega = 2 * np.pi * freq
        trig.append(np.cos(omega * time_days))
        trig.append(np.sin(omega * time_days))

    trig = np.column_stack(trig)

    # Solve least squares
    coeffs, _, _, _ = np.linalg.lstsq(trig, sea_level, rcond=None)

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

    parser.add_argument(
        "directory",
        help="The directory containing txt files with data"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=False,
        help="Print progress"
    )

    args = parser.parse_args()
    dirname = args.directory

    combined_data = process_directory(dirname)  # Store the result

    if combined_data is not None:
        # Example usage of the functions, assuming you have data loaded
        # Replace this with how you want to use the functions
        print("Example Usage (requires data in combined_data):")
        print("-" * 40)

        year_data = extract_single_year_remove_mean(2010, combined_data)
        if not year_data.empty:
            print(f"Data for year 2010 (first 5 rows):\n{year_data.head()}")
        else:
            print("No data for year 2010")

        section_data = extract_section_remove_mean("20100101", "20100110", combined_data)
        if not section_data.empty:
            print(f"Data for date range (first 5 rows):\n{section_data.head()}")
        else:
            print("No data for the date range")

        slope, p_value = sea_level_rise(combined_data)
        if slope is not None and p_value is not None:
            print(f"Sea level rise rate: {slope:.4f} mm/year, p-value: {p_value:.3f}")
        else:
            print("Could not calculate sea level rise.")

        if not combined_data.empty:
            start_time = combined_data.index.min()
            constituents = ['M2', 'S2']
            amplitudes, phases = tidal_analysis(combined_data, constituents, start_time)
            if amplitudes and phases:
                print(f"Tidal analysis for constituents {constituents}:")
                for i, constituent in enumerate(constituents):
                    print(f"  {constituent}: Amplitude = {amplitudes[i]:.2f},"
                          f"Phase = {phases[i]:.2f}")
            else:
                print("Tidal analysis failed.")
        else:
            print("No data available for tidal analysis.")
    else:
        print("No data to process.")

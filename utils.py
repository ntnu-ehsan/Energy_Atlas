import csv
import rasterio
from pyproj import Transformer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import os


def process_tiff(file_path, name=None, plot=False, save_csv=False):
    if name is None:
        name = "output"

    filtered_csv = f"{name}_filtered.csv"
    converted_csv = f"{name}_converted.csv"

    # Open the raster
    with rasterio.open(file_path) as src:
        data = src.read(1)
        coords = []

        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                value = data[y, x]
                if value > 0:
                    lon, lat = src.transform * (x, y)
                    coords.append((lon, lat, value))

    # Write the filtered data to a CSV file
    with open(filtered_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Longitude', 'Latitude', 'Electricity Value'])
        writer.writerows(coords)

    # Initialize transformer from EPSG:3035 to EPSG:4326 (WGS84)
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

    # Convert coordinates to WGS84
    converted_data = []
    with open(filtered_csv, 'r') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip the header

        for row in reader:
            x, y, value = map(float, row)
            lon, lat = transformer.transform(x, y)
            converted_data.append([lon, lat, value])

    df = pd.DataFrame(converted_data, columns=['Longitude (WGS84)', 'Latitude (WGS84)', 'Electricity Value'])

    if save_csv:
        df.to_csv(converted_csv, index=False)
        print(f"Converted CSV saved as {converted_csv}")

    # Plot if required
    if plot:
        logs = np.log10(df['Electricity Value'] + 1)

        plt.figure(figsize=(10, 8))
        plt.scatter(
            df['Longitude (WGS84)'],
            df['Latitude (WGS84)'],
            c=logs,
            cmap='viridis',
            s=1,
            alpha=0.6
        )

        cbar = plt.colorbar(label='Log10(Electricity Demand + 1)')
        cbar.set_ticks([0, 1, 2, 3, 4, 5])
        cbar.set_ticklabels(['1', '10', '100', '1k', '10k', '100k'])

        plt.title('Logarithmic Scale: Electricity Demand Across Europe (2019)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.show()

    return df

# Example usage:
# df = process_tiff("Atlas/electricity_tri_demand_2019.tif", name="electricity_data", plot=True, save_csv=True)




def filter_by_country(df, country_name, shapefile_path, save=False, plot=False, type_name=''):
    """
    Filters electricity demand CSV by country boundary.

    Parameters:
    - csv_path: str, path to the CSV file with longitude, latitude, and electricity values
    - country_name: str, name of the country to filter for
    - shapefile_path: str, path to the European countries shapefile
    - save_path: str or None, if set, saves the filtered CSV to this path
    - plot: bool, if True, plots the filtered demand points and country border

    Returns:
    - filtered_df: pandas DataFrame with only the points inside the country
    """


    # Convert to GeoDataFrame with EPSG:4326
    points_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude (WGS84)"], df["Latitude (WGS84)"]),
        crs="EPSG:4326"
    )

    # Load shapefile
    countries_gdf = gpd.read_file(shapefile_path)

    # Ensure same CRS
    countries_gdf = countries_gdf.to_crs(points_gdf.crs)

    # Filter for selected country
    if country_name not in countries_gdf["name"].values:
        raise ValueError(f"Country '{country_name}' not found in shapefile.")

    country_geom = countries_gdf[countries_gdf["name"] == country_name]

    # Spatial join: keep only points within the selected country
    filtered_gdf = gpd.sjoin(points_gdf, country_geom, how="inner", predicate="within")

    # Drop geometry and join-related columns
    filtered_df = filtered_gdf.drop(columns=["geometry", "index_right"], errors="ignore")


    # Save if needed
    if save:
        save_path = os.path.join(f"M:\\Work\\PowerDig\\Data\\Atlas_European_Energy\\Electricity\\{country_name}\\electricity_{type_name}.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        filtered_df.to_csv(save_path, index=False)
        print(f"Saved filtered data to: {save_path}")

    # Plot if requested
    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        country_geom.boundary.plot(ax=ax, edgecolor='red', linewidth=1)
        filtered_gdf.plot(ax=ax, markersize=2, color='blue', alpha=0.6)
        plt.title(f"Electricity Demand Points in {country_name}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)
        plt.show()

    return filtered_df
# Example usage:
# country_df = filter_by_country(df, "Germany", plot=True, save=True)


# Example usage:
# country_df = filter_by_country(df, "Germany")
#%%
def plot_atlas_data(atlas_data):
    logs = np.log10(atlas_data['Electricity Value'] + 1)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        atlas_data['Longitude (WGS84)'],
        atlas_data['Latitude (WGS84)'],
        c=logs,
        cmap='viridis',
        s=1,
        alpha=0.6
    )

    cbar = plt.colorbar(label='Log10(Electricity Demand + 1)')
    cbar.set_ticks([0, 1, 2, 3, 4, 5])
    cbar.set_ticklabels(['1', '10', '100', '1k', '10k', '100k'])
    plt.title('Logarithmic Scale: Electricity Demand Across Europe (2019)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()
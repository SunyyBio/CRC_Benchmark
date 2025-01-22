import os
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Get the current file path and the output directory
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
output_dir = os.path.join(parent_dir, "Result", "figures","Fig01")
os.makedirs(output_dir, exist_ok=True)

# Data for map visualization
data = {
    'city': [
        'United States of America', 'Canada', 'Germany', 'France', 'India',
        'Italy', 'Japan', 'China', 'Austria', 'HK,China'
    ],
    'region': [
        'United States of America', 'Canada', 'Germany', 'France', 'India',
        'Italy', 'Japan', 'China', 'Austria', 'Hong Kong'
    ],
    'latitude': [42.28, 43.65, 52.52, 48.8566, 23.26, 41.9028, 35.6895, 31.40527, 48.2082, 22.3193],
    'longitude': [-83.74, -79.38, 10.2714, 2.3522, 77.41, 12.4964, 139.6917, 121.48941, 16.3738, 114.1694],
    'CTR': [82, 3, 62, 61, 30, 51, 245, 324, 58, 53],
    'ADA': [30, 22, 0, 40, 0, 27, 67, 117, 46, 0],
    'CRC': [71, 2, 58, 50, 30, 59, 110, 260, 46, 75]
}

# Create GeoDataFrame
df = pd.DataFrame(data)
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Load world map data and merge with statistics
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world.merge(df.groupby('region').sum().reset_index(), how='left', left_on='name', right_on='region')
df.loc[df['city'] == 'HK,China', 'region'] = 'Hong Kong'

# Define color mapping for regions
color_map = {
    'Austria': (71 / 255, 133 / 255, 194 / 255),
    'France': (243 / 255, 199 / 255, 102 / 255),
    'Germany': (77 / 255, 88 / 255, 104 / 255),
    'United States of America': (222 / 255, 190 / 255, 53 / 255),
    'India': (158 / 255, 193 / 255, 186 / 255),
    'Canada': (239 / 255, 168 / 255, 89 / 255),
    'Italy': (255 / 255, 102 / 255, 0 / 255),
    'Japan': (86 / 255, 44 / 255, 141 / 255),
    'China': (214 / 255, 83 / 255, 68 / 255),
    'Hong Kong': (100 / 255, 183 / 255, 225 / 255)
}
default_color = (249 / 255, 238 / 255, 238 / 255)

# Assign colors to countries based on the region
world['color'] = world['region'].apply(lambda x: color_map.get(x, default_color))

# Add Hong Kong data as a separate entry
hong_kong_entry = gpd.GeoDataFrame([{
    'name': 'Hong Kong',
    'geometry': Point(114.1694, 22.3193),
    'color': (100 / 255, 183 / 255, 225 / 255)
}])
world = pd.concat([world, hong_kong_entry], ignore_index=True)

# Plot the world map
fig, ax = plt.subplots(1, 1, figsize=(8, 15))
world.boundary.plot(ax=ax, linewidth=0.2)
world.plot(ax=ax, color=world['color'], edgecolor='black', linewidth=0.5, linestyle='--')
ax.set_title('World Map with Colored Countries', fontsize=16)

# Save the map as a PDF
map_output_path = os.path.join(output_dir, "Fig1_colored_world_map.pdf")
plt.savefig(map_output_path, dpi=300, bbox_inches='tight')
plt.close(fig)

# Data for pie chart visualization
output_path = os.path.join(output_dir, "Fig1_region_pie.pdf")

def draw_and_save_pie_chart(ctr, ada, crc, city, pdf, scale_factor=1):
    """Draw and save a pie chart for a single city to a PDF file."""
    sizes = [ctr, ada, crc]
    labels = [f'CTR: {ctr}', f'ADA: {ada}', f'CRC: {crc}']
    colors = ['#F8C789', '#74B4E3', '#9FCC60']
    total = ctr + ada + crc
    size_scale = math.log(total + 1) * scale_factor

    # Plot the pie chart
    fig, ax = plt.subplots(figsize=(size_scale, size_scale))
    ax.pie(sizes, labels=labels, colors=colors, startangle=90)
    ax.axis('equal')
    ax.set_title(city, fontsize=14)

    # Save the chart to the PDF
    pdf.savefig(fig, transparent=True)
    plt.close(fig)

# Generate and save pie charts for all cities
scale_factor = 0.2
with PdfPages(output_path) as pdf:
    for i in range(len(data['city'])):
        draw_and_save_pie_chart(
            ctr=data['CTR'][i],
            ada=data['ADA'][i],
            crc=data['CRC'][i],
            city=data['city'][i],
            pdf=pdf,
            scale_factor=scale_factor
        )

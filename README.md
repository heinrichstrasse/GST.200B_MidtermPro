25W GST.200UB GIS analysis techniques 2 - Midterm Project – Group 5: Kocheganova Adilya, Wutte Timon, Flor Maximilian
 
Project Titel: "Urban Green Space Accessibility & Suitability Analysis for Graz"

This repository documents a Jupyter Notebook workflow for reproducing a green space suitability and accessibility analysis. 
The methodology follows and adapts the approach from Moisa et al. (2023), implemented entirely with open-source geospatial libraries.

OVERVIEW
Urban green spaces provide essential ecological and social benefits.
Using multi-criteria evaluation (MCE) and network analysis, this project:
1. Identifies suitable areas for potential green space development.
2. Assesses accessibility of existing green spaces for residents.
3. Reveals spatial inequalities in environmental and infrastructural conditions.


DATA SOURCES

- Road network (OSM)
- River network (OSM) 
- DEM (GIS Steiermark)  
- Population density (WorldpopHup)  
- Urban Atlas Graz 2018 (Copernicus Land Monitoring Service) 
- Sentinel 2 NDVI imagery (ESA)  

All datasets were reprojected to EPSG:31256 (MGI / Austria GK M34)


METHOD SUMMARY

1. Preprocessing  
- Mosaicking raster tiles
- Reprojecting (EPSG 31256)  
- Clipping (Graz)  
- Rasterizing (1 m resolution)

2. Deriving Criteria  
- Slope (DEM)  
- NDVI (vegetation)  
- Distance to roads & rivers  
- Population density  
- Landuse (Urban Atlas)

3. Suitability Modeling  
- Normalization of criteria  
- Weighting via Analytic Hierarchy Process (AHP)  
- Weighted overlay - Suitability Index

4. Accessibility Modeling  
- network extraction via OSMnx  
- Walking distance & travel time to green spaces  
- Aggregation to population grid

OUTPUTS

- Suitability map  
- Accessibility surface  
- Combined inequality assessment  


ESSENTIAL TOOLS

- GeoPandas  
- Rasterio / rioxarray  
- OSMnx  
- NumPy / SciPy  
- Matplotlib  


ORIGINAL PAPER
  
Moisa et al. (2023): Urban green space suitability analysis using geospatial techniques: a case study of Addis Ababa, Ethiopia. 
Geocarto International. DOI: 10.1080/10106049.2023.2213674.
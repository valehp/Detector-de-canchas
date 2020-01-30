import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.ops import cascaded_union
from shapely.geometry import Polygon, LineString, Point, MultiPoint, MultiPolygon
import time
import statistics as stats

from DCPA import detCanchasAncho, postJuntarPuntos

fp = './shapefile/20059.shp'

# Extraemos los caminos del ShapeFile y lo hacemos GeoSeries.
data = gpd.read_file(fp)
caminos = data[data.TIPOUSO == "FCAM"]
caminos = caminos["geometry"]
caminos = gpd.GeoSeries(caminos)

# Detectamos canchas.
canchas = detCanchasAncho(caminos, 3, 1.1)
canchas = postJuntarPuntos(caminos, canchas)
canchas = gpd.GeoSeries(canchas)

base = caminos.plot(color='green', alpha=0.5)
canchas.plot(ax=base, color='red')
plt.show()

# Guardamos las canchas detectadas.
centroides = gpd.GeoDataFrame(columns=["centro"],geometry = "centro")
centroides['centro'] = canchas
out = './shapefile/20059_canchas/20059_canchas.shp'
centroides.to_file(out)

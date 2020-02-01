import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.ops import cascaded_union
from shapely.geometry import Polygon, LineString, Point, MultiPoint, MultiPolygon
import time
import statistics as stats
import math as m
from collections import deque

# ------------ Parametros:
# caminos: lista de caminos del shapefile, debe funcionar cuando es GeoSeries o lista de Polygons.
# lengthComp: Comparaciones seguidas que deben cumplir con la sensibilidad para ser cancha, 3 default.
# sensProp: Crecimiento proporcional de ancho que se considera cancha, 1.1 por default.
# ------------ Retorna:
# Lista de centros, en formato Point de Shapely, los cuales son las predicciones de canchas.
def detCanchasAncho(caminos, lengthComp=3, sensProp=1.1):
	canchas = []

	for cam in caminos:
		difs = [0]*lengthComp

		# Concatenamos tanto los puntos interiores como exteriores.
		x, y = cam.exterior.coords.xy
		camInt = list(cam.interiors)
		if camInt:
			for it in camInt:
				xint, yint = it.coords.xy
				x.extend(xint)
				y.extend(yint)

		# Se itera por los puntos del Polygon.
		for i in range(len(x)-1):
			difx = x[i+1]-x[i]
			dify = y[i+1]-y[i]

			# Todo esto es para crear una linea perpendicular a la line analizada.
			if difx == 0:
				m = float('inf')
			else:
				m = dify/difx
			if m == 0:
				perpm = float('inf')
			else:
				perpm = -1/m
			px = (x[i+1]+x[i])/2
			py = (y[i+1]+y[i])/2
			a = py - perpm*px
			if perpm < 1 and perpm > -1:
				finalx1 = px - 50
				finalx2 = px + 50
				finaly1 = perpm*finalx1 + a
				finaly2 = perpm*finalx2 + a
			else:
				finaly1 = py - 50
				finaly2 = py + 50
				if perpm == float('inf'):
					finalx1 = px;
					finalx2 = px;
				else:
					finalx1 = (finaly1 - a)/perpm
					finalx2 = (finaly2 - a)/perpm
			line = LineString([(finalx1, finaly1), (finalx2, finaly2)])
			if not line.is_valid:
				continue

			# Intersectamos la linea creada con el Polygon. 
			pt = Point(px, py)
			inter = bestLine(cam, line, pt)
			if not inter:
				difs = [0]*lengthComp
				continue
			# Se almacenan anchos anteriores en difs, los cuales se vuelven 0 si algun ancho siguiente no cumple con
			# la sensibilidad establecida, si se cumplen por las veces establecidas el aumento de ancho, se añade como
			# cancha.
			if difs[i%lengthComp] != 0:
				widthProp = inter.length / difs[i%lengthComp]
				if widthProp > sensProp:
					canchas.append(inter.centroid)
			for comps in range(lengthComp):
				if comps == i%lengthComp:
					difs[comps] = inter.length
				elif difs[comps] != 0 and inter.length / difs[comps] < sensProp:
					difs[comps] = 0

	return canchas

# ------------ Parametros:
# pol: Polygon del camino en el que se quiere intersectar.
# line: Linea con la cual se va a intersectar.
# pt: Point usado para la creacion de la linea.
# ------------ Retorna:
# La mejor linea que describe la interseccion, la cual es la de ancho más grande y más
# cercana al punto pt. Si no ce encontro una interseccion, o la interseccion no es cercana
# al punto, se retorna 0.
def bestLine(pol, line, pt):
	inter = pol.intersection(line)
	interSelect = 0
	maxLen = 0
	try:
		for i in inter:
			if pt.distance(i) < 0.1 and i.length > maxLen:
				interSelect = i
				maxLen = i.length
	except TypeError:
		interSelect = inter
	return interSelect

# ------------ Parametros:
# caminos: Lista de polygons de caminos del shapefile, debe funcionar cuando es GeoSeries o lista
# centros: Lista de Points de centros de las canchas detectadas (OJO que es lista y no GeoDataFrame)
# separateTol: Tolerancia de distancia a la que no combina los puntos, 25 por default.
# ------------ Retorna:
# Una lista de centros mezcladas.
def postJuntarPuntos(caminos, centros, separateTol=25):
	canchList = []
	for cam in caminos:
		canchList.append(cam)
	MPcanch = MultiPolygon(canchList)

	trueCentros = []
	while centros:
		auxX = [centros[0].x]
		auxY = [centros[0].y]
		minX = centros[0]
		maxX = centros[0]
		minY = centros[0]
		maxY = centros[0]
		centros.pop(0)
		selectedPoints = []
		for j in range(len(centros)):
			if minX.distance(centros[j]) < separateTol or maxX.distance(centros[j]) < separateTol or minY.distance(centros[j]) < separateTol or minY.distance(centros[j]) < separateTol:
				lin1 = LineString([centros[j], minX])
				lin2 = LineString([centros[j], maxX])
				lin3 = LineString([centros[j], minY])
				lin4 = LineString([centros[j], maxY])
				if not MPcanch.contains(lin1) or not MPcanch.contains(lin2) or not MPcanch.contains(lin3) or not MPcanch.contains(lin4):
					break
				if minX.x > centros[j].x:
					minX = centros[j]
				if maxX.x < centros[j].x:
					maxX = centros[j]
				if minY.y > centros[j].y:
					minY = centros[j]
				if maxY.y < centros[j].y:
					maxY = centros[j]
				auxX.append(centros[j].x)
				auxY.append(centros[j].y)
				selectedPoints.append(j)
		meanX = stats.mean(list(auxX))
		meanY = stats.mean(list(auxY))
		trueCentros.append(Point(meanX, meanY))
		popCount = 0
		for j in selectedPoints:
			centros.pop(j - popCount)
			popCount += 1

	return trueCentros

# ------------ Parametros:
# tp_polygons: Polygons que etiquetan las canchas.
# fp_polygons: Polygons que etiquetan lugares que no son canchas.
# lista_centros: Lista de Points.
# ------------ Retorna:
# Arreglo de dos elementos, el primero son los verdaderos positivos,
# mientras que el segundo elemento son los falsos posiivos.

def calcularErrores(tp_polygons, fp_polygons, lista_centros):
	tp = 0
	fp = 0
	tp_pol_copy = tp_polygons.copy()
	fp_pol_copy = fp_polygons.copy()
	for centro in lista_centros:
		found = False
		for poly in tp_pol_copy:
			if centro.within(poly):
				tp += 1 
				tp_pol_copy.remove(poly)
				found = True
				break
		if not found:
			fp += 1
	return (tp, fp)
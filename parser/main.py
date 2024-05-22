import os
import json
import urllib.request
import numpy as np
import math
import requests
from PIL import Image
from tqdm import tqdm
from io import BytesIO

TEMP_PATH = 'temp.png'
SIZE = 600
EARTH_RADIUS = 6371
ZOOM = 19
API_KEY = ''
DATASET_PATH = 'dataset'

def scheme_url(lat, lng):
	return f'https://maps.googleapis.com/maps/api/staticmap?center={lat:.6f},{lng:.6f}&zoom={ZOOM}&size={SIZE}x{SIZE}&maptype=rodamap&style=feature:all|element:labels|visibility:off&key={API_KEY}'

def satellite_url(lat, lng):
	return f'https://maps.googleapis.com/maps/api/staticmap?center={lat:.6f},{lng:.6f}&zoom={ZOOM}&size={SIZE}x{SIZE}&maptype=satellite&key={API_KEY}'

def open_pillow(url):
	urllib.request.urlretrieve(url, TEMP_PATH)
	return Image.open(TEMP_PATH)

def generate(cities, filename_gen, meta):
	for name, city in cities.items():
		if city['count'] <= 0:
			continue
		count = city['count']

		print(f'Generating {name}, {count} images')
		centerLat = city['lat']
		centerLong = city['long']
		radius = city['radius']

		latDegKm = 2 * math.pi * EARTH_RADIUS / 360
		longDegKm = 2 * math.pi * EARTH_RADIUS * math.cos(centerLat / 180 * math.pi) / 360

		for i in tqdm(range(count)):
			lat = round(centerLat + np.random.uniform(-1, 1) * radius / latDegKm, 6)
			lng = round(centerLong + np.random.uniform(-1, 1) * radius / longDegKm, 6)
			filename = next(filename_gen)
			dst = Image.new('RGB', (2 * SIZE, SIZE))
			satellite = open_pillow(satellite_url(lat, lng))
			dst.paste(satellite, (0, 0))
			scheme = open_pillow(scheme_url(lat, lng))
			dst.paste(scheme, (SIZE, 0))
			dst.save(os.path.join(DATASET_PATH, filename))
			meta[filename] = {
				'city': name,
				'lat': f'{lat:.6f}',
				'long': f'{lng:.6f}'
			}

def main():
	np.random.seed(43)

	cities = json.loads(open('cities.json').read())
	os.makedirs(DATASET_PATH, exist_ok=True)
	try:
		meta = json.loads(open('meta.json').read())
	except:
		meta = {}

	for name, city in cities.items():
		for field in ['lat', 'long', 'radius', 'count']:
			if field not in city:
				raise Exception(f'Incorrect config, missing {field} in {name}')

	for name, elem in meta.items():
		if elem['city'] in cities:
			cities[elem['city']]['count'] -= 1

	def gen():
		i = 0
		while True:
			if f'{i}.png' not in meta:
				yield f'{i}.png'
			i += 1

	try:
		generate(cities, gen(), meta)
	except Exception as exc:
		print(exc)
	finally:
		open('meta.json', 'w').write(json.dumps(meta, indent=4))

if __name__ == '__main__':
	main()

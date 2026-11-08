"""City weather from fixed Open-Meteo endpoints; manual input works offline."""
import json,time
import requests
_cache={}

def _json(base,params):
    response=requests.get(base,params=params,timeout=(5,10))
    response.raise_for_status()
    if len(response.content)>256000:raise ValueError('Weather response exceeds the expected size')
    return response.json()

def city_weather(city):
    city=city.strip()
    if not 2<=len(city)<=100:raise ValueError('Enter a city name between 2 and 100 characters')
    cached=_cache.get(city.casefold())
    if cached and time.time()-cached[0]<300:return cached[1]
    found=_json('https://geocoding-api.open-meteo.com/v1/search',{'name':city,'count':1,'language':'en','format':'json'}).get('results',[])
    if not found:raise ValueError('City not found. Include a country, or enter weather manually.')
    place=found[0]
    data=_json('https://api.open-meteo.com/v1/forecast',{'latitude':place['latitude'],'longitude':place['longitude'],'current':'temperature_2m,relative_humidity_2m,precipitation,weather_code','timezone':'auto','forecast_days':1})
    current=data['current'];code=current['weather_code']
    condition='snow' if 71<=code<=77 or code in [85,86] else 'rain' if 51<=code<=67 or 80<=code<=82 or code>=95 else 'fog' if code in [45,48] else 'clear' if code==0 else 'cloudy'
    result={'condition':condition,'temperature_c':current['temperature_2m'],'humidity_percent':current['relative_humidity_2m'],'precipitation_mm':current['precipitation'],'city':', '.join(filter(None,[place['name'],place.get('admin1'),place.get('country')])),'observed_at':current['time'],'source':'open_meteo','source_url':'https://open-meteo.com/','timezone':data.get('timezone'),'attribution':'Weather data: Open-Meteo. Model-based city weather, not an on-device sensor.'}
    _cache[city.casefold()]=(time.time(),result)
    return result

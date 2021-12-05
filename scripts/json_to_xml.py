from json2xml import json2xml
from json2xml.utils import readfromurl, readfromstring, readfromjson

# get the xml from an URL that return json

data = readfromjson("cityscape_json/aachen_000109_000019_gtFine_polygons.json")
print(json2xml.Json2xml(data, item_wrap=False).to_xml())

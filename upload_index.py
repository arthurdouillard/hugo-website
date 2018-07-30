#!/usr/bin/env python3
import json

from algoliasearch import algoliasearch

with open('algolia_config.json') as f:
    j = json.load(f)

client = algoliasearch.Client(j['app_id'], j['admin_key'])
index = client.init_index('Posts')

jindex = json.load(open('public/search.json'))
index.add_objects(jindex)
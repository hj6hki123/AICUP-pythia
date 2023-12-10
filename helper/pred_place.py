import geonamescache
import spacy
from spacy.lang.en.examples import sentences 


gc = geonamescache.GeonamesCache()

# gets nested dictionary for countries
countries = gc.get_countries()

# gets nested dictionary for cities
cities = gc.get_cities()


def gen_dict_extract(var, key):
    if isinstance(var, dict):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, (dict, list)):
                yield from gen_dict_extract(v, key)
    elif isinstance(var, list):
        for d in var:
            yield from gen_dict_extract(d, key)

cities = [*gen_dict_extract(cities, 'name')]
countries = [*gen_dict_extract(countries, 'name')]

nlp = spacy.load("en_core_web_sm")

with open('AICUP/opendid_test/opendid_test.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        doc= nlp(line)
        for ent in doc.ents:
            if ent.label_ == 'GPE':
                if ent.text in countries:
                    print(f"Country : {ent.text}")
                elif ent.text in cities:
                    print(f"City : {ent.text}")
                else:
                    pass
        





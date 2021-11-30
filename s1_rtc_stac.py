#!/usr/bin/env python3
'''
Generate example STAC from test data
'''
import concurrent.futures
import s3fs
import sys
import pystac
import os
import yaml
from stactools.sentinel1.rtc.stac import create_item, create_collection

# Picks up all environment variables on github runner
#print(os.environ)
# e.g. 'AWS_REGION' 'GITHUB_REPOSITORY': 'relativeorbit/aws-rtc-stac'

fs = s3fs.S3FileSystem(anon=True)

with open('CONFIG.yml') as f:
    config = yaml.safe_load(f)
    MGRS = config['input']['MGRS']

def s3_to_http(s3path, region='us-west-2'):
    s3prefix = 'sentinel-s1-rtc-indigo'
    newprefix = f'https://sentinel-s1-rtc-indigo.s3.{region}.amazonaws.com'
    http = s3path.replace(s3prefix, newprefix)
    return http

def get_paths(zone=12, latLabel='S', square='YJ', year='*', date='*'):
    bucket = 'sentinel-s1-rtc-indigo'
    s3Path = f'{bucket}/tiles/RTC/1/IW/{zone}/{latLabel}/{square}/{year}/{date}/'
    print(f'searching {s3Path}...')
    keys = fs.glob(s3Path)
    print(f'{len(keys)} images matching {s3Path}')
    hrefs = [s3_to_http(x) for x in keys]
    return hrefs

def get_current_item_ids(catalog):
    items = [item.id for item in catalog.get_all_items()]
    return items

def save_item_collection(catalog):
    ''' consolidate all catalog items into a single JSON for stackstac
    https://github.com/gjoseph92/stackstac/discussions/86
    '''
    cat = pystac.read_file(catalog)
    itemcol = pystac.ItemCollection(cat.get_all_items())
    itemcol.save_object('mycollection.json')

if __name__ == '__main__':
    if not os.path.isfile('catalog.json'):
        catalog = pystac.Catalog(id='aws-rtc-stac',
                                 description='https://github.com/relativeorbit/aws-rtc-stac')
        collection = create_collection()
        catalog.add_child(collection)
    else:
        catalog = pystac.read_file('catalog.json')
        collection = catalog.get_child('sentinel1-rtc-aws')

    paths = get_paths(zone=MGRS[:2], latLabel=MGRS[2], square=MGRS[3:])
    current_items = get_current_item_ids(catalog)
    print(f'{len(current_items)} Items in STAC catalog')
    new_paths = [p for p in paths if not p[-22:] in current_items]
    print(f'Adding {len(new_paths)} new Items...')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        items = executor.map(create_item, new_paths)

    collection.add_items(items)
    catalog.generate_subcatalogs(template='${sentinel:mgrs}/${year}')
    catalog.normalize_hrefs('./')
    catalog.validate() #catalog.validate_all()
    catalog.save(catalog_type=pystac.CatalogType.RELATIVE_PUBLISHED)

    save_item_collection('catalog.json')

    #https://raw.githubusercontent.com/relativeorbit/aws-rtc-stac/main/catalog.json
    #ROOTURL = f'https://github.com/{os.environ["GITHUB_REPOSITORY"]}'
    #catalog.normalize_hrefs(ROOTURL)
    #catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED, dest_href='./tmp')

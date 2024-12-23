import os
from argparse import ArgumentParser
import requests
from tqdm import tqdm


_url_dict = {'burgers': 'https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/burgers_pino.mat'}

def download_file(url, file_path):
    print('Start downloading...')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=256 * 1024 * 1024)):
                f.write(chunk)
    print('Complete')


def main(args):
    url = _url_dict[args.name]
    file_name = url.split('/')[-1]
    os.makedirs(args.outdir, exist_ok=True)
    file_path = os.path.join(args.outdir, file_name)
    download_file(url, file_path)


if __name__ == '__main__':
    parser = ArgumentParser(description='Parser for downloading assets')
    parser.add_argument('--name', type=str, default='burgers')
    parser.add_argument('--outdir', type=str, default='./')
    args = parser.parse_args()
    main(args)
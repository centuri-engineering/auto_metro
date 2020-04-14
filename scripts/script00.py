import sys
import traceback
import random
from multiprocessing import Pool, Lock, Manager
from datetime import date
from getpass import getpass

import omero.clients
from omero.gateway import BlitzGateway

from auto_metro import batch, imageio, image_decorr


host = "localhost"
port = 4064
instrument_id = 10452
columns = [
    "Id",
    "AquisitionDate",
    "LensNA",
    "ChannelLabel",
    "PhysicalSizeX",
    "C",
    "Z",
    "T",
    "SNR",
    "resolution",
]


def target(lock, im_id, credentials):
    try:
        hf5_record = f"measures_{instrument_id}_{date.today().isoformat()}.hf5"
        conn = BlitzGateway(
            credentials["loggin"], credentials["password"], host=host, port=port
        )
        # Allow connection to all the images
        conn.SERVICE_OPTS.setOmeroGroup("-1")
        with imageio.OmeroImageReader(im_id, conn) as image_reader:
            data = batch.measure_process(
                lock, hf5_record, image_reader, image_decorr.measure, columns
            )
            return data
    except Exception as e:
        _, _, tb = sys.exc_info()
        print(f"Erro {e} for {im_id}")
        traceback.print_tb(tb)


def main():

    loggin = input("OME loggin:")
    password = getpass("OME password:")
    credentials = {"loggin": loggin, "password": password}

    manager = Manager()
    lock = manager.Lock()

    with BlitzGateway(loggin, password, host=host, port=port) as conn:
        conn.SERVICE_OPTS.setOmeroGroup("-1")
        print(instrument_id)
        all_images = [
            im.getId() for im in conn.getObjects("Image", opts={"instrument": 10455})
        ]
        # random.shuffle(all_images)
        all_images = all_images  # [:100]
        print(all_images)
        print(f"There are {len(all_images)} images to analyse")

    # pool = Pool(10)
    # results = pool.starmap_async(
    #     target, [(lock, im_id, credentials) for im_id in all_images]
    # )
    # results.get()


if __name__ == "__main__":
    main()

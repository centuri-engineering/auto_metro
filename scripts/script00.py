import sys
import traceback
import random
from multiprocessing import Pool, Lock, Manager
from datetime import date
from getpass import getpass
import pandas as pd
import omero
import omero.clients
from omero.gateway import BlitzGateway
from omero.rtypes import rlong
from auto_metro import batch, imageio, image_decorr
from omero_utils.images import get_images_from_instrument

host = "localhost"
port = 4064
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
        with imageio.OmeroImageReader(im_id, conn) as image_reader:
            data = batch.measure_process(
                lock, hf5_record, image_reader, image_decorr.measure, columns
            )
            return data
    except Exception as e:
        _, _, tb = sys.exc_info()
        print(f"Erro {e} for {im_id}")
        traceback.print_tb(tb)
        return pd.DataFrame(columns=columns)

def main(instrument_id):

    loggin = input("OME loggin:")
    password = getpass("OME password:")
    credentials = {"loggin": loggin, "password": password}

    manager = Manager()
    lock = manager.Lock()

    with BlitzGateway(loggin, password, host=host, port=port) as conn:
        conn.SERVICE_OPTS.setOmeroGroup("-1")
        print(instrument_id)
        #all_images = get_images_from_instrument(instrument_id, conn)
        all_images = [im.id for im in conn.getObjects("Image")]

        random.shuffle(all_images)
        all_images = all_images[:1000]
        print(f"There are {len(all_images)} images to analyse")

    pool = Pool(6)
    results = pool.starmap_async(
        target, [(lock, im_id, credentials) for im_id in all_images]
    )
    results.get()


if __name__ == "__main__":

    instrument_id = sys.argv[1]
    main(instrument_id)

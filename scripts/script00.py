import sys
import traceback
import random
from multiprocessing import Pool, Lock, Manager
from datetime import date
from getpass import getpass
import omero
import omero.clients
from omero.gateway import BlitzGateway
from omero.rtypes import rlong
from auto_metro import batch, imageio, image_decorr


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


def get_images_from_instrument(instrument_id, conn):
    """Returns a list of images ids
    """
    conn.SERVICE_OPTS.setOmeroGroup("-1")
    params = omero.sys.Parameters()
    params.map = {"instrument": rlong(instrument_id)}
    queryService = conn.getQueryService()
    images = queryService.projection(
        "select i.id from Image i where i.instrument.id=:instrument",
        params,
        conn.SERVICE_OPTS,
    )
    return [im[0].val for im in images]


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


def main(instrument_id):

    loggin = input("OME loggin:")
    password = getpass("OME password:")
    credentials = {"loggin": loggin, "password": password}

    manager = Manager()
    lock = manager.Lock()

    with BlitzGateway(loggin, password, host=host, port=port) as conn:
        conn.SERVICE_OPTS.setOmeroGroup("-1")
        print(instrument_id)
        all_images = get_images_from_instrument(instrument_id, conn)
        random.shuffle(all_images)
        # all_images = all_images  # [:100]
        print(f"There are {len(all_images)} images to analyse")

    pool = Pool(10)
    results = pool.starmap_async(
        target, [(lock, im_id, credentials) for im_id in all_images]
    )
    results.get()


if __name__ == "__main__":

    instrument_id = sys.argv[1]
    main(instrument_id)

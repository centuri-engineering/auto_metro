from multiprocessing import Pool, Lock
from datetime import date
from getpass import getpass

from auto_metro import batch, imageio, image_decorr

import omero.clients
from omero.gateway import BlitzGateway


def target(lock, im_id, conn):

    with imageio.OmeroImageReader(im_id, conn) as image_reader:
        data = batch.measure_process(
            lock, hf5_record, image_reader, image_decorr.measure, columns
        )
        return data


if __name__ == "__main__":

    host = "localhost"
    port = 4064

    loggin = input("OME loggin:")
    password = getpass("OME password:")

    instrument_id = 10457
    hf5_record = f"measures_{instrument_id}_{date.today().isoformat()}.hf5"
    columns = (
        [
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
        ],
    )
    lock = Lock()

    with BlitzGateway(loggin, password, host=host, port=port) as conn:
        all_images = [
            im.getId()
            for im in conn.getObjects("Image", opts={"instrument": instrument_id})
        ][:3]

    pool = Pool(10)
    pool.starmap_async(target, [(lock, im_id, conn.clone()) for im_id in all_images])

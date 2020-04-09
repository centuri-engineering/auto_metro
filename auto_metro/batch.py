import string
import logging
import os
from datetime import date

from itertools import product
import numpy as np
import pandas as pd

log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)

logdir = os.environ.get("MEASURE_LOG_DIRECTORY", ".")
logfile = f"measures_{date.today().isoformat()}.log"
hand = logging.FileHandler(logfile)
hand.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
hand.setFormatter(formatter)


def measure_single(image_reader, measure, columns, progress_bar=None, **kwargs):

    metadata = image_reader.get_metadata()
    size_z = metadata["SizeZ"]
    size_c = metadata["SizeC"]
    size_t = metadata["SizeT"]
    data = pd.DataFrame(
        np.empty((size_z * size_c * size_t, len(columns))), columns=columns
    )
    labels = metadata.get("ChannelLabels", string.ascii_uppercase[:size_c])
    for col in columns:
        if col in metadata:
            data[col] = metadata[col]

    if progress_bar is not None:
        progress_bar.max = size_z * size_c * size_t
        progress_bar.value = 0

    for i, ((c, z, t), plane) in enumerate(image_reader):
        if progress_bar is not None:
            progress_bar.description = f"Frame {i}/{size_z * size_c * size_t}"
            progress_bar.value = i + 1

        data.loc[i, ["C", "Z", "T"]] = c, z, t
        data.loc[i, "ChannelLabel"] = labels[c]
        m = measure(plane, metadata, **kwargs)
        for key in m:
            data.loc[i, key] = m[key]
    data["AquisitionDate"] = pd.to_datetime(data["AquisitionDate"])

    return data


def measure_process(lock, hf5_record, image_reader, measure, columns, **kwargs):

    try:
        data = measure_single(
            image_reader, measure, columns, progress_bar=None, **kwargs
        )
    except Exception as e:
        log.info(
            f"Error {type(e)}: {e} in measuring image {image_reader.id}"
            f" with {measure.__name__} from {measure.__module__}"
        )
        return
    try:
        lock.acquire()
        with pd.HDFStore(hf5_record, "a") as file:
            file.append(
                key=measure.__module__, value=data, data_columns=["AquisitionDate"]
            )
    finally:
        lock.release()
    return data

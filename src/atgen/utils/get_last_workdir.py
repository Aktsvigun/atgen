import os
import pathlib
from datetime import datetime


def get_last_workdir():
    cur_path = pathlib.Path() / "outputs"
    cur_path = cur_path / sorted(os.listdir(cur_path))[-1]
    cur_path = (
        cur_path
        / sorted(
            os.listdir(cur_path),
            key=lambda x: datetime.strptime(x.split("_")[-1], "%H-%M-%S"),
        )[-1]
    )
    return cur_path

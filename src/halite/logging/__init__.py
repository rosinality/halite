import functools

from halite.logging.logger import DistributedLogger, make_logger


@functools.lru_cache(maxsize=128)
def get_logger(
    mesh=None,
    name="main",
    mode="rich",
    abbrev_name=None,
    keywords=("INIT", "FROM"),
):
    if mesh is not None:
        return DistributedLogger(name, mesh, mode, abbrev_name, keywords)

    return make_logger(name, mode, abbrev_name, keywords)


logger = get_logger()

import sys

from loguru import logger


def setup_logger(rank: int, world_size: int):
    logger.remove()  # 移除默认 handler,否则会和下面的重复输出
    logger.add(
        sys.stderr,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>[Rank {extra[rank]}/{extra[world_size]}]</cyan> | "
            "<level>{message}</level>"
        ),
    )
    # 给全局 logger 设默认 extra,之后每条日志都自动带 rank
    logger.configure(extra={"rank": rank, "world_size": world_size})

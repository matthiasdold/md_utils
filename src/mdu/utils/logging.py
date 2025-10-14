import logging
import logging.config

# default_config = {
#     "version": 1,
#     "formatters": {
#         "colored": {
#             "()": "colorlog.ColoredFormatter",
#             "format": "%(log_color)s%(asctime)s - %(levelname)-8s - %(name)-10s:  %(reset)s %(white)s%(message)s",
#         }
#     },
#     "handlers": {
#         "console": {
#             "class": "logging.StreamHandler",
#             "formatter": "colored",
#         },
#     },
#     "root": {
#         "handlers": ["console"],
#     },
# }
#
# # overwriting defaults
# colors = {"DEBUG": "cyan"}
#
# logging.config.dictConfig(default_config)


# have this as a simple wrapper to ensure the updated config is used
def get_logger(
    name: str, log_level: int = logging.DEBUG, propagate: bool = True
) -> logging.Logger:
    logger = logging.getLogger(name)
    root_logger = logging.getLogger()

    # # change colors for all stream handlers
    # for hdl in root_logger.handlers:
    #     if isinstance(hdl, logging.StreamHandler):
    #         hdl.formatter.log_colors.update(colors)

    logger.propagate = propagate

    # if propagate, let the root logger handle the messages
    if (
        propagate and name != ""
    ):  # empty string will result in logger == root_logger -> nothing to remove
        logger.handlers = []
    # else add handlers to the named logger to handle itself
    else:
        for hdl in logger.handlers:
            logger.addHandler(hdl)

    logger.setLevel(log_level)

    return logger


if __name__ == "__main__":
    test_logger = get_logger("test")
    test_logger.setLevel(logging.DEBUG)
    test_logger.debug("DEBUG test")
    test_logger.info("INFO test")
    test_logger.warning("WARNING test")
    test_logger.error("ERROR test")
    test_logger.critical("CRITICAL test")


# -- colors need to be set via escape codes - here are the options
# escape_codes_foreground = {
#     "black": 30,
#     "red": 31,
#     "green": 32,
#     "yellow": 33,
#     "blue": 34,
#     "purple": 35,
#     "cyan": 36,
#     "white": 37,
#     "light_black": 90,
#     "light_red": 91,
#     "light_green": 92,
#     "light_yellow": 93,
#     "light_blue": 94,
#     "light_purple": 95,
#     "light_cyan": 96,
#     "light_white": 97,
# }
#
# escape_codes_background = {
#     "black": 40,
#     "red": 41,
#     "green": 42,
#     "yellow": 43,
#     "blue": 44,
#     "purple": 45,
#     "cyan": 46,
#     "white": 47,
#     "light_black": 100,
#     "light_red": 101,
#     "light_green": 102,
#     "light_yellow": 103,
#     "light_blue": 104,
#     "light_purple": 105,
#     "light_cyan": 106,
#     "light_white": 107,
#     # Bold background colors don't exist,
#     # but we used to provide these names.
#     "bold_black": 100,
#     "bold_red": 101,
#     "bold_green": 102,
#     "bold_yellow": 103,
#     "bold_blue": 104,
#     "bold_purple": 105,
#     "bold_cyan": 106,
#     "bold_white": 107,
# }

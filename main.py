import configparser
import sys
from ForexForcasting.data_loader import ForexLoader


def main(config):
    forex_loader = ForexLoader()
    data = forex_loader.load(
        source_type=config['general']['source_type'],
        path=config['general']['path']
    )
    print(data)




if __name__ == '__main__':
    config = configparser.ConfigParser()
    config_file_path = sys.argv[1]
    config.read(config_file_path)
    main(config)


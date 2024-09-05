import logging
import os

import pandas as pd

from src.processors import GameProcessor
from src.readers import PGNReader

SOURCE_DIRECTORY = 'data/raw'
OUTPUT_DIRECTORY = 'data/processed'

logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    source_files_names = os.listdir(SOURCE_DIRECTORY)
    files_to_process = [f'{SOURCE_DIRECTORY}/{file_name}' for file_name in source_files_names]

    for file_name in files_to_process:
        logging.info(f'Starting to read games from {file_name}')
        games = PGNReader(file_name=file_name).read_games()

        logging.info(f'Starting to process {len(games)} games')
        processed_games = GameProcessor().process_games(games)
        df = pd.DataFrame(processed_games)

        output_file_name = file_name.replace(SOURCE_DIRECTORY, OUTPUT_DIRECTORY).replace('pgn', 'csv')
        logging.info(f'Saving {len(df)} processed game states to file {output_file_name}')
        df.to_csv(output_file_name, index=False)

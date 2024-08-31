import re

EMPTY_SQUARE = 'o'


class FenProcessor:
    @staticmethod
    def get_color(fen: str) -> str:
        return fen.split(' ')[1]

    @staticmethod
    def get_board_string(fen: str) -> str:
        board_fen = fen.split(' ')[0]
        numbers = list(set(re.findall(r'\d+', board_fen)))
        replacements = [EMPTY_SQUARE * int(n) for n in numbers]
        processed_board = board_fen
        for number, replacement in zip(numbers, replacements):
            processed_board = processed_board.replace(number, replacement)
        return processed_board

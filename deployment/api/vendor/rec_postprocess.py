import re

import numpy as np
import paddle


class BaseRecLabelDecode:
    def __init__(
        self,
        character_dict_path: str | None = None,
        use_space_char: bool = False,
    ) -> None:
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str: list[str] = []

        if character_dict_path is None:
            self.character_str = list("0123456789abcdefghijklmnopqrstuvwxyz")
        else:
            with open(character_dict_path, "rb") as file:
                for raw_line in file.readlines():
                    line = raw_line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            if "arabic" in character_dict_path:
                self.reverse = True

        dictionary = self.add_special_char(list(self.character_str))
        self.character = dictionary
        self.dict = {char: index for index, char in enumerate(dictionary)}

    def pred_reverse(self, prediction: str) -> str:
        reversed_parts: list[str] = []
        current = ""
        for char in prediction:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-]", char)):
                if current:
                    reversed_parts.append(current)
                reversed_parts.append(char)
                current = ""
                continue
            current += char
        if current:
            reversed_parts.append(current)
        return "".join(reversed_parts[::-1])

    def add_special_char(self, dict_character: list[str]) -> list[str]:
        return dict_character

    def decode(
        self,
        text_index: np.ndarray,
        text_prob: np.ndarray | None = None,
        is_remove_duplicate: bool = False,
    ) -> list[tuple[str, float]]:
        result_list: list[tuple[str, float]] = []
        ignored_tokens = self.get_ignored_tokens()
        for batch_index, row in enumerate(text_index):
            selection = np.ones(len(row), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = row[1:] != row[:-1]
            for ignored_token in ignored_tokens:
                selection &= row != ignored_token

            char_list = [self.character[text_id] for text_id in row[selection]]
            confidence_list = text_prob[batch_index][selection] if text_prob is not None else [1]
            if len(confidence_list) == 0:
                confidence_list = [0]

            text = "".join(char_list)
            if self.reverse:
                text = self.pred_reverse(text)
            result_list.append((text, float(np.mean(confidence_list).tolist())))
        return result_list

    def get_ignored_tokens(self) -> list[int]:
        return [0]


class CTCLabelDecode(BaseRecLabelDecode):
    def __call__(self, preds: np.ndarray | paddle.Tensor) -> list[tuple[str, float]]:
        if isinstance(preds, (tuple, list)):
            preds = preds[-1]
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        return self.decode(preds_idx, preds_prob, is_remove_duplicate=True)

    def add_special_char(self, dict_character: list[str]) -> list[str]:
        return ["blank", *dict_character]

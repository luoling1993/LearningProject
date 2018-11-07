#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class IMM(object):
    # 逆向最大匹配

    def __init__(self, dict_path):
        self.dictionary = set()
        self.maximum = 0
        self._init(dict_path)

    def _init(self, dict_path):
        with open(dict_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                self.dictionary.add(line)
                length = len(line)
                if length > self.maximum:
                    self.maximum = length

    def cut(self, text):
        seg_cut = list()
        index = len(text)

        while index > 0:
            word = None

            for size in range(self.maximum, 0, -1):
                if index < size:
                    continue

                piece = text[(index - size):index]
                if piece in self.dictionary:
                    word = piece
                    seg_cut.append(word)
                    index -= size
                    break

            if word is None:
                index -= 1

        return seg_cut[::-1]


if __name__ == "__main__":
    string = "南京市长江大桥"
    tokenizer = IMM("./data/imm_dic.utf8")
    print(tokenizer.cut(string))

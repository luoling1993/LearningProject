#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pickle

# FIXME: The seg is worong


class HMM(object):
    # 隐马尔科夫算法

    def __init__(self, model_path=None):
        self.state_list = ['B', 'M', 'E', 'S']

        if model_path:
            self.model_path = model_path
            with open(model_path, 'rb') as f:
                self.a_dict = pickle.load(f)
                self.b_dict = pickle.load(f)
                self.pi_dict = pickle.load(f)
        else:
            self.model_path = "./data/hmm_model.pkl"
            self.a_dict = dict()
            self.b_dict = dict()
            self.pi_dict = dict()

    def train(self, corpus_path):
        counter_dict = dict()

        def init_parameters():
            for state in self.state_list:
                self.a_dict[state] = {s: 0.0 for s in self.state_list}
                self.pi_dict[state] = 0.0
                self.b_dict[state] = dict()
                counter_dict[state] = 0

        def make_label(text):
            out_text = list()

            if len(text) == 1:
                out_text.append('S')
            else:
                out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']
            return out_text

        init_parameters()
        line_num = -1

        words_set = set()
        with open(corpus_path, encoding='utf-8') as f:
            for line in f:
                line_num += 1

                line = line.strip()
                if not line:
                    continue

                word_list = [line_item for line_item in line if line_item != ' ']
                words_set |= set(word_list)

                line_list = line.split()

                line_state = list()
                for w in line_list:
                    line_state.extend(make_label(w))

                assert len(word_list) == len(line_state)

                for k, v in enumerate(line_state):
                    counter_dict[v] += 1
                    if k == 0:
                        self.pi_dict[v] += 1
                    else:
                        self.a_dict[line_state[k - 1]][v] += 1
                        item_value = self.b_dict[line_state[k]].get(word_list[k], 0) + 1.0
                        self.b_dict[line_state[k]][word_list[k]] = item_value

        for k, v in self.pi_dict.items():
            self.pi_dict[k] = v * 1.0 / line_num

        for k, v in self.a_dict.items():
            item_dict = dict()
            for k1, v1 in v.items():
                item_dict[k1] = v1 / counter_dict[k]
            self.a_dict[k] = item_dict

        for k, v in self.b_dict.items():
            item_dict = dict()
            for k1, v1 in v.items():
                item_dict[k1] = (v1 + 1) / counter_dict[k]
            self.b_dict[k] = item_dict

        with open(self.model_path, 'wb') as f:
            pickle.dump(self.a_dict, f)
            pickle.dump(self.b_dict, f)
            pickle.dump(self.pi_dict, f)

        return self

    @staticmethod
    def viterbi(text, states, start_p, trans_p, emit_p):
        v_dict = [dict()]

        path = dict()
        for y in states:
            v_dict[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]

        emit_p_list = list()
        emit_p_list.extend(list(emit_p['S'].keys()))
        emit_p_list.extend(list(emit_p['M'].keys()))
        emit_p_list.extend(list(emit_p['E'].keys()))
        emit_p_list.extend(list(emit_p['B'].keys()))
        emit_p_set = set(emit_p_list)

        for t in range(1, len(text)):
            v_dict.append(dict())
            new_path = dict()

            if text[t] in emit_p_set:
                never_seen = True
            else:
                never_seen = False

            for state1 in states:
                if not never_seen:
                    _emit_p = emit_p[state1].get(text[t], 0.0)
                else:
                    _emit_p = 1.0

                for state2 in states:
                    max_prob = 0.0
                    max_state = states[0]
                    if v_dict[t - 1][state2] > 0:
                        prob = v_dict[t - 1][state2] * trans_p[state2].get(state1, 0) * _emit_p
                        if prob > max_prob:
                            max_prob = prob
                            max_state = state2

                    v_dict[t][state1] = max_prob
                    new_path[state1] = path[max_state] + [state1]

            path = new_path

        emit_p_m = emit_p['M'].get(text[-1], 0)
        emit_p_s = emit_p['S'].get(text[-1], 0)
        if emit_p_m > emit_p_s:
            max_prob, max_state = max([(v_dict[len(text) - 1][y], y) for y in ['E', 'M']])
        else:
            max_prob, max_state = max([(v_dict[len(text) - 1][y], y) for y in states])

        return max_prob, path[max_state]

    def cut(self, text):
        prob, pos_list = self.viterbi(text, self.state_list, self.pi_dict, self.a_dict, self.b_dict)

        begin, next_ = 0, 0

        for idx, word in enumerate(text):
            pos = pos_list[idx]

            if pos == 'B':
                begin = idx
            elif pos == 'E':
                yield text[begin: idx + 1]
                next_ = idx + 1
            elif pos == 'S':
                yield word
                next_ = idx + 1

        if next_ < len(text):
            yield text[next_:]


if __name__ == "__main__":
    hmm = HMM()
    hmm.train("./data/trainCorpus.txt_utf8")

    string = '这是一个非常棒的方案！'
    seg = hmm.cut(string)
    print(list(seg))

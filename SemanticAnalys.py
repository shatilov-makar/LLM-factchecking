#!python3_10/Scripts/python
from allennlp.predictors.predictor import Predictor
import nltk


class SemanticAnalys:
    def __init__(self):
        nltk.download('punkt')
        SRL_MODEL_PATH = "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
        self.predictor = Predictor.from_path(SRL_MODEL_PATH)
        self.punct = set(['.', '!', '?', '/n'])
        self.stop_tags = set(['R-ARG',
                              'ARGM-PRP',
                              'ARGM-PRD'
                              ])

    def __get_framegroups(self, sentence):
        groupes = self.predictor.predict(sentence)
        words = groupes['words']
        res = []
        for i, group in enumerate(groupes['verbs']):
            frames = {}
            for word, tag in zip(words, group['tags']):
                if tag != 'O':
                    tag_name = tag[2::]
                    if tag_name in frames:
                        frames[tag_name].append(word)
                    else:
                        frames[tag_name] = [word]
            temp_res = []
            for tag_name in frames.keys():
                temp_res.append({
                    'group_id': i,
                    'tag': tag_name,
                    'text': ' '.join(frames[tag_name])
                })
            res.append(temp_res)
        return res

    def __framegroup_is_broken(self, framegroup):
        return len(framegroup) < 3

    def __get_embedded_framegroups(self, frame, framegroups):
        embedded_framegroups = []
        for framegroup in framegroups:
            if self.__framegroup_is_broken(framegroup):
                continue
            else:
                framegroup_text = [frame['text'] for frame in framegroup]
            if all([word in frame['text'] for word in framegroup_text]):
                embedded_framegroups.append(framegroup)
            return embedded_framegroups
        return embedded_framegroups

    def __framegroup_is_embedded(self, framegroup,  sentences):
        for sentence in sentences:
            if all([word['text'] in sentence for word in framegroup]):
                return True
        return False

    def __update_current_sentence(self, current_sentence, new_text):
        current_sentence['text'].append(new_text)
        current_sentence['status'] = 0

    def __add_current_sentence_to_result(self, global_sentences, current_sentence, need_to_check=False, frame=None):
        def func(f): return any([tag in f['tag'] for tag in self.stop_tags])
        new_sentence = ' '.join(current_sentence['text'])

        in_use = any(
            [new_sentence in sentence for sentence in global_sentences])
        if not in_use and ((need_to_check and func(frame)) or need_to_check == False):
            global_sentences.append(new_sentence)
            current_sentence['status'] = 1

    def __get_segmented_sentence(self, framegroups):
        global_sentences = []
        for framegroup in framegroups:
            current_sentence = {'text': [], 'status': 0}
            if self.__framegroup_is_broken(framegroup) or self.__framegroup_is_embedded(framegroup, global_sentences):
                continue
            for frame in framegroup:
                self.__add_current_sentence_to_result(
                    global_sentences, current_sentence, True, frame=frame)
                embedded_framegroups = []
                embedded_framegroups = self.__get_embedded_framegroups(
                    frame, framegroups[framegroups.index(framegroup) + 1:])  # ,framegroups[i + 1])
                if len(embedded_framegroups) == 0:
                    self.__update_current_sentence(
                        current_sentence, frame['text'])
                else:
                    for embedded_framegroup in embedded_framegroups:
                        for emb_frame in embedded_framegroup:
                            if emb_frame['text'] not in ' '.join(current_sentence['text']):
                                self.__add_current_sentence_to_result(
                                    global_sentences, current_sentence, True, frame=emb_frame)

                                self.__update_current_sentence(
                                    current_sentence, emb_frame['text'])

                        self.__add_current_sentence_to_result(
                            global_sentences, current_sentence, False)

            if current_sentence['status'] == 0:
                self.__add_current_sentence_to_result(
                    global_sentences, current_sentence, False)
        return global_sentences

    def semantic_analys(self, clean_sentences):
        args_dict = {}
        analized_sentences = []
        for i, sentence in enumerate(clean_sentences):
            framegroups = self.__get_framegroups(sentence)
            sentences = self.__get_segmented_sentence(framegroups)

            for framegroup in framegroups:
                for frame in framegroup:
                    if 'ARG' in frame['tag']:
                        arg = frame['text']
                        if arg in args_dict:
                            args_dict[arg] += 1
                        else:
                            args_dict[arg] = 1
            sem_analys = {
                'sentence_id': i,
                'sentence': sentence,
                'segmented_sentences': sentences
            }
            analized_sentences.append(sem_analys)

        if len(args_dict) > 0:
            topic = max(args_dict, key=args_dict.get)
        else:
            topic = None
        return analized_sentences, topic

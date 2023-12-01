#!python3_10/bin/python
import json
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from llamaapi import LlamaAPI
from SemanticAnalys import SemanticAnalys
import wikipediaapi
import requests
import torch
from torch import nn

CONTRADICTION = 'CONTRADICTION'
NEUTRAL = 'NEUTRAL'
ENTAILMENT = 'ENTAILMENT'
SUSPICIOUS = 'SUSPICIOUS'


class Factchecker:
    def __init__(self) -> None:

        self.bert = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2')
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")

        self.roberta = AutoModelForSequenceClassification.from_pretrained(
            "roberta-large-mnli")

        self.semantic_analys = SemanticAnalys()
        self.punct = set(['.', '!', '?', '/n'])

    # def __get_llm_output(self, prompt: str) -> str:
    #     '''
    #         Returns LLAMA-2 output for the given prompt.
    #     '''
    #     llama = LlamaAPI(self.LLAMA_API_TOKEN)
    #     api_request_json = {
    #         "temperature": 0.1,
    #         "messages": [
    #             {"role": "user", "content": prompt},
    #         ],
    #         "stream": False,
    #     }
    #     response = llama.run(api_request_json)
    #     return json.dumps(response.json()['choices'][0]['message']['content'], indent=2)

    def __coref_resolution(self, text: str) -> str:
        '''
            Solves a problem coreference resolution for a given text.
        '''
        headers = {
            'Content-Type': 'application/json',
        }

        json_data = {
            'input': text,
            'parameters': {
                'data1': None,
                'data2': None
            }
        }
        response = requests.post(
            'http://127.0.0.1:8001/coref_resolution', headers=headers, json=json_data)
        ans = json.loads(response.text)
        return ans['coref_resolution'].strip()

    def __tokenize_sentences(self, text: list) -> list:
        '''
            Performs text preprocessing and sentence-piece tokenization 
        '''
        text = self.__coref_resolution(text)
        print(text)
        with open('intros.txt') as f:
            intros = set([line.rstrip() for line in f])

        # Remove 'However', 'Besides', 'Hence' and other intro-expressions.
        for intro in intros:
            if intro in text:
                text = text.replace(intro, '')
        sentences = []
        if text[-1] in self.punct:  # Remove incomplited sentences
            sentences = [t.strip() for t in sent_tokenize(text)]
        else:
            sentences = [t.strip() for t in sent_tokenize(text)[:-1]]
        sa = self.semantic_analys.semantic_analys(sentences)
        return sa

    def __run_wiki_search(self, topic) -> list:
        print(topic)
        wiki_wiki = wikipediaapi.Wikipedia(
            'Factcheckung_LLM (m.point2011@yandex.ru)', 'en')
        page = wiki_wiki.page(topic)
        refs = []

        if page.exists():
            wiki_sentences = []
            # text = self.__coref_resolution(page.text)
            for p in page.text.split('\n'):
                if p != '':
                    if p[-1] not in self.punct:
                        p += '.'
                    wiki_sentences += sent_tokenize(p)
            refs = [{
                "sentence": sentence,
                "embedding": None
            } for sentence in wiki_sentences]
        return refs

    def __get_entailment_score(self, text, hypothesis):
        if hypothesis[-1] in self.punct:
            sequence_to_classify = ' '.join([hypothesis, text])
        else:
            sequence_to_classify = '. '.join([hypothesis, text])

        inputs = self.tokenizer(sequence_to_classify,
                                return_tensors="pt")
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            logits = self.roberta(**inputs).logits
            res = {}
            for label, score in zip(self.roberta.config.id2label.values(), softmax(logits).numpy()[0]):
                res[label] = score
            return res

    def __get_judgment(self, segment, ref):
        score = self.__get_entailment_score(segment, ref)
        judgment = max(score, key=score.get)
        if judgment in [ENTAILMENT, CONTRADICTION]:
            return judgment
        if score[ENTAILMENT] > score[CONTRADICTION] and score[ENTAILMENT] > 0.10:
            return ENTAILMENT
        return NEUTRAL

    def __get_relevant_refs(self, refs: list, source: str, threshold: float = 0.7):
        source_embedding = self.bert.encode(
            source, convert_to_tensor=True)
        scores = []
        for sentence in refs:
            if sentence['embedding'] is None:
                sentence['embedding'] = self.bert.encode(
                    sentence['sentence'], convert_to_tensor=True)

            score = float(util.pytorch_cos_sim(
                source_embedding, sentence['embedding'])[0][0])
            scores.append({
                'ref': sentence['sentence'],
                'embedding': sentence['embedding'],
                'score': score
            })
            if score > 0.85:
                return [scores[-1]]
        candidates = sorted(list(filter(lambda r: r['score'] > threshold, scores)),
                            key=lambda x: x['score'], reverse=True)[:5]
        return candidates

    def start_factchecking(self, output, threshold):
        '''
            Performs factchecking for a given LLM output
        '''
        llm_output_tokenized, topic = self.__tokenize_sentences(output)
        refs = self.__run_wiki_search(topic)
        fact_checking_results = []
        for sentence in llm_output_tokenized:
            wait_list = []
            sentence_fc = []
            for segment in sentence['segmented_sentences']:
                relevant_refs = self.__get_relevant_refs(
                    refs, segment, threshold)
                relevant_refs_count = len(relevant_refs)
                if relevant_refs_count > 0:  # Есть референсы?
                    if relevant_refs_count > 1:  # Есть несколько референсов?
                        refs_judgment = [self.__get_judgment(
                            segment, ref['ref']) for ref in relevant_refs]
                        if any([s == ENTAILMENT for s in refs_judgment]) and \
                                all([s != CONTRADICTION for s in refs_judgment]):
                            sentence_fc.append({
                                'sentence_id': sentence['sentence_id'],
                                "segment": segment,
                                "judgment": ENTAILMENT,
                                "refs_judgment": refs_judgment,
                                'relevant_refs': [r['ref'] for r in relevant_refs]
                            })
                        elif any([s == CONTRADICTION for s in refs_judgment]):
                            sentence_fc.append({
                                'sentence_id': sentence['sentence_id'],
                                "segment": segment,
                                "judgment": CONTRADICTION,
                                "refs_judgment": refs_judgment,
                                'relevant_refs': [r['ref'] for r in relevant_refs]

                            })
                        else:
                            sentence_fc.append({
                                'sentence_id': sentence['sentence_id'],
                                "segment": segment,
                                "judgment": NEUTRAL,
                                "refs_judgment": refs_judgment,
                                'relevant_refs': [r['ref'] for r in relevant_refs]
                            })

                    else:
                        judgment = self.__get_judgment(
                            segment, relevant_refs[0]['ref'])
                        sentence_fc.append({
                            'sentence_id': sentence['sentence_id'],
                            "segment": segment,
                            "judgment": judgment,
                            'relevant_refs': [r['ref'] for r in relevant_refs]
                        })

                # Предложение разбито на несколько?
                elif len(sentence['segmented_sentences']) > 1:
                    wait_list.append(segment)
                else:
                    sentence_fc.append({
                        'sentence_id': sentence['sentence_id'],
                        "segment": segment,
                        "judgment": SUSPICIOUS
                    })

            for segment in wait_list:
                if any([s["judgment"] == ENTAILMENT for s in sentence_fc]) and \
                        all([s["judgment"] != CONTRADICTION for s in sentence_fc]):

                    sentence_fc.append({
                        'sentence_id': sentence['sentence_id'],
                        "segment": segment,
                        "judgment": NEUTRAL
                    })
                else:
                    sentence_fc.append({
                        'sentence_id': sentence['sentence_id'],
                        "segment": segment,
                        "judgment": SUSPICIOUS
                    })
            fact_checking_results += sentence_fc.copy()
        return fact_checking_results

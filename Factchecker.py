#!python3_10/bin/python
import json
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from llamaapi import LlamaAPI
from SemanticAnalys import SemanticAnalys
import wikipedia
import requests
import torch
from torch import nn
import spacy
import re

CONTRADICTION = 'CONTRADICTION'
NEUTRAL = 'NEUTRAL'
ENTAILMENT = 'ENTAILMENT'
SUSPICIOUS = 'SUSPICIOUS'


class Factchecker:
    def __init__(self, LLAMA_API_TOKEN) -> None:
        self.LLAMA_API_TOKEN = LLAMA_API_TOKEN
        self.bert = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        self.roberta = AutoModelForSequenceClassification.from_pretrained(
            "roberta-large-mnli").to(self.device)

        self.semantic_analys = SemanticAnalys()
        self.punct = set(['.', '!', '?', '/n'])
        self.nlp = spacy.load('en_core_web_sm')

    def __del__(self):
        torch.cuda.empty_cache()

    def __get_llm_output(self, prompt: str) -> str:
        '''
            Returns LLAMA-2 output for the given prompt.
        '''
        llama = LlamaAPI(self.LLAMA_API_TOKEN)
        api_request_json = {
            "temperature": 0.1,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        response = llama.run(api_request_json)
        return json.dumps(response.json()['choices'][0]['message']['content'], indent=2)

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
        if response.status_code == 200:
            ans = json.loads(response.text)
            return ans['coref_resolution'].strip()
        return text

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

    def __run_wiki_search(self, topic: str) -> list:
        '''
            Returns a Wikipedia article by given title
        '''
        page = wikipedia.page(topic, auto_suggest=False)
        sentences = []
        if page:
            doc = self.nlp(page.content)
            for sent in doc.sents:
                if '==' not in sent.text:
                    sentences.append(sent.text.strip())
            text = ' '.join(sentences)
            clean_text = self.__coref_resolution(text)
            doc = self.nlp(clean_text)
            clean_sentences = []
            for sent in doc.sents:
                clean_sentences.append(sent.text.strip())
        return clean_sentences

    def __get_entailment_score(self, hypothesis: str, refs: str) -> list:
        ''' 
            Given a premise sentence and a hypothesis sentence, returns a 
            state depending on whether the premise entails the hypothesis (ENTAILMENT), 
            contradicts the hypothesis (CONTRADICTION), or neither (NEUTRAL).
        '''
        refs_and_hypothesis = []
        for text in refs:
            if hypothesis[-1] in self.punct:
                sequence_to_classify = ' '.join([text, hypothesis])
            else:
                sequence_to_classify = '. '.join([text, hypothesis])
            refs_and_hypothesis.append(sequence_to_classify)

        inputs = self.tokenizer.batch_encode_plus(refs_and_hypothesis,
                                                  padding=True, truncation=True,
                                                  return_tensors="pt").to(self.device)
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            logits = self.roberta(**inputs).logits
            scores = []
            for score in softmax(logits):
                ref_score = {}
                for label, score in zip(self.roberta.config.id2label.values(), score):
                    ref_score[label] = score
                scores.append(ref_score)
            return scores

    def __get_roberta_judgments(self, segment: str, refs: list) -> list:
        '''
        Returns a textual entailment score for the given segment with each reference 
        from given list of references
        '''
        scores = self.__get_entailment_score(segment, refs)
        judgments = []
        for score in scores:

            judgment = max(score, key=score.get)
            if judgment in [ENTAILMENT, CONTRADICTION]:
                judgments.append(judgment)
            elif score[ENTAILMENT] > score[CONTRADICTION] and score[ENTAILMENT] > 0.10:
                judgments.append(ENTAILMENT)
            else:
                judgments.append(NEUTRAL)
        return judgments

    def __fix_segment(self, segment: str, fact: str) -> str:
        '''
        Fix segment based on fact using LLM
        '''

        prompt = f'Source = {segment} \
                    Fact = {fact} \
                    New_Source = Source modified to reflect Fact.\
                    Do not type explanation, comments and score.\
                    Start type with: "New_Source: ...'
        llm_output = self.__get_llm_output(prompt)
        llm_output = llm_output.strip('\n. "').split('New_Source:')[-1].strip()
        if '"' in llm_output:
            llm_output = llm_output.split('"')[1]
        return llm_output

    def __get_llm_judgments(self, segment: str, refs: list) -> list:
        '''
        Returns truthfulness of segment based on list of references 
        '''
        facts = '\n'.join(
            [f'Fact_{i+1}: {reference}' for i, reference in enumerate(refs)])
        facts_enum = ' and '.join([f'Fact_{i+1}' for i in range(len(refs))])
        how_to_print = '\n'.join(
            [f'Fact_{i+1}: corresponds/contradicts/neutral' for i in range(len(refs))])
        prompt = f'''
            Source_1: {segment}
            {facts}
            Determine whether the information from Source_1 contradict, correspond or are neutral to the {facts_enum}. 
            Do not pay attention if the {facts_enum} contain information that is not in the Source_1.
            Define a fact as "correspond" only if the fact matches the information in Source_1 as closely as possible.
            If a Fact contains information that is not directly related to the Source_1, define the fact as "neutral".
            Type:"{how_to_print}"
            Do not type explanation.
        '''
        output = self.__get_llm_output(prompt)
        labels = [('correspond', ENTAILMENT), ('neutral', NEUTRAL),
                  ('contradict', CONTRADICTION)]
        judgments = []
        for i in range(len(refs)):
            fact_number = i + 1
            reg_expr = f'(Fact_{fact_number}.{{1,4}}correspond|Fact_{fact_number}.{{1,4}}neutral|Fact_{fact_number}.{{1,4}}contradict)'
            reg_substr = re.search(reg_expr, output)
            if reg_substr:
                for label in labels:
                    if label[0] in reg_substr[0]:
                        judgments.append(label[1])
            else:
                judgments.append(NEUTRAL)

        return judgments

    def __get_relevant_refs(self, refs: list, source: str, references_count: int = 3, threshold: float = 0.7):
        '''
            Filter sentences from Wikipedia articles, saves only relevant sentences.
        '''
        if (self.wiki_embeddings is None):
            self.wiki_embeddings = self.bert.encode(
                refs, convert_to_tensor=True)

        source_embedding = self.bert.encode(source, convert_to_tensor=True)
        distances = [(i, util.pytorch_cos_sim(source_embedding, embedding)[0][0])
                     for i, embedding in enumerate(self.wiki_embeddings)]
        best_references_scores = filter(lambda reference: reference[1] > threshold,
                                        sorted(distances, key=lambda e: e[1], reverse=True)[:references_count])

        best_references = [refs[candidate[0]]
                           for candidate in list(best_references_scores)]
        return best_references

    def start_factchecking(self, output, references_count, threshold):
        '''
            Performs factchecking for a given LLM output
        '''
        self.wiki_embeddings = None
        llm_output_tokenized, topic = self.__tokenize_sentences(output)
        refs = self.__run_wiki_search(topic)
        fact_checking_results = []
        reliable_sentences = []
        for sentence in llm_output_tokenized:
            wait_list = []
            sentence_fc = []
            for segment in sentence['segmented_sentences']:
                relevant_refs = self.__get_relevant_refs(
                    refs, segment, references_count, threshold)
                relevant_refs_count = len(relevant_refs)
                if relevant_refs_count > 0:  # Есть референсы?

                    roberta_judgments = self.__get_roberta_judgments(
                        segment, relevant_refs)
                    llm_judgments = self.__get_llm_judgments(
                        segment, relevant_refs)
                    fact_index, judgment = next(((i, x) for i, (x, y) in enumerate(zip(llm_judgments, roberta_judgments)) if (
                        x in [ENTAILMENT, CONTRADICTION]) and x == y), (NEUTRAL, -1))
                    if judgment == ENTAILMENT:
                        reliable_sentences.append(segment)
                    elif judgment == CONTRADICTION:
                        fixed_sentence = self.__fix_segment(
                            segment, relevant_refs[fact_index])
                        reliable_sentences.append(fixed_sentence)
                    sentence_fc.append({
                        'sentence_id': sentence['sentence_id'],
                        "segment": segment,
                        "judgment": judgment,
                        "refs_judgment": [roberta_judgments, llm_judgments],
                        'relevant_refs': relevant_refs
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
        torch.cuda.empty_cache()
        if len(reliable_sentences) > 0:
            reliable_sentences = ",".join(
                ['"' + sentence + '"' for sentence in reliable_sentences])
            prompt = f'Write a text using following sentences: {reliable_sentences}\n\
                        Do not use any additional information.  \
                        Do not type explanation and comments. \
                        Start with "Text: ..."'
            llm_output = self.__get_llm_output(prompt)
            return fact_checking_results, llm_output
        return llm_output

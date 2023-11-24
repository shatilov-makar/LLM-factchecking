#!factchecking_v1/python3_10/bin/python
import json
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from serpapi import GoogleSearch
from llamaapi import LlamaAPI
import subprocess
nltk.download('punkt')
##nltk.download('wordnet')
##nltk.download('omw-1.4')


class Factchecker:
    def __init__(self, LLAMA_API_TOKEN, SERP_API_TOKEN) -> None:
        self.LLAMA_API_TOKEN = LLAMA_API_TOKEN
        self.SERP_API_TOKEN = SERP_API_TOKEN
        self.bert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

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
        subprocess.run(['python3_7/bin/python', 'coref_resolution.py', text])
        with open('coref_resolution.txt') as f:
            lines = f.readlines()
        return ' '.join([line.strip() for line in lines])

    def __tokenize_sentences(self, text: str) -> list:
        '''
            Performs text preprocessing and sentence-piece tokenization 
        '''
        text = self.__coref_resolution(text)
        with open('intros.txt') as f:
            intros = set([line.rstrip() for line in f])
        punct = set(['.', '!', '?', '/n'])

        for intro in intros:# Remove 'However', 'Besides', 'Hence' and other intro-expressions.
            if intro in text:
                text = text.replace(intro, '')
        sentences = []
        if text[-1] in punct:# Remove incomplited sentences
            sentences = [t.strip() for t in sent_tokenize(text)]
        else:
            sentences = [t.strip() for t in sent_tokenize(text)[:-1]]
        return sentences

    def __run_google_search(self, query: str) -> list:
        '''
            Returns the Google SERP response to a query.
        '''
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.SERP_API_TOKEN,
            'num': 20
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return results["organic_results"]

    def __get_sentence_similarity(self, sentence_1: str, sentence_2: str) -> float:
        '''
            Returns the semantic similarity of two sentences, calculated using 
            the cosine distance between sentence embeddings.
        '''
        embedding_source = self.bert.encode(sentence_1, convert_to_tensor=True)
        embedding_ref = self.bert.encode(sentence_2, convert_to_tensor=True)
        score = float(util.pytorch_cos_sim(embedding_source, embedding_ref)[0][0])
        return score

    def __get_relevant_refs(self, refs: list, source: list):
        '''
            Filter Google's response, saves only relevant references.
        '''
        all_sentences = []
        for ref in refs:
            init_sentence = ref['snippet']
            sentences = sent_tokenize(init_sentence)
            all_sentences += sentences
        scores = []
        for ref in all_sentences:
            score = self.__get_sentence_similarity(source, ref)
            scores.append({
                'ref': ref,
                'score': score
            })

        candidates = sorted(list(filter(lambda r: r['score'] > 0.725, scores)),
                            key=lambda x: x['score'], reverse=True)
        if len(candidates) > 1:
            for i, c in enumerate(candidates):
                current_refs = candidates[i+1:]
                for ref in current_refs:
                    score = self.__get_sentence_similarity(c['ref'], ref['ref'])
                    if score > 0.45:
                        candidates.remove(ref)
        elif len(candidates) == 0:
            return None
        return candidates
    
    def start_factchecking(self, output):
        '''
            Performs factchecking for a given LLM output
        '''
        llm_output_tokenized = self.__tokenize_sentences(output)
        reliable_sentences = []
        for sentence in llm_output_tokenized:
            google_outputs = self.__run_google_search(sentence)
            relevant_google_outputs = self.__get_relevant_refs(google_outputs, sentence)
            if relevant_google_outputs is None:
                continue
            prompt = f'''Source: "{sentence}" '''
            for i, fact in enumerate(relevant_google_outputs):
                prompt += f'Fact_{i+1}: "{fact}\n"'
            facts_name = "Fact_1"
            if len(relevant_google_outputs) > 1:
                for i, fact in enumerate(relevant_google_outputs[1:]):
                    facts_name += f'and Fact_{i + 2}'
            prompt += f'New_Source = Source modified to reflect {facts_name}.\
                            If the Source approximately reflects the {facts_name},\
                            New_Source matches exactly Source.\
                            Do not type explanation, comments and score.\
                            Start type with: "New_Source: ...'

            llm_output = self.__get_llm_output(prompt)
            llm_output = llm_output.strip('\n. "').split('New_Source:')[-1].strip()
            if '"' in llm_output:
                llm_output = llm_output.split('"')[1]
            score = self.__get_sentence_similarity(sentence, llm_output)
            if score < 0.60:
                continue
            elif score < 0.90:
                reliable_sentences.append(llm_output)
            else:
                reliable_sentences.append(sentence)
        if len(reliable_sentences) == 0:
            return "The generated text was completely incorrect and cannot be corrected!"
        reliable_sentences = ",".join(['"' + sentence + '"' for sentence in reliable_sentences])
        prompt = f'Write a text using following sentences: {reliable_sentences}\n\
                    Do not use any additional information.  \
                    Do not type explanation and comments. \
                    Start with "Text: ..."'
        llm_output = self.__get_llm_output(prompt)
        return llm_output.split('Text:')[-1].strip()
    
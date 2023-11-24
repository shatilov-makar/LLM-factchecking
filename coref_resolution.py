#!factchecking_v1/python3_7/bin/python
import spacy
import neuralcoref
import sys

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)


def coref_resolution(text):
    doc = nlp(text)
    return(doc._.coref_resolved)


if __name__ == "__main__":
    llm_output = sys.argv[1]
    result = coref_resolution(llm_output)
    with open('coref_resolution.txt', 'w') as f:
        f.write(result)

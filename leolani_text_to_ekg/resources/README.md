This folder should contain the fine-tuned LLMs for the conversational Agent.

You can download the LLMs from:
   1. [BERT-go-emotion](https://vu.data.surfsara.nl/index.php/s/VjHn1AHgizlzov6)
   2. [Multilingual-BERT-Conversational-triple-extraction](https://vu.data.surfsara.nl/index.php/s/xL9fPrqIq8bs6NH)
   3. [XLM-RoBERTa-DialogueAct-classification](https://vu.data.surfsara.nl/index.php/s/dw0YCJAVFM870DT)

The link will get you to the university server. Pushing the Download button (at the top-right corner) will save the files as a ```tar.zip``` file.
Unpack the files in the resources folder. 

Note that the Multilingual-BERT-Conversational-triple-extraction is unpacked in a folder named ```24_03_11```.
Rename this folder into ```conversational_triples```.

After donwloading, unpacking and renaming, the resources folder should contain:

- bert-base-go-emotion
- conversational_triples
- midas-da-xlmroberta

The docker-compose.yml file should have a mapping to the local resources folder:

      - ./resources:/leolani-text-to-ekg/app/py-app/resources


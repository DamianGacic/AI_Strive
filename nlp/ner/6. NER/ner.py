import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training import Example
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# TBD: Load preferred model
nlp = spacy.load("en_core_web_sm")

with open("fashion brands.txt") as file:
    dataset = file.read()

doc = nlp(dataset)
#print("entities:",[(ent.text,ent.label_) for ent in doc.ents])
print([ent.text for ent in doc.ents])
words = ["Gucci","Schiaparelli","Chanel","Prada","Dolce & Gabbana ","Armani","Versace","Saint Laurent","Burberry","H&M","Alexander McQueen","Calvin Klein","Louis Vuitton"]

train_data = []
with open("fashion brands.txt") as file:
     dataset = [line.strip() for line in file if line.strip()]

     for sentence in dataset:
         print("######")
         print("sentence: ", sentence)
         print("######")
         sentence = sentence.lower()
         entities = []
         for word in words:
             word = word.lower()
             if word in sentence:
                 start_index = sentence.index(word)
                 end_index = len(word) + start_index
                 print("word: ", word)
                 print("----------------")
                 print("start index:", start_index)
                 print("end index:", end_index)
                 pos = (start_index, end_index, "fashion_brand")
                 entities.append(pos)
         element = (sentence.rstrip('\n'), {"entities": entities})

         train_data.append(element)
         print('----------------')
         print("element:", element)

# STEP 2 - UPDATE MODEL
ner = nlp.get_pipe("ner")

for _, annotations in train_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])
# TBD: load the needed pipeline
ner = nlp.get_pipe("ner")
# TBD: define the annotations
for _, annotations in train_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])
# TBD: train the model

pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]


# TBD: define the number of iterations, the batch size and the drop according to your experience or using an empirical value
# Train model
with nlp.disable_pipes(*unaffected_pipes):
    for iteration in range(10):
        print("Iteration #" + str(iteration))

        # Data shuffle for each iteration
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            for text, annotations in batch:
                # Create an Example object
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                # Update the model
                nlp.update([example], losses=losses, drop=0)
        print("Losses:", losses)

# STEP 3 - TEST THE UPDATED MODEL
# Save the model
output_dir = Path("model")
nlp.to_disk(output_dir)
print("Saved correctly!")

# Load updated model
nlp_updated = spacy.load(output_dir)

# TBD: test with a old sentence
doc = nlp_updated("Cynthia Erivo attends the 92nd Academy Awards, Designer: Versace, Year: 2020")
print("entities:", [(ent.text, ent.label_) for ent in doc.ents])
# TBD: test with a new sentence and an old brand
doc = nlp_updated("some Dude rolls up in Gucci and steals the show")
print("entities:", [(ent.text, ent.label_) for ent in doc.ents])
# TBD: test with a new sentence and a new brand
doc = nlp_updated("Christiano Ronaldo is attending in Nike")
print("entities:", [(ent.text, ent.label_) for ent in doc.ents])

# new sentence, no word
doc = nlp_updated("megan wore a tshirt while she was drinking coffee")
print("entities:", [(ent.text, ent.label_) for ent in doc.ents])



doc = nlp_updated("Hugo Boss used to design NS uniforms")
print("entities:", [(ent.text, ent.label_) for ent in doc.ents])
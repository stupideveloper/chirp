import os
from halo import Halo
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

use_strict_offline = False
model_path = './model/pegasus_paraphrase' # originally from tuner007/pegasus_paraphrase
remote_model_path = 'tuner007/pegasus_paraphrase'

max_length = 2000
temprature = 10
truncation = True
skip_special_tokens = True
num_beams = 30

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO: Add timestamp expiry on download
if(os.path.isdir(model_path) == False):
  download_loader = Halo(text='Downloading model and tokenizer for future use', spinner='bouncingBar', color='white').start()
  tokenizer = PegasusTokenizer.from_pretrained(remote_model_path)
  model = PegasusForConditionalGeneration.from_pretrained(remote_model_path).to(torch_device)  
  tokenizer.save_pretrained(model_path)
  model.save_pretrained(model_path)
  download_loader.succeed("Loaded tokenizer and model")
else:
  print("INFO: Using local tokenizer and model")
  tokenizer = PegasusTokenizer.from_pretrained(model_path)
  model = PegasusForConditionalGeneration.from_pretrained(model_path).to(torch_device)



def get_response(input_text,num_return_sequences):
  batch = tokenizer([input_text],truncation=truncation,padding='longest',max_length=max_length, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=max_length,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=temprature)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=skip_special_tokens)
  return tgt_text

from sentence_splitter import SentenceSplitter

splitter = SentenceSplitter(language='en')

def paraphraze(text):
  sentence_list = splitter.split(text)
  paraphrase = []

  for i in sentence_list:
    a = get_response(i,1)
    paraphrase.append(a)
    paraphrase2 = [' '.join(x) for x in paraphrase]
    paraphrase3 = [' '.join(x for x in paraphrase2) ]
  paraphrased_text = str(paraphrase3).strip('[]').strip("'")
  return paraphrased_text


from transformers import pipeline
import yake

if use_strict_offline:
  summarizer = pipeline("summarization", max_length=max_length)
else:
  summarizer = pipeline("summarization", model=model , tokenizer=tokenizer, max_length=max_length)


kw_extractor = yake.KeywordExtractor()
language = "en"
max_ngram_size = 3
deduplication_threshold = 0.9
numOfKeywords = 20
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)

# ============
# Main Function
# ============

def summarize(text):
  keywords_2 = []
  # summ = summarizer(text)[0]['summary_text']
  #keywords = custom_kw_extractor.extract_keywords(text)
  summ = "text"
  keywords = []
  for i in range(len(keywords)):
    keywords_2.append(keywords[i][0])

  paraphrased_text = paraphraze(text)
  return [summ,keywords_2,paraphrased_text]
print("✔ Loaded Paraphrase Engine")
#print(summarize("Cleopatra VII Philopator  was queen of the Ptolemaic Kingdom of Egypt from 51 to 30 BC, and its last active ruler.A member of the Ptolemaic dynasty, she was a descendant of its founder Ptolemy I Soter, a Macedonian Greek general and companion of Alexander the Great.After the death of Cleopatra, Egypt became a province of the Roman Empire, marking the end of the second to last Hellenistic state and the age that had lasted since the reign of Alexander (336–323 BC).Her native language was Koine Greek, and she was the only Ptolemaic ruler to learn the Egyptian language.In 58 BC, Cleopatra presumably accompanied her father, Ptolemy XII Auletes, during his exile to Rome after a revolt in Egypt (a Roman client state) allowed his rival daughter Berenice IV to claim his throne. Berenice was killed in 55 BC when Ptolemy returned to Egypt with Roman military assistance. When he died in 51 BC, the joint reign of Cleopatra and her brother Ptolemy XIII began, but a falling-out between them led to open civil war. After losing the 48 BC Battle of Pharsalus in Greece against his rival Julius Caesar (a Roman dictator and consul) in Caesar's Civil War, the Roman statesman Pompey fled to Egypt. Pompey had been a political ally of Ptolemy XII, but Ptolemy XIII, at the urging of his court eunuchs, had Pompey ambushed and killed before Caesar arrived and occupied Alexandria. Caesar then attempted to reconcile the rival Ptolemaic siblings, but Ptolemy's chief adviser, Potheinos, viewed Caesar's terms as favoring Cleopatra, so his forces besieged her and Caesar at the palace. Shortly after the siege was lifted by reinforcements, Ptolemy XIII died in the 47 BC Battle of the Nile; Cleopatra's half-sister Arsinoe IV was eventually exiled to Ephesus for her role in carrying out the siege. Caesar declared Cleopatra and her brother Ptolemy XIV joint rulers but maintained a private affair with Cleopatra that produced a son, Caesarion. Cleopatra traveled to Rome as a client queen in 46 and 44 BC, where she stayed at Caesar's villa. After the assassinations of Caesar and (on her orders) Ptolemy XIV in 44 BC, she named Caesarion co-ruler as Ptolemy XV."))
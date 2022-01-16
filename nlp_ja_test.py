import spacy
nlp: spacy.Language = spacy.load('ja_ginza')

# convert text to Doc
text: str = 'コンピューターは, 人間の脳にいかに近づけるでしょうか？'
doc: spacy.tokens.doc.Doc = nlp(text)

# Doc is the iterator of Token class
for token in doc:
  print(token.text, type(token)) # token.text:Japanese Morphological unit

from spacy import displacy

# Dependency Parsing
displacy.render(doc, style="dep", options={"compact":True},  jupyter=True)

# Entity Visualization
displacy.render(doc, style="ent", options={"compact":True},  jupyter=True)

doc2 = nlp('コンピューターサイエンスは, テクノロジー分野の研究領域です')

# noun_chunks:extract noun part
for chunk in doc2.noun_chunks:
  print(chunk.text, type(chunk))

# Extract noun words from part-of-speech tags
for token in doc2:
  if token.pos_ in ['NOUN', 'PROPN']: # NOUNが名詞、PROPNが固有名詞
    print(token.text, token.tag_, type(token))

print('doc1',doc.text)
print('doc2',doc2.text)

#Find the similarity between two sentences
print('cos similarity:', doc.similarity(doc2))

text3 = '''
気象庁の束田進也地震津波監視課長は、午後2時15分から記者会見を開き、北海道から沖縄にかけての広い範囲に出していた津波注意報を解除したことについて「これ以上潮位変化が高くなる可能性は小さくなったとみられる。
しばらく多少の潮位の変化は継続すると考えられるが、そのことを十分理解して行動してもらえれば災害のおそれはないとみられることから、津波注意報をすべて解除した。
海に入っての作業や釣りなどの際は十分に気をつけてほしい」と呼びかけました。
'''
# referenceed article: https://www3.nhk.or.jp/news/html/20220116/k10013433521000.html

doc3 = nlp(text3)

# save important parts into results
results = []
for chunk in doc3.noun_chunks:
    results.append((chunk.text, chunk.similarity(doc3)))

# show top 10 nouns considered to be important
print(sorted(results,key=lambda x: x[1],reverse=True)[:10])
```
Text analysis using NLTK

Visualization of data using Pandas, NumPy, Seaborn modules

Team : ToBeDecided

Ayush Dhoot 20ucs045

Shruti Sharma 20ucs189

Khushi Garg 20ucs096

Arush Kurundodi 20ucs028
```

```
Edit line 49 to show differences in data with and without stopwords:
```

```python
filtered_sentence = [(singularize(w.casefold()), t) for w, t in filtered_sentence if not w.lower() in STOPWORDS and len(w)>1 and w.lower() not in custom] # remove stopwords
filtered_sentence = [(singularize(w.casefold()), t) for w, t in filtered_sentence if len(w)>1 and w.lower() not in custom] # not remove stopwords
```
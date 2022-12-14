 from fuzzywuzzy import fuzz
 from fuzzywuzzy import process

# Simple Ratio
fuzz.ratio("this is a test", "this is a test!")

# Partial Ratio
fuzz.partial_ratio("this is a test", "this is a test!")

# Token Sort Ratio
a = fuzz.ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
print(a)

b = fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
print(b)

# Token Set Ratio
a = fuzz.token_sort_ratio({"fuzzy was a bear", "fuzzy fuzzy was a bear")
print(a)

b = fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
print(b)

# Process,使用字典的方式
choices = {"Atlanta Falcons", "New York Jets", "New York Giants","Dallas Cowboys"}
a = process.extract("new york jets", choices, limit=2)
print(a)

# Process  使用列表的方式
choices = {'a':"Atlanta Falcons", 'b':"New York Jets", 'c':"New York Giants", 'd':"Dallas Cowboys"}
a = process.extract("new york jets", choices, limit=2)
print(a)

b = process.extractOne("cowboys", choices)
print(b)


# You can also pass additional parameters to extractOne method to make it use a specific scorer. 
# A typical use case is to match file paths:
songs = 'Hypnotize'

a = process.extractOne("System of a down - Hypnotize - Heroin", songs)
print(a)

b = process.extractOne("System of a down - Hypnotize - Heroin", songs, scorer=fuzz.token_sort_ratio)
print(b)





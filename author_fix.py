"""
Fixing author names
Arjoon Srikanth
"""


for items in dict1:
	dict1['origAuthor']= dict1['author']
	dict1['author'] = re.findALL("[A-Z][a-z]*\.? [A-Z][a-z]*", dict1['author'])
return dict1
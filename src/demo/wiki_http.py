# Importing modules
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import re


ABBRV_PATTERN = re.compile('\([a-zA-z]+\)')
WHITE_SPACE_PATTERN = re.compile('\s+')


# Qyery request url
keyword = "Convolutional neural network (cnns)"
keyword = re.sub(ABBRV_PATTERN, '', keyword)

wiki_query_template = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&redirects=1&titles={urllib.parse.quote_plus(keyword)}"
  

# opening the url for reading
html = urllib.request.urlopen(wiki_query_template)
  
# parsing the html file
htmlParse = BeautifulSoup(html, 'html.parser')

html_ps = htmlParse.find_all("p")
print(keyword, len(html_ps))

# Getting all the paragraphs
for para in html_ps:
    para_text = para.get_text()
    para_text = para_text.replace("\\n", "")
    para_text = re.sub(WHITE_SPACE_PATTERN, ' ', para_text)

    print(len(para_text), para_text)
    print("-" * 15)

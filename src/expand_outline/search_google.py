'''
    @author Zhenke Chen
    @date 28/09/2021

    Collect data from Google Search with two steps:
    1. Use the Google Programmable Search Engine to fetch all the websites from the Google Search results
    2. Apply BeautifulSoup as the web clawer to collect the text from the websites
'''

# import the required packages
# import urllib.request as urlrequest
import urllib.parse
from bs4 import BeautifulSoup
from bs4.element import Comment
import numpy as np

# pip3 install google
from googlesearch import search
import pdb


# define the module level constants
MIN_PAR_LEN = 5
FAIL = -1


########### this user agent should be modified to your own user agent ###########
# parameters setting for the Google Programmable Search Engine, which can be modified based on different users
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36"
header = {"User-Agent": USER_AGENT}


# define the potential tags the websites may have for subtitles
HEADINGS_LIST = ["h", "h1", "h2", "h3"]
IGNORE_DOMAINS = set([
    "en.wikipedia.org",
    "www.youtube.com"
])


def search_google(question, result_num = 2):
    '''
        Apply the third-party package googlesearch to get the websites list from Google Search

        Keyword arguments:
        question -- the question posted by the user
        result_num -- the intended result number (default 2)
    '''
    unique_websites = []
    seen_domains = set()

    fetched_websites = search(
        query = question,           
        tld = "com",    # define the top level domain
        lang = "en",                
        num = result_num + 7,   # number of results
        start = 0,  # first result position to retrieve
        stop = 60,  # last result position to retrieve
        pause = 2.0 # lapse time between HTTP requests
    )
    
    # Remove duplicate websites
    for web in fetched_websites:
        if len(unique_websites) >= result_num:
            break

        domain = web.split("//")[1]
        domain = domain.split("/")[0]

        should_ignore = domain in IGNORE_DOMAINS
        if not should_ignore and domain not in seen_domains:
            seen_domains.add(domain)
            unique_websites.append(web)
    
    if len(unique_websites) != result_num:
        print("Warning: returned less than requested number of websites")

    return unique_websites



def get_text_chunks_helper(soup, heading_tag):
    """
        Extract full text under each heading_tag element
    """
    doc_text_chunks = []
    heading_elems = soup.find_all(heading_tag)

    # Get text chunk from each section of article
    for i in range(len(heading_elems)):
        cur_sect_elem = heading_elems[i]
        next_sect_text = None if i == len(heading_elems) - 1 else heading_elems[i + 1].get_text()

        cur_chunk = ""
        cur_sibling_elem = cur_sect_elem
        while cur_sibling_elem and cur_sibling_elem.get_text() != next_sect_text:
            cur_chunk += cur_sibling_elem.get_text()
            cur_sibling_elem = cur_sibling_elem.find_next_sibling()

        doc_text_chunks.append(cur_chunk)

    return doc_text_chunks


def get_partition_score(chunk_lens, all_text_len):

    score = 0
    num_chunks = len(chunk_lens)
    total_text_len = np.sum(chunk_lens)
    text_recall = np.sum(chunk_lens) / all_text_len
    
    if num_chunks == 0:
        return float('-inf')
    elif num_chunks < 2 or text_recall < 0.15:
        return -1000

    non_zero_lens = [l for l in chunk_lens if l > 0]
    score -= np.std(non_zero_lens) * (len(non_zero_lens) / num_chunks + 1)

    return score


def get_text_chunks(soup):
    all_text_len = len(soup.get_text())

    best_partition = None 
    best_partition_score = None

    # Choose partition that has the best score
    for heading_tag in HEADINGS_LIST:
        doc_text_chunks = get_text_chunks_helper(soup, heading_tag)

        chunk_lens = [len(t) for t in doc_text_chunks]
        cur_partiton_score = get_partition_score(chunk_lens, all_text_len)

        if best_partition is None or cur_partiton_score > best_partition_score:
            best_partition = doc_text_chunks
            best_partition_score = cur_partiton_score


    return best_partition
    

def extract_text(websites):
    '''
        Store the HTML content of websites into a temporary file and then claw the appropriate text from it using the BeautifulSoup
    
        Keyword arguments:
        websites -- the list storing the websites from Google Search results
    '''
    for url in websites:
        request = urllib.request.Request(url, headers = header)
        response = urllib.request.urlopen(request)

        doc = response.read().decode('utf-8', 'ignore')
        soup = BeautifulSoup(doc, "html.parser")

        text_chunks = get_text_chunks(soup)






if __name__ == "__main__":
    question = "Data structure history"

    websites = search_google(question, 25)

    print(websites)
    # extract_text(websites[:1])
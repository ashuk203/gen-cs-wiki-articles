HEADING_SEP = " [SEP] "
TITLE_SEP = " [EOT] "
PREFIX = "outline"

def clean_input_text(text):
    if len(text) > 2 and text[-1] == "." and text[-2] == ".":
        return text[:-1]

    return text

def clean_title(title):
    return title.replace("_", " ")


def create_model_input(raw_obj):
    res = {}
    res['id'] = raw_obj['index']
    res['prefix'] = PREFIX 
    res['input_text'] = clean_title(raw_obj["page_title"]) + TITLE_SEP + clean_input_text(raw_obj['summary'])
    res['target_text'] = HEADING_SEP.join(raw_obj['sections'])

    return res
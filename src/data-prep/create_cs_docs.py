import json

data_root_dir = "/home/aukey2/cs-wiki-data/cs_outlines_18k"
articles_file = f"{data_root_dir}/article_jsons.txt"


data_root_dir = "data/cs_docs_18k"
out_file = f"{data_root_dir}/docs.csv"


# sect_sep = " [SECTSEP] "
sect_sep = ", "


count = 0
with open(articles_file) as f_in, open(out_file, "w") as f_out:
    while True:
        count += 1
    
        # Get next line from file
        article = f_in.readline()

        # End of file is reached
        if not article or count >= 10:
            break

        article = json.loads(article[:-1])
        # outl = outl_f.readline()[:-1]


        outline = sect_sep.join(article["sections"])
        artic_fields = [
            article["keyword"],
            outline,
            article["summary"]
        ]

        res_line = '\t'.join(artic_fields)
        f_out.write(res_line + '\n')

        # print(f"{kw} outline: {outl}")
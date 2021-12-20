import wikipediaapi 

api_driver = wikipediaapi.Wikipedia('en')

def print_categories(category_name):
    page_title = f"Category:{category_name}"
    categ_wiki = api_driver.page(page_title)

    print(f"Fetching subcategories for {category_name}")
    sub_categories = [c.title for c in categ_wiki.categorymembers.values()]
    print(sub_categories)
    # print_categories_helper(categ_wiki.categorymembers)


def print_categories_helper(categorymembers, level=0, max_level=1):
        for c in categorymembers.values():
            print("%s: %s (ns: %d)" % ("*" * (level + 1), c.title, c.ns))
            if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
                print_categories_helper(c.categorymembers, level=level + 1, max_level=max_level)

if __name__ == '__main__':
    test_category = "Computer science"

    print_categories(test_category)
    


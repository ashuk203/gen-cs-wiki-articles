train_data_dir = "/home/aukey2/gen-cs-wiki-articles/rag_branch/data/cs_train_data"
actual_outl_file = f"{train_data_dir}/test.target"
gen_outl_file = f"{train_data_dir}/test.model_predicted"


def incr_freq(k, freq_dict):
    if k not in freq_dict:
        freq_dict[k] = 1
    else:
        freq_dict[k] += 1

def get_sorted_freqs(freq_dict, max_elems=50):
    res = list(freq_dict.items())
    res.sort(key = lambda t: t[1], reverse=True)

    if max_elems is None:
        return res
    else:
        return res[:max_elems]

def print_sorted_freqs(freq_items, pref=''):
    pref += '  >'
    for s in freq_items:
        print(pref, s)

    print('\n')

def parse_outl(outl, sect_delim=", "):
    return outl.split(sect_delim)

def calc_intersection(outl1, outl2):
    """
        Arguments:
            - outl1, outl2: list of strings representing article outlines

        Returns overlapping sections (using some criteria) between the two
        outlines.
    """

    outl1 = set(outl1)
    outl2 = set(outl2)

    common = outl1.intersection(outl2)

    return common, outl1 - common, outl2 - common

if __name__ == '__main__':

    with open(actual_outl_file) as f:
        actual_outls = f.readlines()

    with open(gen_outl_file) as f:
        gen_outls = f.readlines()

    num_outls = len(gen_outls)   # Should be same as len(actual_outls)

    total_gen_sects = 0
    total_act_sects = 0
    total_common_sects = 0

    common_sect_freqs = {}
    missed_sect_freqs = {}
    extra_sect_freqs = {}


    for i in range(num_outls):
        act_outl = parse_outl(actual_outls[i])
        gen_outl = parse_outl(gen_outls[i])

        common_sects, missed_sects, extra_sects = calc_intersection(act_outl, gen_outl)

        # Update counts for precision, recall, etc. calculations
        total_act_sects += len(act_outl)
        total_gen_sects += len(gen_outl)
        total_common_sects += len(common_sects)

        for s in common_sects:
            incr_freq(s, common_sect_freqs)
        
        for s in missed_sects:
            incr_freq(s, missed_sect_freqs)

        for s in extra_sects:
            incr_freq(s, extra_sect_freqs)

    # Metric calculation
    precision = total_common_sects / total_gen_sects
    recall = total_common_sects / total_act_sects

    most_common_sects = get_sorted_freqs(common_sect_freqs)
    most_missed_sects = get_sorted_freqs(missed_sect_freqs)
    most_extra_sects = get_sorted_freqs(extra_sect_freqs)

    print("Analysis over", num_outls, "examples")
    pref = "\t"

    print(pref, "Precision:", precision)
    print(pref, "Recall:", recall)
    print()

    print(pref, "Most common sections:")
    print_sorted_freqs(most_common_sects, pref)

    print(pref, "Most missed sections:")
    print_sorted_freqs(most_missed_sects, pref)

    print(pref, "Most extra sections:")
    print_sorted_freqs(most_extra_sects, pref)

    
            


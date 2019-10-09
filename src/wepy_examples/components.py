"""Module for getting components out of wepy by name"""

from importlib import import_module

import re

COMPONENT_SUFFIXES = {
    'Runner' : {'root_mod' : 'runners'},
    'Resampler' : {'root_mod' : 'resampling.resamplers'},
    'Distance' : {'root_mod' : 'resampling.distances'},
    'BC' : {'root_mod' : 'boundary_conditions'},
    'Reporter' : {'root_mod' : 'reporter'},
    'Mapper' : {'root_mod' : 'work_mapper'},

}


def parse_camel_case(class_name):

    words = re.findall('[A-Z][^A-Z]*', class_name)

    return words

def resolve_class_name(class_name):


    words = parse_camel_case(class_name)

    prefix = words[0:-1]
    suffix = words[-1]

    # match the suffix
    if suffix not in COMPONENT_SUFFIXES:
        raise ValueError("Suffix type {suffix} not recognized"))
    else:
        root_mod = COMPONENT_SUFFIXES[suffix]['root_mod']

    # then get the final leg

    def cons_caps_segs(words):

        # if there are consecutive caps then don't split them
        i = 0
        segs = []
        curr_seg = ''
        while True:

            print("Word: {}".format(words[i]))

            # if the first letter of the word is caps
            if words[i][0].isupper():
                print("is upper")

                curr_seg += words[i]

            else:
                print('is not upper')
                segs.append(curr_seg)
                curr_seg = ''
                segs.append(words[i])

            i += 1

            if i == len(words):
                segs.append(curr_seg)
                break

        return segs


    '_'.join(prefix)
    
    return module_path

def grab(component):

    

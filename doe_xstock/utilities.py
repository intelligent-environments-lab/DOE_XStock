"""A module containing utility functions that are used by other modules.
Methods
-------
get_read_json(filepath)
    Return json document as dictionary.
"""

import simplejson as json
import os
from datetime import datetime

def read_json(filepath):
    """Return json document as dictionary.
    Parameters
    ----------
    filepath : str
       pathname of JSON document.
    Returns
    -------
    dict
        JSON document converted to dictionary.
    """

    with open(filepath) as f:
        json_file = json.load(f)
    return json_file

def write_json(filepath=None,dictionary=None):
    """Writes dictionary to json file. Returns boolen value of operation success. 
    
    Parameters
    ----------
    filepath : str
        pathname of JSON document.
    dictionary: dict
        dictionary to convert to JSON.
    """

    def default(obj):
        if isinstance(obj,datetime):
            return obj.__str__()

    try:
        with open(filepath,'w') as f:
            json.dump(dictionary,f,ignore_nan=True,sort_keys=False,default=default,indent=2)
    except Exception as e:
        pass

def write_data(data,filepath,mode='w'):
    with open(filepath,mode) as f:
        f.write(data)

def get_data_from_path(filepath):
    if os.path.exists(filepath):
        with open(filepath,mode='r') as f:
            data = f.read()
    else:
        data = filepath

    return data

def unnest_dict(nested_dict,seperator=',',prefix=None):
    master_unnest = {}

    for key, value in nested_dict.items():
        key = f'{"" if prefix is None else str(prefix)+seperator}{key}'
        if isinstance(value,dict):
            child_unnest = unnest_dict(value,prefix=key)
            master_unnest = {**master_unnest,**child_unnest}

        else:
            master_unnest[key] = value

    return master_unnest

def split_lines(text,line_character_limit=24,delimiter=' '):
    text = text.strip()
    
    if len(text) > line_character_limit:
        words = text.split(delimiter)
        lines = []
        
        i = 0
        while i < len(words):
            line = words[i]
            i += 1

            for j in range(i,len(words)): 
                if len(line + words[j]) < line_character_limit:
                    line = delimiter.join([line,words[j]])
                    i += 1
                else:
                    break

            lines.append(line)

        text = '\n'.join(lines)

    else:
        pass

    return text
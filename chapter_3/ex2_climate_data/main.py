import os
import json
def read_output_file():
    # read in all the files in a dictionary structure
    # key is the abbreviated country-code, the value is a list of all the data-lines included in the file converted to float
    file_list = os.listdir("__files")
    with open('raw_data.json', 'w+') as output:
        data_by_country_unsorted = {}
        for filename in file_list:
            with open(f"__files/{filename}", "r") as file:
                values = []
                country_code = filename.replace(".csv", "")
                for value in file:
                    values.append(float(value))
                data_by_country_unsorted[country_code] = values
        # Order the dictionary by the keys
        data_by_country = dict(sorted(data_by_country_unsorted.items()))
        # Output the dictionary to a json-ﬁle raw_data.json with the json.dump-function (help: import json; help(json.dump).
        json.dump(data_by_country, output)

if __name__ == "__main__":
    read_output_file()
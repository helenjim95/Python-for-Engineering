import os
import json

data_by_country = {}
def read_output_file():
    global data_by_country
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

def aggregate():
    with open('aggregated_data.json', 'w+') as output:
        output_by_country = {}
        for key, values in data_by_country.items():
            t_avg = sum(values) / len(values)
            t_max = max(values)
            t_min = min(values)
            output_by_country[key] = {'t_avg': t_avg, 't_max' : t_max, 't_min' : t_min}
        json.dump(data_by_country, output)


if __name__ == "__main__":
    read_output_file()
    aggregate()
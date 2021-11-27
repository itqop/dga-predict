import random
import pandas as pd


def preprocessing_csv(path_input, path_output='.temp'):
    file_input = open(path_input, 'r')
    file_output = open(path_output, 'w')
    temp_array = []
    temp_row = ""
    cols = []
    check = 0

    for line in file_input:
        if check == 0:
            check += 1

            continue

        cols = line[line.index(',') + 1:].split(",")
        text = cols[0]

        if len(text) > 3:
            for i in range(len(text) - 2):
                temp_row += text[i:i + 3] + " "

        else:
            temp_row = text + " "

        temp_array.append(temp_row[:-1] + "," + cols[1])
        temp_row = ""

    random.shuffle(temp_array)
    temp_array.insert(0, 'domain,subclass\n')

    for line in temp_array:
        file_output.write(line)

    file_input.close()
    file_output.close()
    data_temp = pd.read_csv(path_output)

    return [
        data_temp.iloc[:, 0].values,
        data_temp.iloc[:, -1].values
    ]

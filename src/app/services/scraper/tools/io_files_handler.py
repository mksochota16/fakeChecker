import pandas as pd


def get_saved_html_markers():
    markers_dict = {}
    f = open(f"data/html_markers.txt", "r")
    lines = f.readlines()
    for line in lines:
        data = line.split("; ")
        if data[1][0] == '[':
            split_data = data[1][1:-2].replace("\'", "").replace("\\", "").split(', ')
            info_list = []
            for info in split_data:
                info_list.append(info)
            markers_dict[data[0]] = info_list
        else:
            markers_dict[data[0]] = data[1][:-1]
    f.close()
    return markers_dict


def save_new_html_markers(markers_dict):
    f = open(f"data/html_markers.txt", "w")
    for key in markers_dict:
        f.write(f"{key}; {markers_dict[key]}\n")
    f.close()


def get_clusters():
    f = open(f"data/CLUSTERS.txt", 'r', encoding='UTF-8')
    lines = f.readlines()
    clusters_dict = {}
    cluster_name = None
    type_list = []
    for line in lines:
        # title line
        if line[0] == '=':
            # not first iteration
            if cluster_name is not None:
                clusters_dict[cluster_name] = type_list
                type_list = []
            cluster_name = line.replace('=', '')[1:-2]
        else:
            type_list.append(line[:-1])
    clusters_dict[cluster_name] = type_list
    f.close()
    return clusters_dict


def update_clusters(new_type, cluster):
    with open("data/CLUSTERS.txt", "r", encoding='UTF-8') as f:
        contents = f.readlines()

    index = 0
    for line in contents:
        if line[0] == '=' and line.replace('=', '')[1:-2] == cluster:
            break
        else:
            index += 1

    contents.insert(index + 1, new_type + '\n')

    with open("data/CLUSTERS.txt", "w", encoding='UTF-8') as f:
        contents = "".join(contents)
        f.write(contents)


def import_dataframes_from_names_files():
    male_names = pd.read_csv('data/polish-male-names.csv').drop('PŁEĆ', 1)
    female_names = pd.read_csv('data/polish-female-names.csv').drop('PŁEĆ', 1)
    names = pd.concat([male_names, female_names])
    names['sum'] = names.groupby("IMIĘ_PIERWSZE")['LICZBA_WYSTĄPIEŃ'].transform('sum')
    names = names.drop('LICZBA_WYSTĄPIEŃ', 1)

    male_surnames = pd.read_csv('data/polish-male-surnames.csv')
    female_surnames = pd.read_csv('data/polish-female-surnames.csv')
    surnames = pd.concat([male_surnames, female_surnames])
    surnames['sum'] = surnames.groupby("Nazwisko aktualne")['Liczba'].transform('sum')
    surnames = surnames.drop('Liczba', 1)

    return names, surnames
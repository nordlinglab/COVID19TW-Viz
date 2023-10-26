import copy
import ast
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from bidict import bidict
from lifelines import KaplanMeierFitter
from numpy.random import default_rng

def clean_taiwan_data(taiwan_data_sheet, start_index=1, end_index=579):
    # Data cleaning for pandas Taiwan COVID-19 data

    # Clean out ID 530 since it was removed by Taiwan CDC
    if end_index >= 530:
        taiwan_data_sheet = taiwan_data_sheet[taiwan_data_sheet.id != 530]

    # Correct ID 19's ICU time
    data = taiwan_data_sheet.copy()
    case19_icu_date = data[data.id == 19].icu.values
    case19_index = data[data.id == 19].index
    data.loc[case19_index, 'confirmed_date'] = case19_icu_date

    # Set range
    table_start_index = data[data['id'] == start_index].index.to_numpy()[0]
    table_end_index = data[data['id'] == end_index].index.to_numpy()[0]
    data = data.iloc[table_end_index:table_start_index+1]

    # Add asymptomatic
    data = add_asymptomatic_date(data)

    # Replace symptomatic to death to be symptomatic to critical and directly to dead
    for i in data.index:
        if (data.loc[i, 'icu'] == 'C' or data.loc[i, 'icu'] == 'X'):
            if (data.loc[i, 'death_date'] != 'C') and (data.loc[i, 'death_date'] != 'X'):
                data.loc[i, 'icu'] = data.loc[i, 'death_date']

    return data


def add_asymptomatic_date(data):
    confirm_date = data.confirmed_date
    symptom_onset_date = data.onset_of_symptom
    infection_date = data.infection_date
    source_id = data.source_infected_case
    source_id = source_id.str.replace('ID ', '')
    asymptomatic_date = pd.DataFrame(
        {'asymptomatic_date': np.zeros(len(source_id))}, index=data.index)

    # Create array of first spread date based on infection date and source ID
    first_spread_date = pd.DataFrame(
        {'first_spread_date': np.zeros(len(source_id))}, index=data.index)
    for i in source_id.unique():
        if (i != 'C') and (i != 'X'):
            first_spread_date_temp = min(infection_date[source_id == i])
            first_spread_date.loc[data[data.id ==
                                       int(i)].index[0]] = first_spread_date_temp
            first_spread_date = first_spread_date.replace(0, 'C')
    all_date = pd.concat(
        [symptom_onset_date, confirm_date, first_spread_date, infection_date], axis=1)
    # all_date = pd.concat([infection_date], axis=1)
    all_date = all_date.replace('C', np.nan)
    all_date = all_date.replace('N', np.nan)
    all_date = all_date.replace('X', np.nan)
    all_date = all_date.replace('Mid of October', np.nan)
    all_date = all_date.replace('2020-04-26 to 2020-04-27', np.nan)
    all_date = all_date.replace('2020-04-14 to 2020-04-18', np.nan)
    all_date = all_date.replace('2020-04-06, 2020-04-07', np.nan)
    all_date = all_date.replace('2020-01-28 to 2020-02-06', np.nan)

    # Find min date and assign it as asymptomatic date
    for k in all_date.index:
        all_date_row = all_date.loc[k].dropna()
        if len(all_date_row) > 0:
            asymptomatic_date.loc[k] = all_date_row.min()
    asymptomatic_date = asymptomatic_date.replace(0, 'C')

    result = pd.concat([data, asymptomatic_date], axis=1)
    return (result)


def generate_taiwan_contact_network(data):
    '''
    Generate contact network from Taiwan COVID-19 data. The network is undirected and include both the infected and uninfected contacts.

    Parameters
    ----------
    data: pandas dataframe. Taiwan COVID-19 individual subject data.

    Returns
    -------
    contact_network: networkx graph
    contact_type_dict: dictionary for effective contact type
    uninfected_contact_type_dict: dictionary for uninfected contact type
    '''
    # Initialization
    contact_network = nx.MultiGraph()
    contact_type_start_index = data.columns.get_loc('couple')
    contact_type_end_index = data.columns.get_loc('other_unknown_contact')
    uninfected_contact_type_start_index = data.columns.get_loc(
        'number_of_uninfected_contact_travel')
    uninfected_contact_type_end_index = data.columns.get_loc(
        'number_of_uninfected_contact_friend')

    # Construct contact type label
    contact_type_dict = {}
    uninfected_contact_type_dict = {}
    for i, column in enumerate(data.columns[contact_type_start_index:contact_type_end_index+1]):
        contact_type_dict[column] = i
    contact_type_dict = bidict(contact_type_dict)
    contact_type_final_index = contact_type_dict['other_unknown_contact']
    # Add uninfected contact type
    uninfected_contact_type_dict['number_of_uninfected_contact_travel'] = contact_type_dict['travel_together']
    uninfected_contact_type_dict['number_of_uninfected_contact_flight'] = contact_type_dict['the_same_flight']
    uninfected_contact_type_dict['number_of_uninfected_contact_flight_nearby_seats'] = contact_type_dict['the_same_flight_nearby_seat']
    uninfected_contact_type_dict['number_of_uninfected_contact_car'] = contact_type_dict['the_same_car']
    # New label
    uninfected_contact_type_dict['number_of_uninfected_contact_ship'] = contact_type_final_index + 1
    contact_type_final_index += 1
    uninfected_contact_type_dict['number_of_uninfected_contact_live_together'] = contact_type_dict['live_together']
    uninfected_contact_type_dict['number_of_uninfected_contact_family'] = contact_type_dict['family']
    uninfected_contact_type_dict['number_of_uninfected_contact_coworker'] = contact_type_dict['coworker']
    uninfected_contact_type_dict['number_of_uninfected_contact_others'] = contact_type_dict['other_unknown_contact']
    uninfected_contact_type_dict['number_of_uninfected_contact_hospital'] = contact_type_dict['the_same_hospital']
    uninfected_contact_type_dict['number_of_uninfected_contact_quarantine_hotel'] = contact_type_dict['the_same_quarantine_hotel']
    uninfected_contact_type_dict['number_of_uninfected_contact_hotel'] = contact_type_dict['the_same_hotel']
    uninfected_contact_type_dict['number_of_uninfected_contact_school'] = contact_type_dict['the_same_school']
    uninfected_contact_type_dict['number_of_uninfected_contact_couple'] = contact_type_dict['couple']
    uninfected_contact_type_dict['number_of_uninfected_contact_panshi'] = contact_type_dict['panshi_fast_combat_support_ship']
    uninfected_contact_type_dict['number_of_uninfected_contact_friend'] = contact_type_dict['friend']
    uninfected_contact_type_dict = bidict(uninfected_contact_type_dict)

    # Construct contact network
    uninfected_id = 0
    for i, row in data.iterrows():
        row_id = row['id']

        # Loop effective contact columns
        for j in range(contact_type_start_index, contact_type_end_index+1):
            if (row[j] != 'C') and (row[j] != 'X'):  # Effective contact exist
                # Extract effective contact ID
                effective_contact_id_tmp = row[j]
                effective_contact_id_tmp = effective_contact_id_tmp.replace(
                    'ID ', '').replace(' ', '').split(',')
                effective_contact_id_tmp = np.array(
                    effective_contact_id_tmp).astype(int)
                # Extract contact type
                contact_type_tmp = row.index[j]
                contact_type_tmp = contact_type_dict[contact_type_tmp]
                # Add edge
                for effective_contact_id in effective_contact_id_tmp:
                    # Test if the contact already exist
                    try:
                        all_previous_contacts = [contact['contact_type'] for contact in contact_network['I'+str(
                            effective_contact_id)]['I'+str(row_id)].values()]
                    except:
                        all_previous_contacts = []
                    if contact_type_tmp in all_previous_contacts:  # Edge already exist
                        pass
                    else:
                        contact_network.add_edge(
                            'I'+str(row_id), 'I'+str(effective_contact_id), contact_type=contact_type_tmp)

        # Loop uninfected contact columns
        for k in np.arange(uninfected_contact_type_start_index, uninfected_contact_type_end_index+1, 2):
            if (row[k] != 'C') and (row[k] != 'X'):  # Uninfected contact exist
                # Extract uninfected contact ID
                uninfected_contact_type_tmp = row.index[k]
                uninfected_contact_type_tmp = uninfected_contact_type_dict[
                    uninfected_contact_type_tmp]
                # Extract saved linked nodes
                neighbors = []
                try:
                    for neighbor, attributes in contact_network['I'+str(row_id)].items():
                        for attribute in attributes.values():
                            if attribute['contact_type'] == uninfected_contact_type_tmp:
                                neighbors.append(neighbor)
                except:
                    pass
                count_integers = len(
                    [x for x in neighbors if isinstance(x, np.integer)])
                if count_integers == 0:
                    uninfected_id_array = np.arange(
                        uninfected_id, uninfected_id+row[k])
                    for l in uninfected_id_array:
                        contact_network.add_edge(
                            'I'+str(row_id), l, contact_type=uninfected_contact_type_tmp)
                    if (row[k+1] != 'C') and (row[k+1] != 'X'):  # Intersection exist
                        intersection = row[k+1]
                        intersection = ast.literal_eval(intersection)
                        intersection_number = intersection[0]
                        intersection_id = intersection[1::]
                        for l in uninfected_id_array[0:intersection_number]:
                            for m in intersection_id:
                                contact_network.add_edge(
                                    'I'+str(m), l, contact_type=uninfected_contact_type_tmp)
                else:
                    extra_cases = row[k] - count_integers
                    if extra_cases > 0:
                        uninfected_id_array = np.arange(
                            uninfected_id, uninfected_id+extra_cases)
                        for l in uninfected_id_array:
                            contact_network.add_edge(
                                'I'+str(row_id), l, contact_type=uninfected_contact_type_tmp)
                        if (row[k+1] != 'C') and (row[k+1] != 'X'):  # Intersection exist
                            intersection = row[k+1]
                            intersection = ast.literal_eval(intersection)
                            intersection_number = intersection[0]
                            intersection_id = intersection[1::]
                            for l in uninfected_id_array[0:intersection_number]:
                                for m in intersection_id:
                                    contact_network.add_edge(
                                        'I'+str(m), l, contact_type=uninfected_contact_type_tmp)
                uninfected_id = uninfected_id_array[-1]+1

    return (contact_network, contact_type_dict, uninfected_contact_type_dict)


def generate_taiwan_infection_contact_network(data):
    '''
    Generate contact network for the infection path from Taiwan COVID-19 data. The network is directed and include only infected cases.

    Parameters
    ----------
    data: pandas dataframe. Taiwan COVID-19 individual subject data.

    Returns
    -------
    infection_contact_network: networkx graph
    contact_type_dict: dictionary for effective contact type
    uninfected_contact_type_dict: dictionary for uninfected contact type
    '''
    # Initialization
    infection_contact_network = nx.DiGraph()
    contact_type_start_index = data.columns.get_loc('couple')
    contact_type_end_index = data.columns.get_loc('other_unknown_contact')
    uninfected_contact_type_start_index = data.columns.get_loc(
        'number_of_uninfected_contact_travel')
    uninfected_contact_type_end_index = data.columns.get_loc(
        'number_of_uninfected_contact_friend')

    # Construct contact type label
    contact_type_dict = {}
    uninfected_contact_type_dict = {}
    for i, column in enumerate(data.columns[contact_type_start_index:contact_type_end_index+1]):
        contact_type_dict[column] = i
    contact_type_dict = bidict(contact_type_dict)
    contact_type_final_index = contact_type_dict['other_unknown_contact']
    # Add uninfected contact type
    uninfected_contact_type_dict['number_of_uninfected_contact_travel'] = contact_type_dict['travel_together']
    uninfected_contact_type_dict['number_of_uninfected_contact_flight'] = contact_type_dict['the_same_flight']
    uninfected_contact_type_dict['number_of_uninfected_contact_flight_nearby_seats'] = contact_type_dict['the_same_flight_nearby_seat']
    uninfected_contact_type_dict['number_of_uninfected_contact_car'] = contact_type_dict['the_same_car']
    # New label
    uninfected_contact_type_dict['number_of_uninfected_contact_ship'] = contact_type_final_index + 1
    contact_type_final_index += 1
    uninfected_contact_type_dict['number_of_uninfected_contact_live_together'] = contact_type_dict['live_together']
    uninfected_contact_type_dict['number_of_uninfected_contact_family'] = contact_type_dict['family']
    uninfected_contact_type_dict['number_of_uninfected_contact_coworker'] = contact_type_dict['coworker']
    uninfected_contact_type_dict['number_of_uninfected_contact_others'] = contact_type_dict['other_unknown_contact']
    uninfected_contact_type_dict['number_of_uninfected_contact_hospital'] = contact_type_dict['the_same_hospital']
    uninfected_contact_type_dict['number_of_uninfected_contact_quarantine_hotel'] = contact_type_dict['the_same_quarantine_hotel']
    uninfected_contact_type_dict['number_of_uninfected_contact_hotel'] = contact_type_dict['the_same_hotel']
    uninfected_contact_type_dict['number_of_uninfected_contact_school'] = contact_type_dict['the_same_school']
    uninfected_contact_type_dict['number_of_uninfected_contact_couple'] = contact_type_dict['couple']
    uninfected_contact_type_dict['number_of_uninfected_contact_panshi'] = contact_type_dict['panshi_fast_combat_support_ship']
    uninfected_contact_type_dict['number_of_uninfected_contact_friend'] = contact_type_dict['friend']
    uninfected_contact_type_dict = bidict(uninfected_contact_type_dict)

    # Construct contact network
    uninfected_id = 0
    for i, row in data.iterrows():
        row_id = row['id']
        source_id = row['source_infected_case']
        if (source_id != 'C') and (source_id != 'X'):  # Source ID available
            source_id = np.int32(source_id.replace('ID ', ''))
            # Loop effective contact columns
            for j in range(contact_type_start_index, contact_type_end_index+1):
                if (row[j] != 'C') and (row[j] != 'X'):  # Effective contact exist
                    # Extract effective contact ID
                    effective_contact_id_tmp = row[j]
                    effective_contact_id_tmp = np.array(effective_contact_id_tmp.replace(
                        'ID ', '').replace(' ', '').split(','), dtype=int)
                    if source_id in effective_contact_id_tmp:  # Effective contact from source ID
                        contact_type = row.index[j]
                        contact_type = contact_type_dict[contact_type]
                        infection_contact_network.add_edge(
                            'I'+str(source_id), 'I'+str(row_id), contact_type=contact_type)

    return (infection_contact_network, contact_type_dict, uninfected_contact_type_dict)


def generate_edge_list(taiwan_contact_network, infection_contact_network=None):
    '''
    Generate edge list (mixed graph) for Taiwanese individual subject COVID-19 data.

    Parameters
    ----------
    taiwan_contact_network: networkx graph. This network is undirected and include both infected and uninfected cases.
    infection_contact_network: networkx graph. This network is directed and include only infected cases.

    Returns
    -------
    edge_list: list containing [source_id, target_id, contact_type, 'directed/undirected']
    '''
    edge_list = []

    # Construct edge list for the undirected network first
    for source_id, target_id, contact_type in taiwan_contact_network.edges(data=True):
        contact_type = contact_type['contact_type']
        edge_list.append([source_id, target_id, contact_type, False])

    # Merge the directed and undirected network
    if infection_contact_network is not None:
        for source_id, target_id, contact_type in infection_contact_network.edges(data=True):
            contact_type = contact_type['contact_type']
            # Search the row in the edge list
            for i in range(len(edge_list)):
                if edge_list[i][0] == source_id and edge_list[i][1] == target_id:  # Normal case
                    edge_list[i][-1] = True
                elif edge_list[i][1] == source_id and edge_list[i][0] == target_id:  # Reverse case
                    edge_list[i][0], edge_list[i][1] = edge_list[i][1], edge_list[i][0]
                    edge_list[i][-1] = True
                else:
                    continue

    return (edge_list)


def write_cytoscape_file(edge_list, filename):
    # write_cytoscape_file: Create cytoscape file based on edge list. Daily information is ignored.
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'target', 'interaction', 'directed'])
        for row in edge_list:
            source = row[0]
            target = row[1]
            group = row[2]
            direct = row[-1]
            writer.writerow([source, target, group, direct])


def extract_state_data(data, start_state, end_state, exclude_state=0):
    """
    Extract data from a pandas dataframe for a given state.

    Parameters
    ----------
    data: pandas dataframe
        The pandas dataframe to extract the data from.
    start_state: str
        String header of start state such as 'onset_of_symptom'.
    end_state: str
        String header of end state such as 'recovery'
    exclude_state: str
        string header of state, such as 'icu', such that the cases with the state
        should be exclude

    Returns
    -------
    data: pandas dataframe
        The pandas dataframe with the extracted data.
    """
    start_state_date = data[start_state]
    end_state_date = data[end_state]
    if type(exclude_state) == str:
        exclude_state_date = data[exclude_state]
        row_map = (pd.DataFrame(start_state_date).transpose().dtypes != object) & \
            ((pd.DataFrame(end_state_date).transpose().dtypes != object) &
             ~(pd.DataFrame(exclude_state_date).transpose().dtypes != object))
    else:
        row_map = (pd.DataFrame(start_state_date).transpose().dtypes != object) & \
            (pd.DataFrame(end_state_date).transpose().dtypes != object)
    start_state_date = start_state_date[row_map]
    end_state_date = end_state_date[row_map]

    # Calculate state transition days for each cases
    start_to_end_days = end_state_date - start_state_date

    return (start_to_end_days)


def transform_course_object_to_population_data(course_of_disease_data_list, time_limit, detection_rate=1, death_detection_rate=1):
    # Pick death cases
    death_index = np.array([])
    for i, course_of_disease_data in enumerate(course_of_disease_data_list):
        if ~np.isnan(course_of_disease_data.date_of_death):
            if course_of_disease_data.date_of_death <= time_limit:
                death_detection_state = np.random.choice([True, False], size=1, p=[
                    death_detection_rate, 1-death_detection_rate])
                if death_detection_state == True:
                    death_index = np.append(death_index, i)
    # Pick recovered cases
    expected_confirmed_number = round(
        detection_rate*len(course_of_disease_data_list))
    recovered_index = np.arange(len(course_of_disease_data_list))
    recovered_index = np.array(
        [i for i in recovered_index if i not in death_index])
    add_number = round(expected_confirmed_number - len(death_index))
    index = np.sort(np.int64(np.append(death_index, np.random.choice(
        recovered_index, size=add_number, replace=False))))
    course_of_disease_data_list_tmp = course_of_disease_data_list[index]

    infected_dates = np.array([])
    contagious_dates = np.array([])
    symptomatic_dates = np.array([])
    confirmed_dates = np.array([])
    critically_ill_dates = np.array([])
    recovered_dates = np.array([])
    death_dates = np.array([])
    for course_of_disease_data in course_of_disease_data_list_tmp:
        infected_dates = np.append(
            infected_dates, course_of_disease_data.infection_day)
        contagious_dates = np.append(
            contagious_dates, course_of_disease_data.infection_day+course_of_disease_data.latent_period)
        if ~np.isnan(course_of_disease_data.incubation_period):
            symptomatic_dates = np.append(
                symptomatic_dates, course_of_disease_data.infection_day+course_of_disease_data.incubation_period)
        confirmed_dates = np.append(
            confirmed_dates, course_of_disease_data.positive_test_date)
        if ~np.isnan(course_of_disease_data.date_of_critically_ill):
            critically_ill_dates = np.append(
                critically_ill_dates, course_of_disease_data.date_of_critically_ill)
        if ~np.isnan(course_of_disease_data.date_of_recovery):
            recovered_dates = np.append(
                recovered_dates, course_of_disease_data.date_of_recovery)
        if ~np.isnan(course_of_disease_data.date_of_death):
            death_dates = np.append(
                death_dates, course_of_disease_data.date_of_death)

    # Compute the daily counts using np.bincount
    daily_infected_cases = np.bincount(
        np.int32(infected_dates), minlength=time_limit+1)[:time_limit+1]
    daily_contagious_cases = np.bincount(
        np.int32(contagious_dates), minlength=time_limit+1)[:time_limit+1]
    daily_symptomatic_cases = np.bincount(
        np.int32(symptomatic_dates), minlength=time_limit+1)[:time_limit+1]
    daily_confirmed_cases = np.bincount(
        np.int32(confirmed_dates), minlength=time_limit+1)[:time_limit+1]
    daily_critically_ill_cases = np.bincount(
        np.int32(critically_ill_dates), minlength=time_limit+1)[:time_limit+1]
    daily_recovered_cases = np.bincount(
        np.int32(recovered_dates), minlength=time_limit+1)[:time_limit+1]
    daily_death_cases = np.bincount(
        np.int32(death_dates), minlength=time_limit+1)[:time_limit+1]

    return (daily_infected_cases, daily_contagious_cases, daily_symptomatic_cases, daily_confirmed_cases, daily_critically_ill_cases, daily_recovered_cases, daily_death_cases)


def convert_Taiwan_data_to_test_matrix(data):
    # Each column represents age, gender,
    data_matrix = np.ones([len(data), 11])*np.nan
    # time from confirmed to negative tests, infection, symptom-onset, critically ill, recovered,
    # death, size of total unique contacts in each contact type,
    # and size of the infection cluster in each contact type.

    # Age
    ages = copy.deepcopy(data.age.to_numpy())
    ages[ages == '0 to 10'] = 5
    ages[ages == '10 to 20'] = 15
    ages[ages == '20 to 30'] = 25
    ages[ages == '30 to 40'] = 35
    ages[ages == '40 to 50'] = 45
    ages[ages == '50 to 60'] = 55
    ages[ages == '60 to 70'] = 65
    ages[ages == '70 to 80'] = 75
    ages[ages == '80 to 90'] = 85
    ages[(ages == 'C') | (ages == 'N')] = np.nan
    data_matrix[:, 0] = ages

    # Gender, Male=1, Female=0
    genders = copy.deepcopy(data.gender.to_numpy())
    genders[genders == 'Male'] = 1
    genders[genders == 'Female'] = 0
    genders[(genders == 'C') | (genders == 'None')] = np.nan
    data_matrix[:, 1] = genders

    # Course of disease
    data_start_index = data.index[0]
    # Infection to symptomatic
    infection_to_symptomatic_days = extract_state_data(data, 'infection_date',
                                                       'onset_of_symptom', 'recovery')
    infection_to_symptomatic_days_index = infection_to_symptomatic_days.index.to_numpy() - \
        data_start_index
    data_matrix[infection_to_symptomatic_days_index,
                2] = infection_to_symptomatic_days.dt.days.to_numpy()

    # Infection to recovered
    infection_to_recovered_days = extract_state_data(data, 'infection_date',
                                                     'recovery', 'onset_of_symptom')
    infection_to_recovered_days_index = infection_to_recovered_days.index.to_numpy() - \
        data_start_index
    data_matrix[infection_to_recovered_days_index,
                3] = infection_to_recovered_days.dt.days.to_numpy()

    # Symptomatic to critically ill
    symptom_to_critically_ill_days = extract_state_data(
        data, 'onset_of_symptom', 'icu', 'recovery')
    symptom_to_dead_days = extract_state_data(
        data, 'onset_of_symptom', 'death_date', 'icu')
    critically_ill_to_dead_days = extract_state_data(
        data, 'icu', 'death_date', 'recovery')
    # Replace symptomatic to death to be symptomatic to critical and directly to dead
    for i in symptom_to_dead_days.index:
        symptom_to_critically_ill_days[i] = symptom_to_dead_days[i]
        critically_ill_to_dead_days[i] = pd.Timedelta(days=0)
        symptom_to_dead_days[i] = pd.NaT
    symptom_to_critically_ill_days_index = symptom_to_critically_ill_days.index.to_numpy() - \
        data_start_index
    data_matrix[symptom_to_critically_ill_days_index,
                4] = symptom_to_critically_ill_days.dt.days.to_numpy()

    # Symptomatic to recovered
    symptom_to_recover_days = extract_state_data(
        data, 'onset_of_symptom', 'recovery', 'icu')
    symptom_to_recover_days_index = symptom_to_recover_days.index.to_numpy() - \
        data_start_index
    data_matrix[symptom_to_recover_days_index,
                5] = symptom_to_recover_days.dt.days.to_numpy()

    # Critically ill to recovered
    critically_ill_to_recover_days = extract_state_data(
        data, 'icu', 'recovery', 'death_date')
    critically_ill_to_recover_days_index = critically_ill_to_recover_days.index.to_numpy() - \
        data_start_index
    data_matrix[critically_ill_to_recover_days_index,
                6] = critically_ill_to_recover_days.dt.days.to_numpy()

    # Critically ill to death
    critically_ill_to_dead_days_index = critically_ill_to_dead_days.index.to_numpy() - \
        data_start_index
    data_matrix[critically_ill_to_dead_days_index,
                7] = critically_ill_to_dead_days.dt.days.to_numpy()

    # First negative test date to confirmed date
    negative_test_date_1_to_confrimed = extract_state_data(
        data, 'negative_test_date_1', 'confirmed_date')
    negative_test_date_1_to_confrimed_index = negative_test_date_1_to_confrimed.index.to_numpy() - \
        data_start_index
    data_matrix[negative_test_date_1_to_confrimed_index,
                8] = negative_test_date_1_to_confrimed.dt.days.to_numpy()

    # Symptomatic to confirmed
    symptomatic_to_confrimed = extract_state_data(
        data, 'onset_of_symptom', 'confirmed_date')
    symptomatic_to_confrimed_index = symptomatic_to_confrimed.index.to_numpy() - \
        data_start_index
    data_matrix[symptomatic_to_confrimed_index,
                9] = symptomatic_to_confrimed.dt.days.to_numpy()

    # Size of unique contacts
    contacts = copy.deepcopy(data['number_of_contact_total'].to_numpy())
    contacts[contacts == 'C'] = np.nan
    data_matrix[:, 10] = contacts

    return (data_matrix)


def km_plot_transform(state_transition_days):
    """
    Plot the state transition curve to Kaplan-Meier curve and return the uncertainty.

    Parameters:
        state_transition_days (pd.Series): Days of state transition for each case.

    Returns:
        kmf (KaplanMeierFitter): KaplanMeierFitter object.
    """
    # Convert state_transition_days to timedeltas
    state_transition_days = pd.to_timedelta(state_transition_days)

    # Create duration-event table
    event = np.ones(len(state_transition_days))
    pdTable = pd.DataFrame(
        {'duration': state_transition_days.dt.days, 'event': event})

    # Fit Kaplan-Meier model
    kmf = KaplanMeierFitter()
    kmf.fit(pdTable['duration'], pdTable['event'])

    return kmf


def state_transition_plot(state_transition_days, source_state, target_state, save_fig=False):
    start_time = min(0, state_transition_days.min().days)
    end_time = state_transition_days.max().days
    cumulative_states_transition = np.ones((end_time-start_time)+2) *\
        len(state_transition_days[state_transition_days.notna()])
    for i in state_transition_days[state_transition_days.notna()]:
        cumulative_states_transition[i.days+np.abs(start_time)+1::] -= 1
    # plt.stairs(cumulative_states_transition/cumulative_states_transition[0],
    #            edges=np.arange(start_time, end_time+3)-1, linewidth=2)

    kmf = km_plot_transform(state_transition_days)
    plt.plot()
    fig = kmf.plot(label='%s to %s (%s)' % (
        source_state, target_state, int(cumulative_states_transition[0])))
    plt.legend(prop={'size': 20})
    plt.xlim(start_time-1, 81)
    plt.xlabel('Day', fontsize=22)
    plt.xticks(np.arange(start_time, 81, 10), fontsize=22)
    plt.ylabel('Proportion of cases', fontsize=22)
    plt.yticks(np.arange(0, 1+0.2, 0.2), fontsize=22)
    plt.grid()
    plt.minorticks_on()

    if save_fig == True:
        plt.savefig('RW2022_%s_to_%s.pdf' % (source_state, target_state))

    return (cumulative_states_transition)
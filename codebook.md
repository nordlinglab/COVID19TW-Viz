# Individual Data
1. **id**: Integer, Unique identifier for each individual.
2. **abroad/local**: String, Indicates whether the infection was contracted abroad or locally.
3. **infection_source**: String, Country names indicating the source of infection if known.
4. **visited_countries/cities**: String, List of countries/cities visited.
5. **date_of_departure_1**: Date (ISO 8601), Date of departure from Taiwan.
6. **date_of_arrival_1**: Date (ISO 8601), Date of first arrival.
7. **visited_country/city_1**: String, First country/city visited.
8. **date_of_departure_2**: Date (ISO 8601), Date of second departure.
9. **date_of_arrival_2**: Date (ISO 8601), Date of second arrival.
10. **visited_country/city_2**: String, Second country/city visited.
11. **date_of_departure_3**: Date (ISO 8601), Date of third departure.
12. **date_of_arrival_3**: Date (ISO 8601), Date of third arrival.
13. **visited_country/city_3**: String, Third country/city visited.
14. **date_of_departure_4**: Date (ISO 8601), Date of fourth departure.
15. **date_of_arrival_4**: Date (ISO 8601), Date of fourth arrival.
16. **visited_country/city_4**: String, Fourth country/city visited.
17. **date_of_transit**: Date (ISO 8601), Date of transit.
18. **transit_country/city**: String, Transit country/city.
19. **date_of_arrival_to_taiwan**: Date (ISO 8601), Date of arrival in Taiwan.
20. **nationality**: String, Nationality of the individual.
21. **gender**: String, Gender of the individual.
22. **age**: Integer, Age of the individual.
23. **onset_of_symptom**: Date (ISO 8601), Date when symptoms first appeared.
24. **report_to_cdc**: Date (ISO 8601), Date of reporting to CDC.
25. **confirmed_date**: Date (ISO 8601), Date of COVID-19 confirmation.
26. **icu**: Date (ISO 8601), Date of entering ICU.
27. **recovery**: Date (ISO 8601), Date of recovery.
28. **death_date**: Date (ISO 8601), Date of death if applicable.
29. **symptom**: String, Symptoms experienced by the individual.
30. **way_of_discover**: String, How the case was discovered.
31. **date_of_contact_with_infected_case**: Date (ISO 8601), Date of contact with an infected case.
32. **source_infected_cases_all**: String, All source cases of infection.
33. **source_infected_case**: String, Specific source case of infection.
34. **earliest_infection_date**: Date (ISO 8601), Earliest known date of infection.
35. **negative_test_date_1**: Date (ISO 8601), Date of first negative test.
36. **negative_test_date_2**: Date (ISO 8601), Date of second negative test.
37. **negative_test_date_3**: Date (ISO 8601), Date of third negative test.
38. **negative_test_date_4**: Date (ISO 8601), Date of fourth negative test.
39. **detail**: String, Detailed description or notes regarding the case.
40. **source**: String, Source of data or additional context.

# Summary
1. **announce_date**: Date (ISO 8601), The date of the official announcement.
2. **number_of_suspected_case_repoted_to_cdc**: Integer, Total number of suspected cases reported to CDC on the given date.
3. **number_of_excluded_cases**: Integer, Total number of cases excluded from consideration as COVID-19 infections on the given date.
4. **number_of_abroad_positive_cases**: Integer, Total positive cases in individuals arriving in Taiwan from abroad.
5. **number_of_local_positive_cases**: Integer, Total number of positive cases identified locally on the given date.
6. **number_of_positive_cases_from_panshi_ship**: Integer, Number of positive cases identified from the Panshi fast combat support ship.
7. **number_of_unknown_positive_cases**: Integer, Number of positive cases where the source of infection is unknown.
8. **dead**: Integer, Total number of deaths related to COVID-19 reported on the given date.
9. **recovery**: Integer, Total number of recoveries from COVID-19 reported on the given date.
10. **hospital__quarantine_reported**: Integer, Number of hospital or quarantine reports submitted on the given date.
11. **note**: String, Additional notes or clarifications for the data provided.
12. **reference**: String, Reference or source of the data.

# Specimen
1. **announce_date**: Date (ISO 8601), The date of the announcement or report.
2. **notifications_of_infectious_diseases**: Integer, Total number of notifications received for infectious diseases.
3. **home_quarantine_and_inspection**: Integer, Total number of specimens collected from individuals under home quarantine.
4. **expanded_monitoring**: Integer, Total number of specimens collected under expanded monitoring efforts.
5. **total**: Integer, Total number of specimens collected on the given date.

# Edge list
1. **source**: String, Case ID of the source individual. IDs starting with 'I' denote positive COVID-19 cases; other IDs indicate negative (uninfected) cases.
2. **target**: String, Case ID of the contact individual. Follows the same ID conventions as the source.
3. **interaction**: Integer, Type of contact between the source and target, encoded as follows:
   - 0: Couple
   - 1: Parent/Child
   - 2: Grandparent/Grandchild
   - 3: Brother/Sister
   - 4: Family
   - 5: Friend
   - 6: Live together
   - 7: The same flight
   - 8: The same flight nearby seat
   - 9: Travel together
   - 10: The same school
   - 11: The same car
   - 12: The same hotel
   - 13: The same quarantine hotel
   - 14: Coworker
   - 15: The same hospital
   - 16: Panshi fast combat support ship
   - 17: Coral Princess
   - 18: Other unknown contact
4. **directed**: Boolean, True if there is a directional link from the source to the target, indicating the direction of potential transmission.

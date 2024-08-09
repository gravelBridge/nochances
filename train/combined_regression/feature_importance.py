from inference import attribute_features, college_data
import streamlit as st
import pandas as pd
import altair as alt
import torch
import random

dataset_path = 'train/combined_regression/shortened_numerical_fake_data_demographicless.pt'
dataset = torch.load(dataset_path)
attribute_labels = [
    "Income",
    "GPA",
    "Log(# AP/IBs + 1)",
    "AP/IB Scores",
    "Test Scores",
    "Location",
    "First Gen",
    "Natl/Intl ECs",
    "Regional ECs",
    "Local ECs",
    "Volunteering",
    "Entrepre",
    "Intern",
    "Add. Classes",
    "Research",
    "Sports",
    "Work",
    "Leadership",
    "Community Impact",
    "EC Years",
    "Intl. Award",
    "Natl. Award",
    "State Award",
    "Local Award",
    "Other Award",
    "In State",
    "Admit. Round",
    "EC 1",
    "EC 2",
    "EC 3",
    "EC 4",
    "EC 5",
    "EC 6",
    "EC 7",
    "EC 8",
    "EC 9",
    "EC 10",
    "Award 1",
    "Award 2",
    "Award 3",
    "Award 4",
    "Award 5"
]

constructed_dataset = random.sample(dataset.data, 512)

def attribute_features_school(school_id, major_id):
    college_information = college_data.iloc[school_id]
    for row in constructed_dataset:
        row[27] = float(college_information['Applicants total']/college_information['Admissions total'])
        row[28] = float(college_information['SAT Critical Reading 75th percentile score'])
        row[29] = float(college_information['SAT Math 75th percentile score'])
        row[31] = college_information['combined'][major_id]
        row[32] = major_id

    test_loader = torch.utils.data.DataLoader(constructed_dataset, batch_size=128, num_workers=2)
    for _, batch in enumerate(test_loader):
        inputs = batch['inputs']
        continue

    attributes = attribute_features(inputs)
    flattened_attributes = [0 for _ in range(len(attributes[0]))]
    for row in attributes:
        for i in range(len(row)):
            flattened_attributes[i] += abs(float(row[i]))
    
    return(flattened_attributes)

major_options = ["Engineering / Applied Sciences",
                 "Computer Science / Information Systems",
                 "Business / Marketing / Finance",
                 "Social Sciences",
                 "Biology / Neuroscience",
                 "Math / Data Science / Statistics",
                 "Physical Sciences",
                 "Education",
                 "Communications / Journalism",
                 "Humanities: History, Psychology, English, Group Studies, etc.",
                 "Fine Arts / Performing Arts",
                 "Languages / Linguistics"
                 ]
major_selection = major_options.index(st.selectbox('Intended Major Type', options=major_options))

school_options = college_data['Name']
school_selection = college_data.index[college_data['Name'] == st.selectbox('College Name', options=school_options)].tolist()[0]

if st.button("Get Feature Importance"):
    attributions = attribute_features_school(school_selection, major_selection)
    activities_attributions = [
        sum(attributions[32+15*i:32+15*(i+1)]) for i in range(10)
    ] + [
        sum(attributions[182+10*i:182+10*(i+1)]) for i in range(5)
    ]
    attributions = attributions[:32] + activities_attributions

    freq_attributions = [(attribution * 100)/sum(attributions) for attribution in attributions]

    freq_attributions.pop(31)
    freq_attributions.pop(30)
    freq_attributions.pop(28)
    freq_attributions.pop(27)
    freq_attributions.pop(26)

    attributions = pd.DataFrame({
        "Feature": attribute_labels,
        "Prediction Contribution, %": freq_attributions,
    })
    st.write(alt.Chart(attributions).mark_bar().encode(
        x="Prediction Contribution, %",
        y=alt.X("Feature", sort=None)
    ).properties(width=700))
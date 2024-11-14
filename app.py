import streamlit as st
import pandas as pd
import json
from openai import OpenAI
from dotenv import dotenv_values
from pycaret.clustering import setup, create_model, assign_model, save_model, load_model, predict_model
import plotly.express as px
import os

st.set_page_config(page_title="Find Friends",page_icon=":boy:")

env = dotenv_values(".env")

if 'OPENAI_API_KEY' in st.secrets:
    env['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
 
openai_client = OpenAI(api_key=env["OPENAI_API_KEY"])

##################################################################

MODEL_NAME = 'Data/welcome_survey_clustering_pipeline_v2'
DATA = 'Data/welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'Data/welcome_survey_cluster_names_and_descriptions_v2.json'

##################################################################

@st.cache_data
def data_read(data):
    return pd.read_csv(data, sep=';')

def model_save(data):
    setup(data, session_id=123)
    kmeans = create_model('kmeans', num_clusters=8)
    #df_with_clusters = assign_model(kmeans)
    save_model(kmeans, MODEL_NAME, verbose=False)

def model_load(data):
    return load_model(data)

def prompt_create(data):
    cluster_descriptions = {}
    for cluster_id in data['Cluster'].unique():
        cluster_df = data[data['Cluster'] == cluster_id]
        summary = ""
        for column in data:
            if column == 'Cluster':
                continue

            value_counts = cluster_df[column].value_counts()
            value_counts_str = ', '.join([f"{idx}: {cnt}" for idx, cnt in value_counts.items()])
            summary += f"{column} - {value_counts_str}\n"

        cluster_descriptions[cluster_id] = summary
    
    prompt = "Użyliśmy algorytmu klastrowania."
    for cluster_id, description in cluster_descriptions.items():
        prompt += f"\n\nKlaster {cluster_id}:\n{description}"

    prompt += """
    Wygeneruj najlepsze nazwy dla każdego z klasterów oraz ich opisy

    Użyj formatu JSON. Przykładowo:
    {
        "Cluster 0": {
            "name": "Klaster 0",
            "description": "W tym klastrze znajdują się osoby, które..."
        },
        "Cluster 1": {
            "name": "Klaster 1",
            "description": "W tym klastrze znajdują się osoby, które..."
        }
    }
    """
    return prompt

def desc_create(p):
    response = openai_client.chat.completions.create(
    model="gpt-4o",
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": [{"type": "text", "text": p}],
        }
    ],
    )
    result = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
    cluster_names_and_descriptions = json.loads(result)
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "w") as f:
        f.write(json.dumps(cluster_names_and_descriptions))


@st.cache_data
def desc_load(file):
    with open(file, "r", encoding='utf-8') as f:
        return json.loads(f.read())

###################################################################

df = pd.DataFrame()
df_with_clusters = pd.DataFrame()
kmeans_pipeline = None
desc = None

if os.path.exists(DATA):
    #st.toast(f"Plik Danych istnieje.")
    df = data_read(DATA)
    if not os.path.exists(MODEL_NAME + ".pkl"):
        st.toast(f"Plik Modelu nie istnieje.")
        model_save(df)
        st.toast(f"Plik Modelu utworzony.")
    
    kmeans_pipeline = model_load(MODEL_NAME)
    df_with_clusters = predict_model(kmeans_pipeline, data=df)
    
    if not os.path.exists(CLUSTER_NAMES_AND_DESCRIPTIONS):
        st.toast(f"Plik Klastrów nie istnieje.")
        prompt = prompt_create(df_with_clusters)
        desc_create(prompt)
        st.toast(f"Plik Klastrów utworzony.")
    desc = desc_load(CLUSTER_NAMES_AND_DESCRIPTIONS)
else:
    st.toast(f"Plik Danych nie istnieje.")


###################################################################

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

model = kmeans_pipeline
all_df = df_with_clusters
cluster_names_and_descriptions = desc

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.write(f"Najbliżej Ci do grupy:")
st.header(f"{predicted_cluster_data['name']}")
st.markdown(f"*{predicted_cluster_data['description']}*")
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
col0, col1 = st.columns(2)
with col0:
    st.metric("Liczba wszystkich", len(df))
with col1:
    st.metric("Liczba twoich znajomych", len(same_cluster_df))

st.header("Osoby z grupy")
fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

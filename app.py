import os
import openai
import streamlit as st
from langchain import LLMChain
from dotenv import load_dotenv
from transformers import pipeline
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


load_dotenv()
openai.api_key =  st.secrets["OPENAI_API_KEY"]
HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGING_FACE_API_KEY"]


def image2text(url):
    image_to_text = pipeline("image-to-text", model = "Salesforce/blip-image-captioning-base", max_new_tokens = 100)

    text = image_to_text(url)[0]["generated_text"]
    print(text)

    return text



story_schema = ResponseSchema(name = "story",
                              description = "generated story HERE")

response_schema = [story_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schema)
format_instructions = output_parser.get_format_instructions()


def generate_story(scenario, language, theme, temperature, person_name, person_role):
    prompt_template = """
    You are a storyteller.
    You can generate a short story based on a simple narrative. 
    The story should be maximum of 200 words.
    The theme of the story must be {theme}. 
    the context of the story must be {scenario}.
    The name of the person must be {person_name} as a {person_role}
    the language of the story must be {language}.
    the person names in the story must be {language}.
    
    {format_instructions}
    """
    llm_model = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = temperature, max_tokens = 500)
    input_variables = ["scenario", "language", "theme", "person_name", "person_role"]
    prompt = ChatPromptTemplate(messages = [HumanMessagePromptTemplate.from_template(prompt_template)],
                                input_variables = input_variables,
                                template = prompt_template,
                                partial_variables = {"format_instructions": format_instructions})

    story_llm = LLMChain(llm = llm_model,
                         prompt = prompt,
                         output_parser = output_parser,
                         verbose=True)

    story = story_llm.predict_and_parse(scenario = [scenario],
                                        language = [language],
                                        theme = [theme],
                                        person_name = [person_name],
                                        person_role = [person_role])

    return story.get("story")


def main():
    st.set_page_config(page_title = "img 2 audio story", page_icon = "🤖")
    st.header("TURN YOUR IMAGE INTO A STORY")

    language = st.selectbox("Select Language for the Story", ["English", "Turkish", "French", "Spanish"])
    theme = st.selectbox("Select a Theme for the Story", ["Happy", "Love", "Hate", "Sad", "Fun"])


    person_name = st.text_input("Enter the Name of the Charachter 👇")
    person_role = st.text_input("What is Going to be the Role of that Charachter (Mom 🤷🏼‍♀️ ? Dad 🙅🏼‍♂️? Parrot🦜 ? )👇")

    temperature = (st.slider("Select Creativity Level of Your Story", 0, 100, 50)) / 100
    uploaded_file = st.file_uploader("Choose an image...", type = "jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()

        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)

        st.image(uploaded_file, caption = "Uploaded Image.", use_column_width = True)
        scenario = image2text(uploaded_file.name)
        story = generate_story(scenario, language, theme, temperature, person_name, person_role)

        with st.container():
            st.subheader("Text From the Image")
            st.write(scenario)

        with st.container():
            st.subheader("Generated Story")
            st.write(story)

if __name__ == "__main__":
    main()



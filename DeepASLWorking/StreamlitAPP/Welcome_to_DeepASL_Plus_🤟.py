import streamlit as st

#These are for checking and change working directory (for images)
# import os
# st.write("Current working directory:", os.getcwd())
# # os.environ["STREAMLIT_CONFIG_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml") 
# os.chdir(os.path.dirname(os.path.abspath(__file__)))



# Home page title
st.markdown("# DeepASL Plus 🤟")
st.sidebar.markdown("# About DeepASL Plus 🤟")
st.sidebar.caption("DeepASL Plus is an American Sign Language interpretation project based on Cesar Almendarez's DeepASL (https://github.com/cesarealmendarez/DeepASL) for learning Convolutional Neural Network model training and building the user interface with Python.")
st.subheader(":rainbow[Try out American Sign Language in real time!]")

# 1st paragraph

st.divider()


st.subheader("☺️What is :orange[American Sign Language (ASL)]?")
st.caption("American Sign Language (ASL) is a natural language that serves as the predominant sign language of Deaf communities in the United States and most of Anglophone Canada. ASL is a complete and organized visual language that is expressed by employing both manual and nonmanual features.🤙")

st.image("./streamlitImage/ASL26Letter10Digit.png", caption="The 26 letters and 10 digits of American Sign Language (ASL).", use_column_width=True)
st.markdown("Image Source: _https://www.researchgate.net/figure/The-26-letters-and-10-digits-of-American-Sign-Language-ASL_fig1_328396430_", unsafe_allow_html=True)

# 2nd paragraph

st.divider()

st.subheader("🧐How did :orange[ASL] develop?")
st.caption("American Sign Language (ASL) developed in the early 19th century at the American School for the Deaf (ASD), which was founded in 1817 by Thomas Gallaudet. The language emerged from a unique blend of different signing systems, particularly Old French Sign Language (LSF, from langue des signes française), various village sign languages, and home sign systems brought by the diverse group of students attending the school. This language contact between multiple signing traditions played a crucial role in the creation of ASL.")

st.image("./streamlitImage/ASLhistory.jpg", caption="Deaf and dumb children of St. Rica's School, Cincinnati, singing the Star Spangled Banner with sign language, 1918.", use_column_width=True)
st.markdown("Image Source: _https://loc.getarchive.net/media/deaf-and-dumb-children-of-st-ricas-school-cincinnati-singing-star-spangled_", unsafe_allow_html=True)

# 3rd paragraph

st.divider()

st.subheader("🤔Why are :orange[sign languages] important?")
st.caption("American Sign Language (ASL) developed in the early 19th century at the American School for the Deaf (ASD), which was founded in 1817 by Thomas Gallaudet. The language emerged from a unique blend of different signing systems, particularly Old French Sign Language (LSF, from langue des signes française), various village sign languages, and home sign systems brought by the diverse group of students attending the school. This language contact between multiple signing traditions played a crucial role in the creation of ASL.")

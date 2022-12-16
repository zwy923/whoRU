import streamlit as st

def main():
    # Define UI elements in the sidebar
    st.sidebar.text("Sidebar")
    st.sidebar.button("Button")

    # Define main content
    st.title("Main content")
    st.text("Hello, world!")

if __name__ == "__main__":
    main()

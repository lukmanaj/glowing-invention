import streamlit as st
import subprocess
import os
import numpy as np
import pandas as pd
import tempfile



def run_aligner():
    st.title("Cohere-Parallel-Language-Sentence-Alignment")

    # getting the API key
    cohere_api_key = "lT27lL4uyB6e8KnTl9tRUDjF1TMksqDWKL8oDjHU"

    # Upload source and target files
    src_file = st.file_uploader("Upload source file", type=["txt"])
    trg_file = st.file_uploader("Upload target file", type=["txt"])

    # Run the aligner and display the output
    if st.button("Align"):
        if src_file is None or trg_file is None:
            st.warning("Please upload both source and target files.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save the uploaded files to the temporary directory
                src_file_path = os.path.join(tmpdir, "src.txt")
                with open(src_file_path, "wb") as f:
                    f.write(src_file.read())
                trg_file_path = os.path.join(tmpdir, "trg.txt")
                with open(trg_file_path, "wb") as f:
                    f.write(trg_file.read())

                # Set the path for the output file in the temporary directory
                output_file_path = os.path.join(tmpdir, "output.csv")

                # Run the aligner command
                command = [
                    "python3",
                    "-u",
                    "cohere_align.py",
                    "--cohere_api_key", cohere_api_key,
                    "-m", "embed-multilingual-v2.0",
                    "-s", src_file_path,
                    "-t", trg_file_path,
                    "-o", output_file_path,
                    "--retrieval", "nn",
                    "--dot",
                    "--cuda"
                ]
                try:
                    result = subprocess.run(command, capture_output=True, cwd="/app/cohere-project-test", text=True)
                except Exception as e:
                    st.error(f"Error running the aligner command: {e}")
                    return

                # Check if the command was successful and display the output
                if result.returncode == 0:
                    # Load the output file into a pandas dataframe
                    output_df = pd.read_csv(output_file_path)
                    st.dataframe(output_df)
                    
                    # Allow the user to download the output file as a text file
                    st.download_button(
                        "Download Output",
                        output_df.to_csv(index=False),
                        file_name="output.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Error running the aligner command. stdout: {result.stdout}, stderr: {result.stderr}")
# run the aligner function
run_aligner()

import os
import subprocess
import json
import openai
import re
import random
import pandas as pd
import sys
import csv
import streamlit as st
from io import StringIO

# Get ExifTool path
def get_exiftool_path():
    # Try to find exiftool in common paths
    possible_paths = [
        "/usr/bin/exiftool",
        "/usr/local/bin/exiftool",
        "exiftool",  # Assuming it's in the PATH
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# Run ExifTool to extract metadata
def run_exiftool(image_path):
    exiftool_path = get_exiftool_path()
    if exiftool_path is None:
        st.error("ExifTool not found. Please ensure ExifTool is installed and in the system PATH.")
        return None
    try:
        process = subprocess.Popen(
            [exiftool_path, '-j', image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
    except Exception as e:
        st.error(f"Exception running exiftool: {str(e)}")
        return None
    if process.returncode != 0:
        st.error("Error running exiftool: " + stderr.decode())
        return None
    try:
        metadata = json.loads(stdout.decode())
        # Extract description from relevant tags
        description = metadata[0].get('Description') or metadata[0].get('ImageDescription') or metadata[0].get('Title') or ''
        return description.strip()
    except json.JSONDecodeError:
        st.error("Failed to decode JSON output from exiftool.")
        return None

# Initialize OpenAI API key
def init_openai_api_key(api_key):
    openai.api_key = api_key

# Call GPT API for concise names and detailed context
def call_gpt_api(prompt, temperature=0.6, model="gpt-4o-mini", max_tokens=77):
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates metadata."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        response = openai.ChatCompletion.create(**params)
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return ""

# Helper functions (clean_metadata, clean_numbers_and_periods, etc.)
def clean_metadata(description):
    description = re.split(r'Job ID:|--\S*', description)[0].strip()
    description = description.replace('"', '').replace("'", '')
    return description

def clean_numbers_and_periods(keywords):
    cleaned_keywords = []
    for keyword in keywords:
        cleaned_keyword = re.sub(r'\d+|\.', '', keyword).strip()
        if cleaned_keyword:
            cleaned_keywords.append(cleaned_keyword)
    return cleaned_keywords

def remove_duplicate_keywords(keywords_list):
    # Convert to lowercase for case-insensitive comparison
    lowercase_keywords = [keyword.lower() for keyword in keywords_list]
    # Use a dictionary to preserve order while removing duplicates
    unique_keywords = list(dict.fromkeys(lowercase_keywords))
    # Map back to original case
    original_case_keywords = [next(kw for kw in keywords_list if kw.lower() == unique_kw) for unique_kw in unique_keywords]
    return original_case_keywords

def generate_keywords_with_retries(description, keywords_rule, min_keywords=40, max_keywords=49, retries=3):
    keywords_set = set()

    for _ in range(retries):
        keywords_prompt = f"{keywords_rule}\nDescription: {description}\nExisting keywords: {', '.join(keywords_set)}\nGenerate only new, unique keywords."
        keywords = call_gpt_api(keywords_prompt).strip()
        keywords_list = [k.strip() for k in keywords.split(',')]

        # Ensure all keywords are strings
        keywords_list = clean_numbers_and_periods(keywords_list)
        keywords_list = [str(k) for k in keywords_list]

        # Add only new keywords to the set
        new_keywords = [k for k in keywords_list if k.lower() not in {kw.lower() for kw in keywords_set}]
        keywords_set.update(new_keywords)

        if len(keywords_set) >= min_keywords:
            break

    # Convert set to list and sort (reverse) the keywords
    keywords_list = list(keywords_set)
    keywords_list.sort(key=lambda x: x.lower(), reverse=True)

    # Ensure we have the correct number of keywords
    if len(keywords_list) > max_keywords:
        keywords_list = random.sample(keywords_list, max_keywords)
    elif len(keywords_list) < min_keywords:
        # If we're still short on keywords, we might need to generate more
        pass

    # Final deduplication step
    keywords_list = remove_duplicate_keywords(keywords_list)

    return ', '.join(keywords_list)

def generate_concise_names(description):
    concise_names_rule = (
         "Generate a concise name and detailed context for the input, within 150 characters. No copyright, No colon (:), in the style of others, trademarks, privacy rights, property rights, no number, no ensuring organization and clarity. "
         "Avoid photography-related words like realistic, natural lighting, photography, etc. "
         "must not answer with real place name, use some thing to describe instead for example instead eiffel, we use tower "
         "No quotation marks or dashes, using commas for separation. Focus on straightforward, richly descriptive titles without vague language or mentioning camera specifics or photography techniques. "
         "Ensure the response is a single line."
    )
    concise_names_prompt = f"{concise_names_rule}\nDescription: {description}"
    result = call_gpt_api(concise_names_prompt, temperature=0.6, max_tokens=77).strip()
    result = result.replace('Title:', '').strip()
    result = result.replace('Name:', '').strip()
    result = result.replace('"', '')
    result = result.replace('\n', ' ')
    return result

def generate_metadata(description, use_concise_names):
    # Rule for generating keywords
    keywords_rule = (
        "Generate 40-49 single-word keywords for a microstock image based on the provided title. Ensure the first ten keywords are directly derived from the title, and the rest are related concepts. Avoid plurals, and ensure keywords are relevant to the title. Format with commas."
        "First 10 keywords should come from the title, and the remaining should relate to it. Force the result to reach 49 keywords."
    )

    # Generate title from description
    title = generate_concise_names(description) if use_concise_names else clean_metadata(description)
    title = replace_newlines_except_3d(title)

    # Generate keywords using the title
    keywords = generate_keywords_with_retries(title, keywords_rule)

    # Create metadata dictionary
    metadata = {'title': title, 'keywords': keywords}
    return metadata

def categorize_title(title):
    categories = [
        "1. Animals", "2. Buildings and Architecture", "3. Business",
        "4. Drinks", "5. The Environment", "6. States of Mind",
        "7. Food", "8. Graphic Resources", "9. Hobbies and Leisure",
        "10. Industry", "11. Landscape", "12. Lifestyle",
        "13. People", "14. Plants and Flowers", "15. Culture and Religion",
        "16. Science", "17. Social Issues", "18. Sports",
        "19. Technology", "20. Transport", "21. Travel"
    ]
    categories_prompt = (
        "Given the following categories, choose the most appropriate one for the title provided. If it's something related to Culture and Religion, make it Culture and Religion priority."
        "Respond with only the category number.\n"
        f"Categories:\n{categories}\n\nTitle: {title}"
    )
    response = call_gpt_api(categories_prompt, temperature=0.3)
    match = re.search(r'\b(\d+)\b', response)
    if match:
        return int(match.group(1))
    return None

def replace_newlines_except_3d(text):
    # Use a regex to replace newlines except those followed by '3D' or '3d'
    result = re.sub(r'\n(?!3[Dd])', ' ', text)
    return result

def generate_metadata_for_images(directory, api_key):
    init_openai_api_key(api_key)
    metadata_list = []
    files = [f for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not files:
        st.warning("No image files found in the directory.")
        return None
    for i, filename in enumerate(files):
        file_path = os.path.join(directory, filename)
        description = run_exiftool(file_path)
        if description:
            # Generate metadata including title, keywords, and category
            st.info(f"Processing {filename}...")
            metadata = generate_metadata(description, use_concise_names=True)
            title = metadata['title']
            keywords = metadata['keywords']
            category = categorize_title(title)
            metadata_list.append({'Filename': filename, 'Title': title, 'Keywords': keywords, 'Category': category})
        else:
            st.warning(f"Could not extract description from {filename}.")
    return metadata_list

# Main function to run everything
def main():
    st.title("Title Keyworder By Gasia AI")
    st.write("This app generates metadata (title, keywords, category) for images in a folder.")

    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    directory = st.text_input("Enter the directory containing images:")

    if st.button("Generate Metadata"):
        if not api_key:
            st.error("Please enter your OpenAI API key.")
        elif not directory:
            st.error("Please enter the directory containing images.")
        else:
            if not os.path.isdir(directory):
                st.error("The specified directory does not exist.")
            else:
                # Proceed to generate metadata
                with st.spinner("Generating metadata..."):
                    metadata_list = generate_metadata_for_images(directory, api_key)
                if metadata_list:
                    # Convert metadata_list to DataFrame
                    df = pd.DataFrame(metadata_list)
                    # Save CSV to a buffer
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    st.success("Metadata generation complete.")
                    # Provide download link
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="image_metadata.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No images found or failed to generate metadata.")

# Run the script
if __name__ == "__main__":
    main()

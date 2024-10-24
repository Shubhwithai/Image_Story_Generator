import streamlit as st
import time
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO

class RateLimitHandler:
    def __init__(self):
        self.last_request_time = 0
        self.min_delay = 2
        self.max_delay = 30
        self.base_delay = 5
        self.max_retries = 3

    def wait(self):
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed + random.uniform(0.1, 1.0))
        
        self.last_request_time = time.time()

    def handle_rate_limit(self, attempt: int):
        if attempt >= self.max_retries:
            raise Exception("Max retries exceeded")
        
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        delay += random.uniform(0.1, 2.0)
        
        st.warning(f"Rate limit hit. Waiting {delay:.2f} seconds before retry...")
        time.sleep(delay)

rate_limiter = RateLimitHandler()

def initialize_together_client() -> OpenAI:
    return OpenAI(
        api_key=st.secrets["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1"
    )

def safe_api_call(func, *args, **kwargs):
    for attempt in range(rate_limiter.max_retries):
        try:
            rate_limiter.wait()
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) and attempt < rate_limiter.max_retries - 1:
                rate_limiter.handle_rate_limit(attempt)
                continue
            raise e
    raise Exception("Max retries exceeded")

def generate_story_prompts(client: OpenAI, topic: str) -> Dict[str, str]:
    prompt = {
        "role": "user",
        "content": f"""Create 3 story lines and image prompts about: {topic}
        Return only a JSON object with this structure:
        {{
            "story_line_1": "image_prompt_1",
            "story_line_2": "image_prompt_2",
            "story_line_3": "image_prompt_3"
        }}"""
    }

    def make_request():
        response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=[prompt]
        )
        return json.loads(response.choices[0].message.content)

    return safe_api_call(make_request)

def generate_image(client: OpenAI, prompt: str) -> str:
    def make_request():
        response = client.images.generate(
            model="black-forest-labs/FLUX.1-schnell-Free",
            prompt=prompt,
        )
        return response.data[0].url

    return safe_api_call(make_request)

def generate_story(client: OpenAI, image_prompt: str, story_line: str) -> str:
    prompt = f"""Write a short story (100 words) that combines these elements:
    1. Scene description: {image_prompt}
    2. Story line: {story_line}
    Make the story vivid and descriptive, as if describing a scene from a painting."""

    def make_request():
        response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    return safe_api_call(make_request)

def create_multi_story_app(client: OpenAI, topic: str, progress_bar) -> List[Tuple[str, str, str]]:
    try:
        progress_bar.progress(0.1, "Generating story prompts...")
        story_prompts = generate_story_prompts(client, topic)
        
        results = []
        for i, (story_line, image_prompt) in enumerate(story_prompts.items(), 1):
            try:
                # Adjust progress calculations for 3 stories
                progress_value = 0.1 + (i-1)*0.3
                progress_bar.progress(progress_value, f"Generating image {i}...")
                image_url = generate_image(client, image_prompt)

                progress_bar.progress(progress_value + 0.1, f"Generating story {i}...")
                story = generate_story(client, image_prompt, story_line)

                results.append((image_url, story_line, story))

            except Exception as e:
                st.error(f"Error processing story {i}: {str(e)}")
                continue

        progress_bar.progress(1.0, "Complete!")
        return results

    except Exception as e:
        st.error(f"Error in story generation process: {str(e)}")
        return []

def display_story(image_url: str, story_line: str, story: str, index: int):
    st.subheader(f"Story {index}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption=f"Generated Image {index}", use_column_width=True)
        except Exception as e:
            st.error(f"Unable to display image. Error: {str(e)}")
            st.write(f"Image URL: {image_url}")
    
    with col2:
        st.write("**Story Line:**")
        st.write(story_line)
        st.write("**Story:**")
        st.write(story)

def main():
    st.set_page_config(page_title="AI Story Generator", layout="wide")
    
    st.title("AI Story Generator")
    st.write("Generate three unique stories with AI-generated images based on your topic!")

    # Topic input
    topic = st.text_input("Enter a topic for your stories:", 
                         placeholder="e.g., A magical forest where animals play musical instruments")

    if st.button("Generate Stories") and topic:
        try:
            client = initialize_together_client()
            progress_bar = st.progress(0)
            
            with st.spinner('Generating your stories and images... This may take a few minutes.'):
                results = create_multi_story_app(client, topic, progress_bar)
            
            if results:
                st.success("Stories generated successfully!")
                for i, (image_url, story_line, story) in enumerate(results, 1):
                    st.markdown("---")
                    display_story(image_url, story_line, story, i)
            else:
                st.warning("No stories were generated. Please try again.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("\nTroubleshooting tips:")
            st.write("1. Check if the secrets.toml file is properly configured")
            st.write("2. Wait a few minutes if you've hit rate limits")
            st.write("3. Try generating fewer stories if the issue persists")
    
    st.markdown("---")
    st.markdown("Created with ❤️ By BuildFastWithAI")

if __name__ == "__main__":
    main()

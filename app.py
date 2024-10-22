import streamlit as st
import time
import json
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO

# 1. Rate Limit Handler to manage API calls
class RateLimitHandler:
    def __init__(self):
        self.last_request_time = 0
        self.min_delay = 2  # Minimum seconds between requests
        
    def wait(self):
        """Wait if needed before making next API call"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        
        self.last_request_time = time.time()

# 2. Initialize API Client
def initialize_client():
    return OpenAI(
        api_key=st.secrets["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1"
    )

# 3. Generate Story Prompts
def generate_story_prompts(client: OpenAI, topic: str) -> dict:
    """Generate 3 story ideas with image prompts"""
    rate_limiter = RateLimitHandler()
    rate_limiter.wait()
    
    prompt = {
        "role": "user",
        "content": f"""Create 3 related story lines and image prompts about: {topic}
        Return only a JSON object with this structure:
        {{
            "story_line_1": "image_prompt_1",
            "story_line_2": "image_prompt_2",
            "story_line_3": "image_prompt_3"
        }}"""
    }
    
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[prompt]
    )
    return json.loads(response.choices[0].message.content)

# 4. Generate Image
def generate_image(client: OpenAI, prompt: str) -> str:
    """Generate image from prompt"""
    rate_limiter = RateLimitHandler()
    rate_limiter.wait()
    
    response = client.images.generate(
        model="black-forest-labs/FLUX.1-schnell-Free",
        prompt=prompt,
    )
    return response.data[0].url

# 5. Generate Story
def generate_story(client: OpenAI, image_prompt: str, story_line: str) -> str:
    """Generate story based on image prompt and story line"""
    rate_limiter = RateLimitHandler()
    rate_limiter.wait()
    
    prompt = f"""Write a short story (100 words) that combines these elements:
    1. Scene description: {image_prompt}
    2. Story line: {story_line}
    Make the story vivid and descriptive, as if describing a scene from a painting."""
    
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 6. Display Story
def display_story(image_url: str, story_line: str, story: str, index: int):
    """Display story with image in a nice layout"""
    st.subheader(f"Story {index}")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    # Column 1: Display image
    with col1:
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption=f"Generated Image {index}", use_column_width=True)
        except Exception as e:
            st.error(f"Unable to display image: {str(e)}")
    
    # Column 2: Display story
    with col2:
        st.write("**Story Line:**")
        st.write(story_line)
        st.write("**Story:**")
        st.write(story)

# 7. Main App Function
def main():
    st.set_page_config(page_title="AI Story Generator", layout="wide")
    st.title("AI Story Generator")
    
    # Get user input
    topic = st.text_input(
        "Enter a topic for your stories:", 
        placeholder="e.g., A magical forest where animals play musical instruments"
    )
    
    if st.button("Generate Stories") and topic:
        try:
            # Initialize API client
            client = initialize_client()
            
            # Show progress bar
            progress_bar = st.progress(0)
            
            with st.spinner('Generating stories and images...'):
                # Generate story prompts
                progress_bar.progress(0.1, "Generating story prompts...")
                story_prompts = generate_story_prompts(client, topic)
                
                # Generate each story and image
                for i, (story_line, image_prompt) in enumerate(story_prompts.items(), 1):
                    progress_value = 0.1 + (i-1)*0.3
                    
                    # Generate image
                    progress_bar.progress(progress_value, f"Generating image {i}...")
                    image_url = generate_image(client, image_prompt)
                    
                    # Generate story
                    progress_bar.progress(progress_value + 0.1, f"Generating story {i}...")
                    story = generate_story(client, image_prompt, story_line)
                    
                    # Display story
                    st.markdown("---")
                    display_story(image_url, story_line, story, i)
                
                progress_bar.progress(1.0, "Complete!")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Try again in a few minutes if you've hit rate limits.")

if __name__ == "__main__":
    main()

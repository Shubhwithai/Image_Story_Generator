import streamlit as st
import time
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple
import together

# Initialize Together AI client with API key from secrets
together.api_key = st.secrets["TOGETHER_API_KEY"]

class RateLimitHandler:
    """Handle rate limiting with exponential backoff and request tracking."""
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

def generate_story_prompts(topic: str) -> Dict[str, str]:
    prompt = {
        "role": "user",
        "content": f"""Create 2 related story lines and image prompts about: {topic}
        Return only a JSON object with this structure:
        {{
            "story_line_1": "image_prompt_1",
            "story_line_2": "image_prompt_2"
        }}"""
    }
    
    def make_request():
        response = together.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=[prompt]
        )
        return json.loads(response.choices[0].message.content)
    
    return safe_api_call(make_request)

def generate_image(prompt: str) -> str:
    def make_request():
        response = together.images.generate(
            model="black-forest-labs/FLUX.1-schnell-Free",
            prompt=prompt,
        )
        return response.data[0].url
    
    return safe_api_call(make_request)

def generate_story(image_prompt: str, story_line: str) -> str:
    prompt = f"""Write a short story (100 words) that combines these elements:
    1. Scene description: {image_prompt}
    2. Story line: {story_line}
    Make the story vivid and descriptive, as if describing a scene from a painting."""
    
    def make_request():
        response = together.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    return safe_api_call(make_request)

def create_multi_story_app(topic: str) -> List[Tuple[str, str, str]]:
    results = []
    
    with st.spinner("Generating story prompts..."):
        try:
            story_prompts = generate_story_prompts(topic)
            st.success("Story prompts generated successfully")
            
            for i, (story_line, image_prompt) in enumerate(story_prompts.items(), 1):
                st.write(f"\nProcessing story {i}/2...")
                
                try:
                    with st.spinner(f"Generating image {i}..."):
                        image_url = generate_image(image_prompt)
                        st.success(f"Image {i} generated successfully")
                    
                    with st.spinner(f"Generating story {i}..."):
                        story = generate_story(image_prompt, story_line)
                        st.success(f"Story {i} generated successfully")
                    
                    results.append((image_url, story_line, story))
                    
                except Exception as e:
                    st.error(f"Error processing story {i}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            st.error(f"Error in story generation process: {str(e)}")
            return []

def main():
    st.set_page_config(page_title="Story Generator", page_icon="📚")
    
    st.title("📚 AI Story Generator")
    st.write("Generate unique stories with matching images based on your topic!")
    
    # Input section
    topic = st.text_input("Enter a topic for your stories:", 
                         placeholder="e.g., A magical forest where animals play musical instruments")
    
    num_stories = st.slider("Number of story pairs to generate:", 
                           min_value=1, max_value=3, value=1)
    
    if st.button("Generate Stories", type="primary"):
        if not topic:
            st.warning("Please enter a topic first!")
            return
            
        try:
            for i in range(num_stories):
                st.write(f"\n## Story Set {i+1}")
                results = create_multi_story_app(topic)
                
                if results:
                    for j, (image_url, story_line, story) in enumerate(results, 1):
                        st.subheader(f"Story {j}")
                        st.write(f"**Story Line:** {story_line}")
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.image(image_url, caption="Generated Image", use_column_width=True)
                        with col2:
                            st.write("**Story:**")
                            st.write(story)
                        
                        st.divider()
                else:
                    st.error("No stories were generated successfully. Please try again later.")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("\n**Troubleshooting tips:**")
            st.write("1. Current rate limits are hitting the maximum. Please wait 5-10 minutes before trying again.")
            st.write("2. Consider reducing the number of stories if the issue persists.")
            st.write("3. Check your Together AI account status and limits.")

if __name__ == "__main__":
    main()

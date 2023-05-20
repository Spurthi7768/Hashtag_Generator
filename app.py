import streamlit as st
from transformers import AutoProcessor,BlipForConditionalGeneration,AutoTokenizer
import openai
from itertools import cycle
from tqdm import tqdm
from PIL import Image
import torch
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(".env")

API_KEY=os.getenv("API_KEY")

#Object Creation Model,Tokenizer and Processor from HuggingFace
processor=AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
tokenizer=AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")

#Setting the device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Getting the key from environment
openai.api_key=f"{API_KEY}"
openai_model="text-davinci-002"   #OpenAI Model

def caption_generator(des):
    caption_prompt=('''Please generate two unique and creative captions to use on Instagram for a photo that shows'''
                    +des+'''. The captions should be catchy and innovative.
                    Captions:
                    1. 
                    2. 
                    '''
    )
    #caption generation
    response=openai.Completion.create(
        engine=openai_model,
        prompt=caption_prompt,
        max_tokens=(175*3),
        n=1,
        stop=None,
        temperature=0.7,
    )
    caption=response.choices[0].text.strip().split("\n")
    return (caption)

def hashtag_generator(des):
    #Prompt
    hashtag_prompt=('''Please generate ten relevant and accurate hashtags to use on Instagram for a photo that shows '''
                    + des + '''. The hashtag should be trendy and be used by more than five thousand people.
    Please also provide in this format.
    Hashtags:
    #[Hashtag1]#[Hashtag2]#[Hashtag3]#[Hashtag4]#[Hashtag5]#[Hashtag6]#[Hashtag7]#[Hashtag8]#[Hashtag9]#[Hashtag10]
    ''')
    #Hashtag Generation
    response=openai.Completion.create(
        engine=openai_model,
        prompt=hashtag_prompt,
        max_tokens=(20*10),
        n=1,
        stop=None,
        temperature=0.7,
    )
    hashtag=response.choices[0].text.strip().split("\n")
    return (hashtag)

def prediction(img_list):

    max_length=16
    num_beams=4
    gen_kwargs={"max_length":max_length,"num_beams":num_beams}

    img=[]

    for image in tqdm(img_list):
        i_image=Image.open(image) #Storing of image
        st.image(i_image,width=200) #Display of image

        if i_image.mode!="RGB": #Check if the image is in RGB mode
            i_image=i_image.convert(mode="RGB")
        
        img.append(i_image) #Add image to the list
    
    # Image data to pixel values

    pixel_val=processor(images=img,return_tensors="pt").pixel_values
    pixel_val=pixel_val.to(device)

    #Using model to generate output from the pixel values of image
    output=model.generate(pixel_val,**gen_kwargs)

    #To convert output to text
    predict=tokenizer.batch_decode(output,skip_special_tokens=True)
    predict=[pred.strip() for pred in predict]

    return predict

def upload():

    #Form uploader inside tab

    with st.form("uploader"):
        #Image input
        image=st.file_uploader("Upload Image",accept_multiple_files=True,type=["jpg","png","jpeg"])
        #Generate Button
        submit=st.form_submit_button("Generate")

        if submit: #submit condition
            description=prediction(image)            
            

            st.subheader("Captions")
            captions=caption_generator(description[0]) #Function call to generate captions
            for caption in captions: #Present Captions
                st.write(caption)
            
            st.subheader("Hashtags")
            hashtags=hashtag_generator(description[0]) #Function call to generate hashtags
            for hashtag in hashtags: #Present Hashtags
                st.write(hashtag)


def main():
    #Title on the tab

    st.set_page_config(page_title="Caption and Hashtag Generation")

    #Title of the page

    st.title("Caption and Hashtag Generator")
   
    
    upload() #Upload Images tab
    

if __name__=='__main__':
    main()




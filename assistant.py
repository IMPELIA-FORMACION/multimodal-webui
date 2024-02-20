import io
import os
import shutil
import base64
import numpy as np
import gradio as gr
from PIL import Image
from openai import OpenAI

class ASSITANTWEBUI():

    def __init__(self):
        self.current_exe_path = os.getcwd()
        self.imgs_path = os.path.join(os.getcwd(), "imgs")
        self.audios_path = os.path.join(os.getcwd(), "audios")
        self.client = OpenAI(api_key="your_open_api_key_here")

    def text_response(self, prompt):
        print("[+] Text response")
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"Eres un asistente personal, piensa paso a paso y ayudame en todo lo que necesite porfavor"
                },
                {
                    "role": "user",
                    "content": f"{prompt}"
                }
            ]
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content


    def text_and_img_response(self, image, prompt):
        print("[+] Text with Image response")
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    }
                ],
                }
            ],
            max_tokens=300,
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    
    def process_audio(self, audio):
        print(str(audio))
        try:
            os.remove(self.audios_path+"/audio.wav")
            shutil.move(str(audio), self.audios_path+"/audio.wav")
            audio_file= open("audios/audio.wav", "rb")
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file, 
                response_format="text"
            )
            print(transcript)
            return transcript
        except Exception as e:
            print(e)
            return ""


    ## BASIC USAGE SECTION ##
    def generate_response(self, prompt, image, audio):

        # Process audio from file o micro record
        audio_txt = self.process_audio(audio)
        
        # If image is loaded from webUI
        try:
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG")
                image_str = base64.b64encode(buffered.getvalue()).decode()
                return self.text_and_img_response(image_str, prompt+" "+audio_txt)
            else:
                image_str = image
                return self.text_and_img_response(image_str, prompt+" "+audio_txt)
        except Exception as e:
            return self.text_response(prompt+" "+audio_txt)

    ## LLM UI ##
    def run_webui(self):

        with gr.Blocks(theme=gr.themes.Default()) as demo:
            
            gr.HTML(f"<img src='https://impelia.org/wp-content/uploads/2022/08/impelia_logo-cabecera_-02.png' alt='Logo' style='width:100%;height:100px;object-fit:scale-down;'>")  # Ajusta el width según necesites   
            
            # MODEL BASIC USAGE SECTION
            with gr.Tab("PREGUNTAME"):
                with gr.Row():
                    response_data = gr.TextArea(label="Response", max_lines=50, value="")
                with gr.Row():
                    prompt = gr.Textbox(label="Escribe tu prompt:", placeholder="¿Ves algo raro en esta imagen?", value="¿Ves algo raro en esta imagen?", lines=5)
                with gr.Row():
                    imagen = gr.Image(label="Procesa cualquier imagen", value=self.imgs_path+"/extreme_ironing.jpg")
                    audio = gr.Audio(label="Usa el micro o un archivo de voz", type="filepath")
            generate_response_button = gr.Button("RESPONDE")
            generate_response_button.click(self.generate_response, inputs=[prompt, imagen, audio], outputs=response_data)

        demo.launch(share=True)# share=True / server_port=80
    
# USAGE
if __name__ == '__main__':
    llms_webui = ASSITANTWEBUI()
    llms_webui.run_webui()
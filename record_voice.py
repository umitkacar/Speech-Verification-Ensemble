
import speech_recognition as sr 


mic_name = 'default'
sample_rate = 16000
chunk_size = 2048
#Initialize the recognizer 
r = sr.Recognizer() 
  
mic_list = sr.Microphone.list_microphone_names()
print(mic_list)
  
for i, microphone_name in enumerate(mic_list): 
    if microphone_name == mic_name: 
        device_id = i 
  
with sr.Microphone(device_index = device_id, sample_rate = sample_rate,  
                        chunk_size = chunk_size) as source: 
    
    r.adjust_for_ambient_noise(source) 
    print("Say Something")

    audio = r.listen(source, timeout=15) 
    print("finished") 

    # write audio to a WAV file
    with open("umit4.wav", "wb") as f:
        f.write(audio.get_wav_data())

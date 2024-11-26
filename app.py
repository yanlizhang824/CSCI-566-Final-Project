import time
import cv2
import pyaudio
import wave
import threading
import speech_recognition as sr
import os
from pynput import keyboard

# Flag to control the loop
keep_running = True

# Video settings
video_file = 'output.mp4'
video_codec = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20.0


# Audio settings
audio_file = 'output.wav'
audio_format = pyaudio.paInt16
channels = 1
rate = 44100
chunk = 1024
record_seconds = 5


def ensure_file_exists(file_path):
    """
    Check if a file exists; if not, create an empty file.

    Args:
        file_path (str): Path to the file.

    Returns:
        None
    """
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            pass  # Creates an empty file


def wav_to_text(audio, output_txt_file):
    """
    Convert a WAV audio file to text and save it to a text file.

    Args:
        audio (str): Path to the input WAV file.
        output_txt_file (str): Path to the output text file.

    Returns:
        None
    """
    recognizer = sr.Recognizer()

    try:
        # Load the audio file
        with sr.AudioFile(audio) as source:
            # print("Processing audio...")
            audio_data = recognizer.record(source)  # Read the entire audio file

        # Recognize speech using Google Web Speech API (free)
        print("Converting speech to text...")
        text = recognizer.recognize_google(audio_data, language="zh-CN,en")

        # Save the result to a text file
        with open(output_txt_file, 'w', encoding='utf-8') as file:
            file.write(text)

        print(f"Text Conversion complete.")
    except sr.UnknownValueError:
        print("Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Speech Recognition service; {e}")
    except FileNotFoundError:
        print(f"File not found: {audio}")
    except Exception as e:
        print(f"An error occurred: {e}")

    ensure_file_exists('output.txt')


def record_audio():
    """Function to record audio."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=audio_format, channels=channels,
                        rate=rate, input=True,
                        frames_per_buffer=chunk)
    print("Recording audio...")
    frames = []

    for _ in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("Audio recording complete.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save audio to file
    with wave.open(audio_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(audio_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))


def record_video():
    """Function to record video."""
    camera = cv2.VideoCapture(0)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_file, video_codec, fps, (width, height))

    print("Recording video...")
    start_time = time.time()
    while time.time() - start_time < record_seconds:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture video frame.")
            break
        out.write(frame)
        # cv2.imshow("Recording Video", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):  # Optional manual stop
        #     break

    print("Video recording complete.")
    camera.release()
    out.release()
    cv2.destroyAllWindows()


def save_first_frame(video_file, output_image_file):
    """
    Save the first frame of a video as a JPG file.

    Args:
        video_file (str): Path to the video file.
        output_image_file (str): Path to save the output image file (e.g., 'frame.jpg').

    Returns:
        None
    """
    # Open the video file
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_file}")
        return

    # Read the first frame
    ret, frame = cap.read()

    if ret:
        # Save the frame as a JPG file
        cv2.imwrite(output_image_file, frame)
        print(f"First frame saved as {output_image_file}")
    else:
        print("Error: Unable to read the first frame.")

    # Release the video capture object
    cap.release()



def on_press(key):
    global keep_running
    try:
        # Check if the pressed key is 'q'
        if key.char == 'q':
            print("Key 'q' pressed. Exiting the app.")
            keep_running = False  # Change the flag to exit the loop
    except AttributeError:
        # Handle special keys if necessary
        pass


# Start listening for key presses
listener = keyboard.Listener(on_press=on_press)
listener.start()

print("Press 'q' to exit.")

# Main loop
while keep_running:
    # Run audio and video recording in parallel
    audio_thread = threading.Thread(target=record_audio)
    video_thread = threading.Thread(target=record_video)

    audio_thread.start()
    video_thread.start()

    audio_thread.join()
    video_thread.join()

    wav_to_text('output.wav', 'output.txt')
    save_first_frame("output.mp4", "output.jpg")
    pass
    keep_running = False

# Wait for the listener thread to stop before exiting
listener.stop()
print("Program terminated.")

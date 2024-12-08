from ai2thor.controller import Controller
from leolani_client import Action
import numpy as np
from PIL import Image
import openai
import json
import base64
import requests
import cv2
from sentence_transformers import SentenceTransformer, util

ACTIONS = ["find", "describe", "move", "turn", "head"]
with open('openaikey.txt') as f:
    api_key = f.read().strip()

openai.api_key = api_key

class Ai2ThorClient:
    def __init__(self):
        self._answers = []
        self._actions = []
        self._perceptions = []
        self._controller = Controller()
        self._controller.renderObjectImage = True
        self._controller.agentMode = "arm"
        #self._event = None
        self._event = self._controller.step(action="Initialize")
        self._human_description = ""
        
        self.initialize_human_description()

    def initialize_human_description(self):
        """
        Ask the evaluator to describe what they see in one sentence and store their description.
        """
        print("Robot> Please describe in one sentence what you see in the image shown.")
        human_input = input("Evaluator> ")  # Capture evaluator's response
        self._human_description = human_input.strip()
        print(f"Robot> Thank you! I have stored your description: \"{self._human_description}\"")

    def perform_360_view(self):
        """
        Perform a 360° view from the current position, stitch images into a panorama,
        and return the stitched panorama image path.
        """
        images = []
        for _ in range(4):  # Rotate 4 times, 90° each
            self._event = self._controller.last_event
            images.append(self._event.frame)
            self._controller.step(action="RotateRight", degrees=90)

        # Stitch images into a panorama
        panorama = self.stitch_images(images)

        # Save panorama image
        panorama_path = "room_panorama.jpg"
        cv2.imwrite(panorama_path, panorama)
        return panorama_path

    def stitch_images(self, images):
        """
        Stitch a list of images into a 360° panorama using OpenCV.
        """
        # Convert images to OpenCV format
        cv_images = [cv2.cvtColor(np.array(Image.fromarray(img)), cv2.COLOR_RGB2BGR) for img in images]

        # Use OpenCV's stitcher
        stitcher = cv2.Stitcher_create()
        status, stitched_image = stitcher.stitch(cv_images)

        if status == cv2.Stitcher_OK:
            return stitched_image
        else:
            raise Exception(f"Image stitching failed with status code {status}")
            
    def describe_current_scene(self):
        return main(self._controller)
        
    def getdistance(self, coord1, coord2):
        distance = np.sqrt(
            (coord2['x'] - coord1['x'])**2
            + (coord2['y'] - coord1['y'])**2
            + (coord2['z'] - coord1['z'])**2)
        return distance

    def search_for_object_in_view(self, objectType):
        found = []
        for obj in self._event.metadata['objects']:
            if obj['objectType'].lower() == objectType.lower():
                coord = obj['position']
                image = Image.fromarray(self._controller.last_event.frame)
                found.append((obj, objectType, coord, image))
        return found

    def search_for_object(self, objectType):
        answer = ""
        found = self.search_for_object_in_view(objectType)
        rotate = 0
        while not found and rotate < 4:
            self._event = self._controller.step(Action.RotateRight.name)
            found = self.search_for_object_in_view(objectType)
            rotate += 1
        if not found:
            answer = "I could not find it. Tell me to move?"
        else:
            answer = f"I found {len(found)} instances of type {objectType} in my view."
            for f, objectType, coord, _ in found:
                answer += f"\n{f['name']} at {coord}"
        return answer, found

    def what_do_you_see(self):
        answer = f"I see {len(self._event.metadata['objects'])} things there.\n"
        for obj in self._event.metadata['objects']:
            answer += obj['objectType'] + "\n"
        return answer

    def do_action(self, action, target):
        answer = ""
        found_objects = []
        if action == "find":
            answer, found_objects = self.search_for_object(target)
        #elif action == "describe":
            #answer = self.what_do_you_see()
        elif action == "head":
            if target == "up":
                self._event = self._controller.step(Action.LookUp.name)
            elif target == "down":
                self._event = self._controller.step(Action.LookDown.name)
        elif action in ["move", "turn"]:
            if target == "forward":
                self._event = self._controller.step(Action.MoveAhead.name)
            elif target == "back":
                self._event = self._controller.step(Action.MoveBack.name)
            elif target == "left":
                self._event = self._controller.step(Action.RotateLeft.name)
            elif target == "right":
                self._event = self._controller.step(Action.RotateRight.name)
        #elif action == "explore":
            # Step 1: Perform 360° scan and generate panorama
            #panorama_path = self.perform_360_view()

            # Step 2: Describe the stitched panorama
            #answer = self.describe_image(panorama_path)
        return answer, found_objects

    def capture_scene_frame(self):
        """
        Capture the current AI2-THOR scene as an image and save it locally.
        """
        image = Image.fromarray(self._controller.last_event.frame)
        image_path = "captured_image.jpg"
        image.save(image_path)
        return image_path

    def encode_image(self, image_path):
        """
        Convert an image file into a Base64-encoded string.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def describe_image_with_gpt(self, image_path):
        """
        Send the captured image to GPT for a detailed description.
        """
        base64_image = self.encode_image(image_path)

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a visual assistant. Describe the contents of images provided. Remember to also mention the number of each type of objects there."
                },
                {
                    "role": "user",
                    "content": [
                    {
                      "type": "text",
                      "text": "What’s in this image?"
                    },
                    {
                      "type": "image_url",
                      "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                      }
                    }
                  ]
                }
            ],
            "max_tokens": 200
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai.api_key}", "Content-Type": "application/json"},
            json=payload
        )

        if response.status_code == 200:
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Failed to get GPT response. Status: {response.status_code}, Response: {response.text}")
            
    def describe_image_object_with_gpt(self, target, image_path):
        """
        Send the captured image to GPT for a detailed description.
        """
        base64_image = self.encode_image(image_path)

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a visual assistant. Describe the contents of images provided."
                },
                {
                    "role": "user",
                    "content": [
                    {
                      "type": "text",
                      "text": f"Describe the '{target}' in the image."
                    },
                    {
                      "type": "image_url",
                      "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                      }
                    }
                  ]
                }
            ],
            "max_tokens": 100
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai.api_key}", "Content-Type": "application/json"},
            json=payload
        )

        if response.status_code == 200:
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Failed to get GPT response. Status: {response.status_code}, Response: {response.text}")

    def nlu_parse(self, prompt):
        """
        Parse the natural language prompt into structured action and target.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """You are a virtual assistant for controlling a robot. Convert the user's natural language instructions into a JSON object with the format: {\"action\": \"<action>\", \"target\": \"<target>\"}. If the user asks a general question or does not provide a target, return {\"action\": \"<action>\", \"target\": \"\"}."The robot can perform actions in {ACTIONS} like "find", "describe", "move", "turn", "head". Action "move" and "turn" should be linked with target "forward", "back", "left" or "right". Action 'head' should be connected with target 'up' or 'down'."""},
                    {"role": "user", "content": prompt}
                ]
            )
            structured_command = response['choices'][0]['message']['content']

            # Convert the JSON-like response to a Python dictionary
            return json.loads(structured_command)  # Use json.loads instead of eval()
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse GPT response into a dictionary: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error while parsing instruction: {e}")
            
    def semantic_similarity(ai_desc):
        human_desc = self._human_description
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        embeddings = model.encode([human_desc, ai_desc])
        similarity_score = util.cos_sim(embeddings[0], embeddings[1])
        return similarity_score.item()

    def compare_descriptions(self, ai_description, human_description):
        """
        Use ChatGPT to compare descriptions and assess confidence.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "An agent and a human will separately give you a description of whay they see. You are a helpful assistant to check if the image taken by the robot and the image seen by human point to the same region of a room. Or at least if they overlap a lot."
                            "Provide a confidence level (0-100%) based on the possibility that the two scenes described by them point to (almost) the same region of a room. "
                            "Respond with a single integer of percentage, the confidence percentage. For instance, if you feel 80% sure, then return 80."
                        ),
                    },
                    {"role": "user", "content": f"Human description: {human_description}\nRobot description: {ai_description}"},
                ],
                max_tokens=10,
                temperature=0,
            )
            #print(human_description)
            # Extract confidence percentage from the response
            confidence = response['choices'][0]['message']['content'].strip()
            #print(confidence)
            return int(confidence)
        except Exception as e:
            print(f"Error during confidence assessment: {e}")
            return 0
            
    def process_instruction(self, prompt):
        self._answers = []
        self._actions = []
        self._perceptions = []
        answer = ""

        try:
            # Use GPT for parsing
            parsed_command = self.nlu_parse(prompt)
            action = parsed_command.get("action", "").lower()
            target = parsed_command.get("target", "").lower()

            if action in ACTIONS:
                self._event = self._controller.step(Action.MoveAhead.name)
                # Inform the user about the action being executed
                if action != "describe" and target == '':
                    confirmation_message = f"I understand. I will now '{action}'... Master, the task is completed."
                    self._answers.append(confirmation_message)
                elif action == "describe" and target == '':
                    confirmation_message = "I understand. You want me to describe what I see."
                    self._answers.append(confirmation_message)
                    image_path = self.capture_scene_frame()
                    description = self.describe_image_with_gpt(image_path)

                    human_desc = self._human_description
                    confidence_level = self.compare_descriptions(human_desc, description)
                    self._answers.append(description + f" I think at least there is a possibility of '{confidence_level}' percents that this region overlaps with the scene you're looking for.")
                elif action == "describe" and target != '':
                    confirmation_message = f"I understand. You want me to describe that object '{target}'."
                    self._answers.append(confirmation_message)
                    image_path = self.capture_scene_frame()
                    description = self.describe_image_object_with_gpt(target, image_path)
                    self._answers.append(description)
                #elif action == "explore":
                    # Step 1: Perform 360° scan and generate panorama
                    #confirmation_message = f"I understand. You want me to take a round look of the room and describe what is in this room."
                    #self._answers.append(confirmation_message)
                    #print("Mission complete, master. I have took a round look of the room.")
                    # Step 2: Describe the stitched panorama
                    #answer = self.do_action(action, target)
                    #self._answers.append(answer)
                else:
                    confirmation_message = f"I understand. I will now '{action}' with the target '{target}'... Master, the task is completed."
                    self._answers.append(confirmation_message)
                #print(f"AI> {confirmation_message}")

                # Execute the action
                answer, found_objects = self.do_action(action, target)
                if answer:
                    self._answers.append(answer)
                if found_objects:
                    self._perceptions.extend(found_objects)

            else:
                # If the action is not understood, notify the user
                error_message = f"Sorry, I do not understand the action '{action}'."
                self._answers.append(error_message)
                print(f"AI> {error_message}")
        except Exception as e:
            # Handle parsing or execution errors
            error_message = f"Error processing the instruction: {str(e)}"
            self._answers.append(error_message)
            print(f"AI> {error_message}")

        return answer

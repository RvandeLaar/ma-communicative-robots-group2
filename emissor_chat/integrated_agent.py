import numpy as np
import cv2
import random
import json
import base64
import requests
import openai
import pprint as pp
import prior
import matplotlib.pyplot as plt
import math

from PIL import Image
from ai2thor.controller import Controller
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple, Dict, Any

ACTIONS = ["find", "describe", "move", "turn", "head", "explore_room", "switch_room", "return_to_previous_room"]

class Agent:
    def __init__(self):
        # Initialize AI2-THOR controller
        self.controller = Controller()
        self.controller.step(action="Initialize")
        
        # Set initial agent position
        self.agent_position = self.controller.last_event.metadata["agent"]["position"]
        
        # Initialize room data (from AgentNavigator)
        self.room_data = self.initialize_room_data()

        # Attributes from Ai2ThorClient
        self._answers = []
        self._actions = []
        self._perceptions = []
        self._human_description = ""

        # Initialize description to start the search
        self.initialize_human_description()

    def initialize_room_data(self) -> dict:
        return {
            'rooms': {
                'room_1': {
                    'doors': {},
                    'visited_doors': set(),
                    'visible_objects': {},
                    'previous_room': None,
                    'visited_positions': {(self.agent_position['x'], self.agent_position['y'], self.agent_position['z'])},
                    'visited_objects': set()
                }
            },
            'current_room': 'room_1'
        }

    def get_reachable_positions(self):
        event = self.controller.step(action='GetReachablePositions')
        return event.metadata['actionReturn']

    def calculate_distance(self, position_1, position_2):
        pos1 = np.array([position_1['x'], position_1['y'], position_1['z']])
        pos2 = np.array([position_2['x'], position_2['y'], position_2['z']])
        return np.linalg.norm(pos1 - pos2)

    def find_best_position_near_door(self, agent_pos, door_coords):
        reachable_positions = self.get_reachable_positions()
        reachable_positions.sort(key=lambda pos: self.calculate_distance(pos, door_coords))
        closest_four = reachable_positions[:4] if len(reachable_positions) >= 4 else reachable_positions
        best_position = max(closest_four, key=lambda pos: self.calculate_distance(pos, agent_pos))
        return best_position

    def perform_360_view(self):
        """
        Perform a 360° view from the current position and update room data.
        Keeping your version of the method, including commented out code.
        """
        current_room_key = self.room_data['current_room']
        current_room = self.room_data['rooms'][current_room_key]
        images = []

        def process_view(event):
            images.append(event.frame)
            for obj in event.metadata['objects']:
                if obj['visible']:
                    obj_id = obj['objectId']
                    pos = obj['position']
                    current_room['visible_objects'][obj_id] = {
                        'name': obj['name'],
                        'object_type': obj['objectType'],
                        'position': pos
                    }
                    # Store doors
                    if obj['objectType'] == 'Doorway' and obj_id not in current_room['doors']:
                        current_room['doors'][obj_id] = {
                            'coordinates': pos,
                            'explored': False
                        }

        for i in range(4):
            event = self.controller.last_event
            process_view(event)
            if i < 3:
                event = self.controller.step(action='RotateRight', degrees=90)

        # If you decide to stitch:
        panorama = self.stitch_images(images)
        panorama_path = "room_panorama.jpg"
        cv2.imwrite(panorama_path, panorama)
        return panorama_path

    # Optionally, if you want to keep stitch_images:
    def stitch_images(self, images):
        cv_images = [cv2.cvtColor(np.array(Image.fromarray(img)), cv2.COLOR_RGB2BGR) for img in images]
        stitcher = cv2.Stitcher_create()
        status, stitched_image = stitcher.stitch(cv_images)
        if status == cv2.Stitcher_OK:
            return stitched_image
        else:
            raise Exception(f"Image stitching failed with status code {status}")

    def explore_room(self):
        current_room_key = self.room_data['current_room']
        current_room = self.room_data['rooms'][current_room_key]
        agent_pos = self.agent_position
        reachable_positions = self.get_reachable_positions()

        visited_positions_list = [
            {'x': x, 'y': y, 'z': z}
            for (x, y, z) in current_room['visited_positions']
        ]

        positions_within_range = [
            pos for pos in reachable_positions
            if any(self.calculate_distance(vp, pos) <= 3.0 for vp in visited_positions_list)
        ]

        unvisited_positions = [
            pos for pos in positions_within_range
            if (pos['x'], pos['y'], pos['z']) not in current_room['visited_positions']
        ]

        if not unvisited_positions:
            print("No new positions to explore within range.")
            return

        target_position = max(
            unvisited_positions,
            key=lambda pos: self.calculate_distance(agent_pos, pos)
        )

        try:
            event = self.controller.step(action='Teleport', position=target_position)
            if not event.metadata['lastActionSuccess']:
                print(f"Teleport failed to position: {target_position}")
                return
            current_room['visited_positions'].add((target_position['x'], target_position['y'], target_position['z']))
            self.agent_position = event.metadata['agent']['position']
            print(f"Agent moved to new position: {target_position}")
        except Exception as e:
            print(f"Error while moving to new position: {e}")

    def teleport_to_door(self, door_id: str):
        current_room_key = self.room_data['current_room']
        current_room = self.room_data['rooms'][current_room_key]
        agent_pos = self.controller.last_event.metadata['agent']['position']

        door_info = current_room['doors'][door_id]
        door_coords = door_info['coordinates']

        target_position = self.find_best_position_near_door(agent_pos, door_coords)
        if not target_position:
            print("No valid position found near the door.")
            return False
        try:
            event = self.controller.step(action='Teleport', position=target_position)
            if not event.metadata['lastActionSuccess']:
                print(f"Failed to teleport to position {target_position}.")
                return False
            self.agent_position = event.metadata['agent']['position']
        except Exception as e:
            print(f"Exception occurred during teleport: {e}")
            return False

        current_room['doors'][door_id]['explored'] = True

        room_number = len(self.room_data['rooms']) + 1
        new_room_key = f"room_{room_number}"
        if new_room_key not in self.room_data['rooms']:
            self.room_data['rooms'][new_room_key] = {
                'doors': {},
                'visited_doors': set(),
                'visible_objects': {},
                'previous_room': current_room_key,
                'visited_positions': {(self.agent_position['x'], self.agent_position['y'], self.agent_position['z'])},
                'visited_objects': set()
            }

            self.room_data['rooms'][new_room_key]['doors'][door_id] = {
                'coordinates': door_coords,
                'explored': True
            }
            self.room_data['current_room'] = new_room_key
        print(f"Entered new room: {new_room_key}")
        self.close_door()
        print("Closed the door.")

    def close_door(self):
        event = self.controller.last_event
        objects = event.metadata['objects']

        open_doors = [
            obj for obj in objects
            if obj['objectType'] == 'Doorway' and obj.get('openable', False) and obj.get('isOpen', False)
        ]

        if not open_doors:
            print("No open doors found.")
            return

        closest_door = min(
            open_doors,
            key=lambda d: self.calculate_distance(self.agent_position, d['position'])
        )

        door_id = closest_door['objectId']
        close_event = self.controller.step(action='CloseObject', objectId=door_id)
        if not close_event.metadata['lastActionSuccess']:
            print("Failed to close the door.")

    def switch_new_room(self):
        current_room_key = self.room_data['current_room']
        current_room = self.room_data['rooms'][current_room_key]

        unexplored_doors = [
            (door_id, door_info) for door_id, door_info in current_room['doors'].items()
            if not door_info['explored']
        ]
        if not unexplored_doors:
            print("No unexplored doors available in the current room.")
            return False

        door_id, _ = unexplored_doors[0]
        return self.teleport_to_door(door_id)

    def return_previous_room(self):
        current_room_key = self.room_data['current_room']
        current_room = self.room_data['rooms'][current_room_key]
        prev_room_key = current_room.get('previous_room', None)

        if not prev_room_key:
            print("No previous room recorded. Cannot return.")
            return False

        previous_room = self.room_data['rooms'][prev_room_key]
        visited_positions = previous_room.get('visited_positions', None)

        if not visited_positions:
            print(f"No visited positions recorded for previous room: {prev_room_key}")
            return False

        visited_positions_list = list(visited_positions)
        chosen_pos_tuple = random.choice(visited_positions_list)
        chosen_pos = {'x': chosen_pos_tuple[0], 'y': chosen_pos_tuple[1], 'z': chosen_pos_tuple[2]}

        event = self.controller.step(action='Teleport', position=chosen_pos)
        if event.metadata['lastActionSuccess']:
            self.agent_position = event.metadata['agent']['position']
            self.room_data['current_room'] = prev_room_key
            print(f"Returned to previous room: {prev_room_key} at {chosen_pos}")
            return True
        else:
            print(f"Failed to teleport to position {chosen_pos} in previous room {prev_room_key}.")
            return False

    # Methods from Ai2ThorClient (adjusted to use self.controller):
    def initialize_human_description(self):
        print("Robot> Please describe in one sentence what you see in the image shown.")
        human_input = input("Evaluator> ")
        self._human_description = human_input.strip()
        print(f"Robot> Thank you! I have stored your description: \"{self._human_description}\"")

    def search_for_object_in_view(self, objectType):
        event = self.controller.last_event
        found = []
        for obj in event.metadata['objects']:
            if obj['objectType'].lower() == objectType.lower():
                coord = obj['position']
                found.append((obj, objectType, coord, Image.fromarray(event.frame)))
        return found

    def search_for_object(self, objectType):
        answer = ""
        found = self.search_for_object_in_view(objectType)
        rotate = 0
        while not found and rotate < 4:
            self.controller.step(action="RotateRight", degrees=90)
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
        event = self.controller.last_event
        answer = f"I see {len(event.metadata['objects'])} things there.\n"
        for obj in event.metadata['objects']:
            answer += obj['objectType'] + "\n"
        return answer

    def do_action(self, action, target):
        # This will depend on whether you have Action enums or not.
        # For simplicity, let's handle moves as strings:
        answer = ""
        found_objects = []
        if action == "find":
            answer, found_objects = self.search_for_object(target)
        elif action == "head":
            if target == "up":
                self.controller.step(action="LookUp")
            elif target == "down":
                self.controller.step(action="LookDown")
        elif action in ["move", "turn"]:
            if target == "forward":
                self.controller.step(action="MoveAhead")
            elif target == "back":
                self.controller.step(action="MoveBack")
            elif target == "left":
                self.controller.step(action="RotateLeft", degrees=90)
            elif target == "right":
                self.controller.step(action="RotateRight", degrees=90)
        return answer, found_objects

    def capture_scene_frame(self):
        image = Image.fromarray(self.controller.last_event.frame)
        image_path = "captured_image.jpg"
        image.save(image_path)
        return image_path

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def describe_image_with_gpt(self, image_path):
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
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """You are a virtual assistant for controlling a robot. Convert the user's natural language instructions into a JSON object with the format: {"action": "<action>", "target": "<target>"}. If the user asks a general question or does not provide a target, return {"action": "<action>", "target": ""}."The robot can perform actions in {ACTIONS} like "find", "describe", "move", "turn", "head", "explore_room", "switch_room", "return_to_previous_room". Action "move" and "turn" should be linked with target "forward", "back", "left" or "right". Action 'head' should be connected with target 'up' or 'down'."""},
                    {"role": "user", "content": prompt}
                ]
            )
            structured_command = response['choices'][0]['message']['content']
            return json.loads(structured_command)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse GPT response into a dictionary: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error while parsing instruction: {e}")

    def compare_descriptions(self, ai_description, human_description):
        # Using GPT to compare (as in original Ai2ThorClient)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "An agent and a human will separately give you a description of what they see. Provide a confidence level (0-100%) that the two scenes overlap."
                        ),
                    },
                    {"role": "user", "content": f"Human description: {human_description}\nRobot description: {ai_description}"},
                ],
                max_tokens=10,
                temperature=0,
            )
            confidence = response['choices'][0]['message']['content'].strip()
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
            parsed_command = self.nlu_parse(prompt)
            action = parsed_command.get("action", "").lower()
            target = parsed_command.get("target", "").lower()

            if action in ACTIONS:
                if action == "describe" and target == '':
                    # Describe current scene
                    confirmation_message = "I understand. You want me to describe what I see."
                    self._answers.append(confirmation_message)
                    image_path = self.capture_scene_frame()
                    description = self.describe_image_with_gpt(image_path)
                    human_desc = self._human_description
                    confidence_level = self.compare_descriptions(human_desc, description)
                    self._answers.append(description + f" I think there is a possibility of '{confidence_level}'% overlap.")
                elif action == "describe" and target != '':
                    # Describe a specific object
                    confirmation_message = f"I understand. You want me to describe that object '{target}'."
                    self._answers.append(confirmation_message)
                    image_path = self.capture_scene_frame()
                    description = self.describe_image_object_with_gpt(target, image_path)
                    self._answers.append(description)
                else:
                    confirmation_message = f"I understand. I will now '{action}' with the target '{target}'..."
                    self._answers.append(confirmation_message)

                answer, found_objects = self.do_action(action, target)
                if answer:
                    self._answers.append(answer)
                if found_objects:
                    self._perceptions.extend(found_objects)
            else:
                error_message = f"Sorry, I do not understand the action '{action}'."
                self._answers.append(error_message)
                print(error_message)
        except Exception as e:
            error_message = f"Error processing the instruction: {str(e)}"
            self._answers.append(error_message)
            print(error_message)

        return answer
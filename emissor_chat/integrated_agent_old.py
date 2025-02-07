import numpy as np
import cv2
import random
import json
import base64
import requests
import openai

from PIL import Image
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple, Dict, Any

ACTIONS = ["find", "describe", "move", "turn", "head", "explore_room", "switch_room", "return_to_previous_room", "perform_360_view"]

class Agent:
    def __init__(self, controller: object):
        # Set initial agent position
        self.controller = controller
        self.agent_position = self.controller.last_event.metadata["agent"]["position"]
        
        # Initialize room data (from AgentNavigator)
        self.room_data = self.initialize_room_data()

        # Count the number of actions (environment interactions)
        self.actions = 0

        # Attributes from Ai2ThorClient
        self._answers = []
        self._actions = []
        self._perceptions = []
        self._human_description = ""

    def increment_actions(self, event):
        # If the event was successful and changed state, increment action count
        if event.metadata['lastActionSuccess']:
            self.actions += 1

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

        # Initial view without turning
        event = self.controller.last_event
        process_view(event)

        for i in range(3):
            rotate_event = self.controller.step(action='RotateRight', degrees=90)
            self.increment_actions(rotate_event)
            process_view(rotate_event)

        # Stitch images
        panorama = self.stitch_images(images)
        panorama_path = "room_panorama.jpg"
        cv2.imwrite(panorama_path, panorama)
        return panorama_path
    
    def stitch_images(self, images):
        # If less than 2 images, there's nothing to stitch
        if len(images) < 2:
            # Just return the first image as is
            return cv2.cvtColor(np.array(Image.fromarray(images[0])), cv2.COLOR_RGB2BGR)
        
        cv_images = [cv2.cvtColor(np.array(Image.fromarray(img)), cv2.COLOR_RGB2BGR) for img in images]
        
        # Check if all images are the same size and type
        shapes = [img.shape for img in cv_images]
        if len(set(shapes)) > 1:
            print("Images have different shapes, cannot stitch. Returning first image only.")
            return cv_images[0]

        stitcher = cv2.Stitcher_create()
        status, stitched_image = stitcher.stitch(cv_images)
        if status == cv2.Stitcher_OK:
            return stitched_image
        else:
            print(f"Image stitching failed with status code {status}. Returning first image only.")
            return cv_images[0]  # fallback to just using the first image

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
                self._answers.append(f"Teleport failed to position: {target_position}")
                return
            current_room['visited_positions'].add((target_position['x'], target_position['y'], target_position['z']))
            self.agent_position = event.metadata['agent']['position']
            self._answers.append(f"I moved to a new position in the room. If you would like to perform another 360-view here, please let me know.")
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
            self._answers.append("No valid position found near the door.")
            return False
        try:
            event = self.controller.step(action='Teleport', position=target_position)
            self.increment_actions(event)
            if not event.metadata['lastActionSuccess']:
                self._answers.append(f"Failed to teleport to position {target_position}.")
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
        self._answers.append(f"Entered new room: {new_room_key}")
        self.close_door()

    def close_door(self):
        event = self.controller.last_event
        objects = event.metadata['objects']

        open_doors = [
            obj for obj in objects
            if obj['objectType'] == 'Doorway' or obj['objectType'] == 'Door' and obj.get('openable', False) and obj.get('isOpen', False)
        ]

        if not open_doors:
            self._answers.append("No open doors found.")
            return

        closest_door = min(
            open_doors,
            key=lambda d: self.calculate_distance(self.agent_position, d['position'])
        )

        door_id = closest_door['objectId']
        close_event = self.controller.step(action='CloseObject', objectId=door_id)
        self.increment_actions(close_event)
        if not close_event.metadata['lastActionSuccess']:
            self._answers.append("Failed to close the door.")
        else:
            self._answers.append("Closed the door.")

    def switch_new_room(self):
        current_room_key = self.room_data['current_room']
        current_room = self.room_data['rooms'][current_room_key]

        unexplored_doors = [
            (door_id, door_info) for door_id, door_info in current_room['doors'].items()
            if not door_info['explored']
        ]
        if not unexplored_doors:
            self._answers.append("No unexplored doors available in the current room.")
            self.return_previous_room()
            return False

        door_id, _ = unexplored_doors[0]
        return self.teleport_to_door(door_id)

    def return_previous_room(self):
        current_room_key = self.room_data['current_room']
        current_room = self.room_data['rooms'][current_room_key]
        prev_room_key = current_room.get('previous_room', None)

        if not prev_room_key:
            self._answers.append("No previous room recorded. Cannot return.")
            return False

        previous_room = self.room_data['rooms'][prev_room_key]
        visited_positions = previous_room.get('visited_positions', None)

        if not visited_positions:
            self._answers.append(f"No visited positions recorded for previous room: {prev_room_key}")
            return False

        visited_positions_list = list(visited_positions)
        chosen_pos_tuple = random.choice(visited_positions_list)
        chosen_pos = {'x': chosen_pos_tuple[0], 'y': chosen_pos_tuple[1], 'z': chosen_pos_tuple[2]}

        event = self.controller.step(action='Teleport', position=chosen_pos)
        self.increment_actions(event)
        if event.metadata['lastActionSuccess']:
            self.agent_position = event.metadata['agent']['position']
            self.room_data['current_room'] = prev_room_key
            self._answers.append(f"Returned to previous room: {prev_room_key} at {chosen_pos}")
            return True
        else:
            self._answers.append(f"Failed to teleport to position {chosen_pos} in previous room {prev_room_key}.")
            return False

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
            rotate_event = self.controller.step(action="RotateRight", degrees=90)
            self.increment_actions(rotate_event)
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
        answer = ""
        found_objects = []

        if action == "find":
            answer, found_objects = self.search_for_object(target)
            

        elif action == "head":
            if target == "up":
                event = self.controller.step(action="LookUp")
                self.increment_actions(event)
            elif target == "down":
                event = self.controller.step(action="LookDown")
                self.increment_actions(event)

        elif action == "move":
            if target == "forward":
                event = self.controller.step(action="MoveAhead")
                self.increment_actions(event)
            elif target == "back":
                event = self.controller.step(action="MoveBack")
                self.increment_actions(event)

        elif action == "turn":
            if target == "left":
                event = self.controller.step(action="RotateLeft", degrees=90)
                self.increment_actions(event)
            elif target == "right":
                event = self.controller.step(action="RotateRight", degrees=90)
                self.increment_actions(event)

        elif action == "explore_room":
            # Perform the exploration and any relevant messages are stored in self._answers
            self.explore_room()

        elif action == "switch_room":
            # Attempt to switch room
            self.switch_new_room()
            answer = "Would you like me to perform a 360 view of the new room and describe it?"

        elif action == "return_to_previous_room":
            self.return_previous_room()

        elif action == "perform_360_view":
            panorama_path = self.perform_360_view()
            description = self.describe_image_with_gpt(panorama_path)
            confidence_level = self.compare_descriptions(description, self._human_description)
            answer = f"This is what I see: {description}.\n I think there is a possibility of '{confidence_level}'% that the object shows in my current view.\n What would you like to do next? If it's low probability, maybe I should switch room. Otherwise maybe I should find the object in this room."

        # Always return a tuple, even if empty
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
            "max_tokens": 300
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
                    "content": "You are a visual assistant. Describe the object in the image provided."
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

    def nlu_parse(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """You are a virtual assistant for controlling a robot. Convert the user's natural language instructions into a JSON object with the format: {"action": "<action>", "target": "<target>"}. If the user asks a general question or does not provide a target, return {"action": "<action>", "target": ""}."The robot can perform actions in {ACTIONS} like "find", "describe", "move", "turn", "head", "explore_room", "switch_room", "return_to_previous_room", "perform_360_view". Action "move" and "turn" should be linked with target "forward", "back", "left" or "right". Action 'head' should be connected with target 'up' or 'down'."""},
                    {"role": "user", "content": prompt}
                ]
            )
            structured_command = response['choices'][0]['message']['content']
            return json.loads(structured_command)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse GPT response into a dictionary: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error while parsing instruction: {e}")

    def compare_descriptions(self, human_description, image_path):
        try:
            # Encode the image to base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
            # Define the payload for the request
            payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict parser. Given a human description of a scene, and their description of an object in the scene they need to find, "
                        "compare the human textual saying with the visual data from the image. Output a single integer (0 to 100) representing the confidence "
                        "that the described object is possibly in the image. Do not include any additional words, explanations, or symbols. Only output the integer."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Human description: {human_description}"
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
            "max_tokens": 10,
            "temperature": 0,
            }
        
            # Make the API request
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai.api_key}", "Content-Type": "application/json"},
                json=payload
            )

            # Process the response
            if response.status_code == 200:
                response_json = response.json()
                confidence_str = response_json["choices"][0]["message"]["content"].strip()
                confidence = int(confidence_str)  # Convert to integer
                return confidence
            else:
                raise Exception(f"Failed to get GPT response. Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            print(f"Error during confidence assessment: {e}")
            return 0  # Return 0 if something goes wrong


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
                    confidence_level = self.compare_descriptions(human_desc, image_path)
                    self._answers.append(description + f" I think there is a possibility of '{confidence_level}'% that the object shows in my current view.")
                elif action == "describe" and target != '':
                    # Describe a specific object
                    confirmation_message = f"I understand. You want me to describe that object '{target}'."
                    self._answers.append(confirmation_message)
                    image_path = self.capture_scene_frame()
                    description = self.describe_image_object_with_gpt(target, image_path)
                    self._answers.append(description)
                elif action == "perform_360_view":
                    confirmation_message = f"I understand. I will now '{action}'"
                    self._answers.append(confirmation_message)
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
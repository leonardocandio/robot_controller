#!/usr/bin/env python3

import ollama
import cv2
import numpy as np
import logging
import re
import time
import os
from pathlib import Path
import json
from datetime import datetime

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelEvaluator")

class ModelEvaluator:
    def __init__(self, model_name, images_dir, model_prompt):
        self.model_name = model_name
        self.images_dir = Path(images_dir)
        self.ollama_client = ollama.Client()
        self.model_prompt = model_prompt
        self.metrics = {
            'response_times': [],
            'parsing_success_rate': 0,
            'responses': []
        }
        self.modelfile= f"""from {self.model_name}
            parameter temperature 0.5
            system "{self.model_prompt}"
            """
        
        # Mock LiDAR data for testing
        self.mock_lidar_scenarios = [
            {
                "front": 1.5, "front_left": 2.0, "front_right": 1.8,
                "left": 2.5, "right": 2.2, "rear": 3.0
            },
            {
                "front": 0.5, "front_left": 0.8, "front_right": 0.3,
                "left": 1.5, "right": 1.2, "rear": 2.0
            },
            {
                "front": 3.0, "front_left": 3.0, "front_right": 3.0,
                "left": 3.0, "right": 3.0, "rear": 3.0
            }
        ]

    def create_model(self):
        
        try:
            self.ollama_client.create(model=self.model_name, modelfile=self.modelfile)
            logger.info(f"Model '{self.model_name}' created successfully")
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            raise

    def translate_response(self, response):
        """Extract control values from model response"""
        linear_pattern = r"<linear>(.*?)</linear>"
        angular_pattern = r"<angular>(.*?)</angular>"
        reasoning_pattern = r"<reasoning>(.*?)</reasoning>"

        try:
            linear_match = re.search(linear_pattern, response, re.DOTALL)
            angular_match = re.search(angular_pattern, response, re.DOTALL)
            reasoning_match = re.search(reasoning_pattern, response, re.DOTALL)

            return {
                "linear": float(linear_match.group(1).strip()) if linear_match else None,
                "angular": float(angular_match.group(1).strip()) if angular_match else None,
                "reasoning": reasoning_match.group(1).strip() if reasoning_match else None,
            }
        except Exception as e:
            logger.error(f"Failed to parse response: {str(e)}")
            return None

    def evaluate_image(self, image_path, lidar_data):
        """Evaluate a single image with the model"""
        try:
            # Read and encode image
            image = cv2.imread(str(image_path))
            _, buffer = cv2.imencode('.jpg', image)
            image_bytes = buffer.tobytes()

            # Prepare prompt with LiDAR context
            lidar_context = (
                f"LiDAR readings: "
                f"Front: {lidar_data['front']:.2f}m, "
                f"Front-Left: {lidar_data['front_left']:.2f}m, "
                f"Front-Right: {lidar_data['front_right']:.2f}m, "
                f"Left: {lidar_data['left']:.2f}m, "
                f"Right: {lidar_data['right']:.2f}m, "
                f"Rear: {lidar_data['rear']:.2f}m. "
            )
            prompt = f"Analyze this image. {lidar_context} Generate control commands."

            # Time the model response
            start_time = time.time()
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_bytes],
                keep_alive="1h"
            )
            response_time = time.time() - start_time

            # Parse response
            parsed_response = self.translate_response(response.response)
            
            return {
                'image_name': image_path.name,
                'response_time': response_time,
                'raw_response': response.response,
                'parsed_response': parsed_response,
                'lidar_data': lidar_data
            }

        except Exception as e:
            logger.error(f"Error evaluating image {image_path}: {str(e)}")
            return None

    def run_evaluation(self, num_iterations=1):
        """Run evaluation on all images in the directory"""
        logger.info(f"Starting evaluation with {num_iterations} iterations per image")
        
        results = []
        image_files = list(self.images_dir.glob('*.jpeg')) + list(self.images_dir.glob('*.png'))
        
        for image_path in image_files:
            logger.info(f"Evaluating image: {image_path.name}")
            
            for i in range(num_iterations):
                # Rotate through mock LiDAR scenarios
                lidar_data = self.mock_lidar_scenarios[i % len(self.mock_lidar_scenarios)]
                
                result = self.evaluate_image(image_path, lidar_data)
                if result:
                    results.append(result)
                    
                # Log immediate results
                if result and result['parsed_response']:
                    logger.info(
                        f"Iteration {i+1} - Response Time: {result['response_time']:.3f}s, "
                        f"Linear: {result['parsed_response']['linear']}, "
                        f"Angular: {result['parsed_response']['angular']}"
                    )
                
                time.sleep(1)  # Prevent overwhelming the API
        
        self.save_results(results)
        self.analyze_results(results)

    def analyze_results(self, results):
        """Analyze and log evaluation results"""
        if not results:
            logger.error("No results to analyze")
            return

        # Calculate metrics
        response_times = [r['response_time'] for r in results]
        parsing_success = len([r for r in results if r['parsed_response'] is not None])
        
        metrics = {
            'total_evaluations': len(results),
            'average_response_time': np.mean(response_times),
            'std_response_time': np.std(response_times),
            'parsing_success_rate': parsing_success / len(results),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times)
        }

        logger.info("\n=== Evaluation Results ===")
        logger.info(f"Total evaluations: {metrics['total_evaluations']}")
        logger.info(f"Average response time: {metrics['average_response_time']:.3f}s")
        logger.info(f"Response time std dev: {metrics['std_response_time']:.3f}s")
        logger.info(f"Parsing success rate: {metrics['parsing_success_rate']*100:.1f}%")
        logger.info(f"Min response time: {metrics['min_response_time']:.3f}s")
        logger.info(f"Max response time: {metrics['max_response_time']:.3f}s")

    def save_results(self, results):
        """Save detailed results to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Detailed results saved to {filename}")

def main():
    # Configuration
    model_name = "minicpm-v"
    images_dir = "photos"  # Directory containing test images
    model_prompt = (
        """
    You are a TurtleBot3 ROS2-powered robot navigating a racetrack. The track is defined by carboard boxes, multiple obstacles and a glass wall. Your objective is to navigate the racetrack and win the race. There will be other robots in the race, try to avoid them.

    Your task is to generate ROS2 Twist control commands based on the following input:
    - LiDAR data: obstacle distances in meters for front, front left, front right, left, and right directions.
    - Camera image: a real-time view of the environment.

    Format your response strictly as:
    - Provide a short one sentence reasoning explanation behind your commands enclosed in <reasoning> tags.
    - Provide the forward velocity in m/s (linear.x) value enclosed in <linear> tags.
    - Provide the angular velocity in rad/s (angular.z) value enclosed in <angular> tags.

    Important: You should always respond with a command even if the robot is not inside the specified location

    Explanation of <linear> and <angular>
    1: Moving a Robot Forward
    If you want to move a robot forward with a linear velocity of 1 meter per second and no angular velocity, you would use the following command
    <linear>1.0</linear>
    <angular>0.0</angular>


    2: Rotating the Robot
    To rotate a robot around its z-axis (turning in place), you would set the angular velocity and leave the linear velocitiy at zero:
    <linear>0.0</linear>
    <angular>1.0</angular>

    3: Moving and Rotating Simultaneously
    If you want the robot to move forward while rotating, you can set both the linear and angular velocities. For example, the robot might move forward at 0.5 m/s while rotating at 0.2 rad/s:
    <linear>0.5</linear>
    <angular>0.2</angular>

    Example response:
    <reasoning>To follow the sharp right turn, the robot should turn right</reasoning>
    <linear>0.4</linear><angular>0.5</angular>



    DO NOT include any text outside of these tags. Do not provide explanations, comments, or additional information.
    DO NOT nest tags
    """
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(model_name, images_dir, model_prompt)
    
    try:
        # Create/update model
        evaluator.create_model()
        
        # Run evaluation
        evaluator.run_evaluation(num_iterations=1)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")

if __name__ == "__main__":
    main()

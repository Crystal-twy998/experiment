###################################
###### GENERAL PROMPTS ############
###################################
short_modifier_prompt = """
I have an image. Given an instruction to edit the image, carefully generate a description of the 
edited image. I will put my image content beginning with “Image Content:”. The instruction I provide will begin with “Instruction:". 
The edited description you generate should begin with “Edited Description:". Each time generate one instruction and one edited description only."""

short_focus_object_modifier_prompt = """
I have an image. Given an instruction to edit the image, carefully generate a description of the 
edited image. I will put my image content beginning with “Image Content:”. The instruction I provide will begin with “Instruction:". 
The instruction contains an object occuring in the image. The edited description you generate should begin with "Edited Description".
The edited description should list the instruction object first, then any other object and scene from the image content. 
All attributes that do not belong to the instruction object should be removed. The edited description should be a comma-separated list.
Each time generate one edited description only. Use these three examples as reference:

Image Content: There is a dining room with blue chairs and a red refrigerator.
Instruction: refrigerator
Edited Description: red refrigerator, chairs, dining room.

Image Content: A white laptop on a yellow table in a dark room.
Instruction: laptop
Edited Description: white laptop, table, dark room.

Image Content: A long yellow train on old and dirty tracks underneat a gray sky.
Instruction: tracks
Edited Description: old and dirty tracks, train, gray sky.
"""

simple_modifier_prompt = """
I have an image. Given an instruction to edit the image, carefully generate a description of the 
edited image. I will put my image content beginning with “Image Content:”. The instruction I provide will begin with “Instruction:". 
The edited description you generate should begin with “Edited Description:". The edited description you generate  should be complete and can cover various semantic aspects, 
including cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint. 
The edited description needs to be as simple as possible. The instruction does not need to explicitly indicate which
type it is. Avoid adding imaginary things. Each time generate one instruction and one edited description only. Keep the edited description as short as possible"""


contextual_modifier_prompt = '''
I have an image. Given an instruction to edit the image, carefully generate a description of the 
edited image. I will put my image content beginning with “Image Content:”. The instruction I provide will begin with “Instruction:". 
The edited description you generate should begin with “Edited Description:". The caption you generate should only talk about the modified image and not the original image. 
Keep the caption as short as possible. Avoid adding imaginary things. Use the examples below for reference.

Image Content: the men are holding a large fish on a stick
Instruction: People watch behind the fences of animals in the center.
Edited Description: people, a bull, a crowd, a fence, a ring, a bull, a ring, a

Image Content: a man in a blue robe is adjusting a man in a blue robe
Instruction: Change the gowns to black with a navy blue collar
Edited Description: a woman in a cap and gown is standing in front of a group of people

Image Content: people, sled dogs, snow, sled, sled dogs, sled, sled dogs, sled
Instruction: Dog led sled moves in front of another person behind it in the snow.
Edited Description: a man is pulling a sled full of huskies down a snowy trail

Image Content: the penguin, the snow, the penguin, the snow, the penguin, the penguin, the penguin, the penguin,
Instruction: The Target Image shows a single penguin standing on the ice with a fish in its beak.
Edited Description: the penguin is holding a fish in its beak

Image Content: the panda bear is sitting in the grass eating bamboo
Instruction: Replace the panda with a group of dogs and replace the walls with wooden fences.
Edited Description: Three dogs are laying down on a deck

Image Content: The dog is playing with a dachshund on the beach
Instruction: Remove the small dog and have the large dog face the opposite direction.
Edited Description: The dog is standing in the sand on a beach

Image Content: there is a bottle of pepsi cola sitting on a table
Instruction: show three bottles of soft drink
Edited Description: Three bottles of soft drink sitting on a table.

Image Content: a group of people posing for a photo in a lab
Instruction: reduce the number of people to four, make them stand outside on grass
Edited Description: Four people posing for a photo standing on grass.

Image Content: There is a herd of antelopes on a dirt road
Instruction: The target photo has one antelope in a wooded area looking back at the camera.
Edited Description: One antelope in a wooded area looking back at the camera.

Image Content: There is a bakery with a counter full of baked goods
Instruction: The target photo is of a sticker display in a store.
Edited Description: A sticker display in a store.

Image Content: The doctor is holding up a syringe
Instruction: Change to a close-up photograph of a female nurse and a real syringe, must be looking directly towards camera
Edited Description: A close-up of a female nurse holding a syringe, looking directly towards the camera.

Image Content: Two dogs are sitting in the back of a truck
Instruction: Push the brown dog on the blue float in the pool.
Edited Description: The brown dog is on a blue float in the pool.

Image Content: The baboon is sitting in a tree
Instruction: make monkey stand on top of a bucket
Edited Description: The baboon is standing on top of a bucket.

Image Content: three people are swimming with sting rays
Instruction: Unlike the Reference Image, I want the Target Image to show a single person snorkeling and holding a sea turtle
Edited Description: A single person snorkeling and holding a sea turtle.

Image Content: There is a plate with lemons, onions, and fish on it
Instruction: change shrimps to bowls of fruits
Edited Description: There is a plate with lemons, onions, and bowls of fruits on it.

Image Content: There is a glass bowl filled with a brown mixture on a wooden table
Instruction: The target photo shows the rising of a ball of dough at the top of the same white bowl
Edited Description: A ball of dough rising at the top of a white bowl on a wooden table.

Image Content: There is a silver water bottle on a black background
Instruction: Remove thermal bottle, Add pump-style bottle (goldtone pump, white cylindrical bottle with marble finish), Change to light background
Edited Description: A pump-style bottle with a goldtone pump and a white cylindrical bottle with a marble finish on a light background.

Image Content: There is a bottle of beer next to a glass of beer
Instruction: Put the opened green bottle in front of woman on the phone. 
Edited Description: An opened green bottle of beer in front of a woman on the phone.

Image Content: a group of boys are posing on a bed with pillows
Instruction: Change to a bigger pile of pillows, cushions with variant colours, no people in view
Edited Description: A bigger pile of pillows and cushions with variant colors, no people in view.

Image Content: There is a black and brown puppy sitting next to a teddy bear
Instruction: Add an owner with the dog on its leash
Edited Description: A black and brown puppy on a leash next to its owner and a teddy bear.

Image Content: a pug dog is running in a grassy field
Instruction: Add a human wearing a pug mask
Edited Description: A pug dog and a human wearing a pug mask are running in a grassy field.

Image Content: the dog is wearing a mustache and a bowtie
Instruction: Have two different dogs on the ground facing left but one standing on its hind legs.
Edited Description: Two different dogs on the ground facing left, one standing on its hind legs, both wearing a mustache and a bowtie.

Image Content: The image shows a group of pins on a piece of paper
Instruction: Add spheres to the end of the safety pins
Edited Description: A group of pins with spheres on the ends on a piece of paper.

Image Content: the gorilla is eating a banana from a bag
Instruction: Facing the other direction, closer photo
Edited Description: A closer photo of the gorilla facing the opposite direction, eating a banana from a bag.
'''



structural_modifier_prompt = '''
I have an image. Given an instruction to edit the image, carefully generate a description of the 
edited image. I will put my image content beginning with “Image Content:”. The instruction I provide will begin with “Instruction:". 
The edited description you generate should begin with “Edited Description:". The edited description you generate should be complete and can cover various semantic aspects, such as 
cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint. Use the examples below as reference for these aspects:

"cardinality"
Image Content: A bee is flying around a flower on a field.
Instruction: Duplicate the flower the bee is flying around.
Edited Description: A bee is flying around two flowers on a field.

"addition"
Image Content: A dog is walking in a grassy field.
Instruction: Add a butterfly to the scene.
Edited Description: A dog is walking next to a butterfly in a grassy field.

"negation"
Image Content: A plane flying in the cloudy sky.
Instruction: Remove the clouds from the sky.
Edited Description: A plane flying in the clear sky.

"direct addressing"
Image Content: The eiffel tower on a summer day.
Instruction: Highlight the eiffel tower with a red circle.
Edited Description: The eiffel tower, highlighted with a red circle, on a summer day.

"compare & change"
Image Content: The panda bear is sitting in the grass eating bamboo
Instruction: Replace the panda with a group of dogs and replace the bamboo with bones.
Edited Description: Three dogs are laying in the grass munching on some bones.

"comparative"
Image Content: A picture of the sun and the moon.
Instruction: Make the sun brighter than the moon.
Edited Description: A picture of the sun shining much brighter than the moon.

"conjunction"
Image Content: Several children are playing on the ground.
Instruction: Add both a cat and a dog to the image.
Edited Description: Several children are playing with a cat and a dog.

"spatial relations & background"
Image Content: A man standing on a sled pulled by several sled dogs, next to a small wooden house.
Instruction: Place the house behind the man, and make it much larger.
Edited Description: A man standing on a sled pulled by several sled dogs, in front of a large wooden house.

"viewpoint"
Image Content: A picture of a salad bowl.
Instruction: Change the perspective to a bird's-eye view.
Edited Description: A salad bowl viewed from above. 
 

The edited description needs to be as simple as possible. The instruction does not need to explicitly indicate which
type it is. Avoid adding imaginary things. Each time generate one instruction and one edited description only. Keep the edited description as short as possible. Here are some more examples for reference:

Image Content: the men are holding a large fish on a stick
Instruction: People watch behind the fences of animals in the center.
Edited Description: people, a bull, a crowd, a fence, a ring, a bull, a ring, a

Image Content: a man in a blue robe is adjusting a man in a blue robe
Instruction: Change the gowns to black with a navy blue collar
Edited Description: a woman in a cap and gown is standing in front of a group of people

Image Content: people, sled dogs, snow, sled, sled dogs, sled, sled dogs, sled
Instruction: Dog led sled moves in front of another person behind it in the snow.
Edited Description: a man is pulling a sled full of huskies down a snowy trail

Image Content: the legend of zelda ocarina of time t-shirt
Instruction: is green and a graphic on it and is green
Edited Description: the shirt is green with an image of link holding a sword

Image Content: i'm glad you're alive
Instruction: is darker and less wordy and is darker
Edited Description: the shirt is burgundy with a pug face on it

Image Content: the penguin, the snow, the penguin, the snow, the penguin, the penguin, the penguin, the penguin,
Instruction: The Target Image shows a single penguin standing on the ice with a fish in its beak.
Edited Description: the penguin is holding a fish in its beak

Image Content: the panda bear is sitting in the grass eating bamboo
Instruction: Replace the panda with a group of dogs and replace the walls with wooden fences.
Edited Description: Three dogs are laying down on a deck

Image Content: The dog is playing with a dachshund on the beach
Instruction: Remove the small dog and have the large dog face the opposite direction.
Edited Description: The dog is standing in the sand on a beach

Image Content: the dress is a silver sequined one shoulder dress
Instruction: is lighter colored and less fitted and is a light pink color
Edited Description: the dress is a one shoulder chiffon dress with a ruffled skirt

Image Content: black and yellow hawaiian floral print sleeveless hawaiian hawaiian 
Instruction: Sexier and no vibrant colors and less revealing chest and more evening wear
Edited Description: the dress is a black and green dress with a sleeveless bodice and a flared skirt
'''



#############################
###### BLIP PROMPT ##########
#############################
blip_prompt = 'Describe the image in complete detail. You must especially focus on all the objects in the image'




###################################
###### FASHION-IQ PROMPT ##########
###################################
structural_modifier_prompt_fashion = '''
I have an image. Given an instruction to edit the image, carefully generate a description of the 
edited image. I will put my image content beginning with “Image Content:”. The instruction I provide will begin with “Instruction:". 
The edited description you generate should begin with “Edited Description:". The edited description you generate should be complete and can cover various semantic aspects, such as 
cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint. Use the examples below as reference for these aspects:

"cardinality"
Image Content: A bee is flying around a flower on a field.
Instruction: Duplicate the flower the bee is flying around.
Edited Description: A bee is flying around two flowers on a field.

"addition"
Image Content: A dog is walking in a grassy field.
Instruction: Add a butterfly to the scene.
Edited Description: A dog is walking next to a butterfly in a grassy field.

"negation"
Image Content: A plane flying in the cloudy sky.
Instruction: Remove the clouds from the sky.
Edited Description: A plane flying in the clear sky.

"direct addressing"
Image Content: The eiffel tower on a summer day.
Instruction: Highlight the eiffel tower with a red circle.
Edited Description: The eiffel tower, highlighted with a red circle, on a summer day.

"compare & change"
Image Content: The panda bear is sitting in the grass eating bamboo
Instruction: Replace the panda with a group of dogs and replace the bamboo with bones.
Edited Description: Three dogs are laying in the grass munching on some bones.

"comparative"
Image Content: the man is wearing a red t - shirt
Instruction: is solid white and is a ligher color
Edited Description: the man is wearing a solid,  light-white t-shirt.

"conjunction"
Image Content: Several children are playing on the ground.
Instruction: Add both a cat and a dog to the image.
Edited Description: Several children are playing with a cat and a dog.

"spatial relations & background"
Image Content: A man standing on a sled pulled by several sled dogs, next to a small wooden house.
Instruction: Place the house behind the man, and make it much larger.
Edited Description: A man standing on a sled pulled by several sled dogs, in front of a large wooden house.

"viewpoint"
Image Content: A picture of a salad bowl.
Instruction: Change the perspective to a bird's-eye view.
Edited Description: A salad bowl viewed from above. 
 
The edited description needs to be as simple as possible. The instruction does not need to explicitly indicate which
type it is. Avoid adding imaginary things. Each time generate one instruction and one edited description only. Keep the edited description as short as possible. Here are some more examples for reference:

Image Content: the man is wearing a white tank top and black shorts
Instruction: looks faded and cheaper and is longer
Edited Description: The man is wearing a faded, cheap-looking and elongated white tank top, giving a worn-out slightly oversized and more casual appearance.

Image Content: the only winning move is not to play t shirt
Instruction: The shirt is black with a skeleton and is red
Edited Description: The image shows a black t-shirt with a red skeleton design on it. Is says "the only winning move is not to play".

Image Content: the man is wearing a black polo shirt
Instruction: is less formal with less buttons and is gray with no collar
Edited Description: The man is wearing a less formal, gray polo shirt with no collar and fewer buttions, giving it a more casual and relaxed appearance.

Image Content: the legend of zelda ocarina of time t-shirt
Instruction: is green and a graphic on it and is green
Edited Description: the shirt is green with an image of link holding a sword

Image Content: The woman is wearing a tan shirt and jeans
Instruction: is the same and appears to be exactly the same
Edited Description: The woman is wearing a tan shirt and jeans

Image Content: The woman is wearing a green top with hearts on it
Instruction: is pinched more below the bust and is brown in color
Edited Description: The woman is wearing a brown top with hearts on it, which is pinched more below the bust.

Image Content: the woman is wearing a yellow tank top
Instruction: is very similar but deeper red and is red
Edited Description: The woman is wearing a deep red tank top.

Image Content: the woman is wearing a polka dot dress
Instruction: has thinner straps and is darker
Edited Description: The woman is wearing a darker polka dot dress with thin straps, giving it a more delicate, elegant look.

Image Content: the dress is a silver sequined one shoulder dress
Instruction: is lighter colored and less fitted and is a light pink color
Edited Description: the dress is a one shoulder chiffon dress with a ruffled skirt

Image Content: black and yellow hawaiian floral print sleeveless hawaiian hawaiian 
Instruction: Sexier and no vibrant colors and less revealing chest and more evening wear
Edited Description: the dress is a black and green dress with a sleeveless bodice and a flared skirt
'''


###################################
########## NEW PROMPTS ############
###################################

mllm_structural_predictor_prompt_CoT = '''
- You are an image description expert. You are given an original image and manipulation text.
- Your goal is to generate a target image description that reflects the changes described based on manipulation intents while retaining as much image content from the original image as possible.
- You should carefully generate an image description of the target image with a thought of your understanding of the manipulation intents.

## Guidelines on generating the Original Image Description
    - Ensure that the original image description is thorough and detailed, capturing all visible objects, attributes, and elements. Specific attention should be given to any objects breeds, relationships, color, scenes, and the overarching domain of the image to provide a complete understanding.
    - The original image description should be as accurate as possible, reflecting the content and context of the image.

## Guidelines on generating the Thoughts
    - In your Thoughts, explain your understanding of the manipulation intents and how you formulated the target image description.
    - Provide insight into how you interpreted the manipulation intent detailed in the manipulation text, considering various semantic aspects.
    - Conclude with how these understandings were utilized to formulate the target image description, ensuring a logical and visually coherent transformation.

    ### Guidelines on generating the Reflections
    - In your Reflections, summarize how the manipulation intent influenced your approach to transforming the original image description.
    - Explain how the changes made reflect the specific semantic aspects involved, such as addition, negation, spatial relations, or viewpoint.
    - Highlight key decisions that were made to preserve the coherence and context of the original image while meeting the manipulation intent.
    - Reflect on the impact these changes have on the overall appearance or narrative of the image.
    - Ensure that your reflections provide a concise yet insightful summary of the considerations and strategies applied in crafting the target image description, offering a logical connection between the original content and the final description.

## Guidelines on generating Target Image Description
    - The target image description you generate should be complete and can cover various semantic aspects, such as cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint. Use the examples below as reference for these aspects:
    - The target image description only contain the target image content and needs to be as simple as possible. The instruction does not need to explicitly indicate which type it is. Minimize aesthetic descriptions as much as possible. Each time generate one target image description only. Keep the target image description as short as possible. Here are some more examples for reference:

## On the input format <Input>
- Input consist of two parts: The original image and the manipulation text.
{
    "Original Image": <image_url>,
    "Manipulation text": <manipulation_text>.
}
    - The original image is a URL provided in the image_url field of the user content data type, which furnishes the content of the original image.
    - The manipulation text is the text that describes the changes to be made to the original image.

## Guidelines on determining the response <Response>
- Responses consist of Four parts: the Original Image Context, Thoughts, Reflections, and the Target Image Description.
{
    "Original Image Description": <original_image_description>,
    "Thoughts": <thoughts>,
    "Reflections": <reflections>,
    "Target Image Description": <target_image_description>
}

Here are some more examples for reference:

## Example 1
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "Shows a dog of the same breed with darker brown fur laying in the grass with a woven basket of flowers."
}
<Response>
{
    "Original Image Description": "The image shows a man in a green field with a dog of Hungarian Vizsla breed. The man, wearing a dark green jacket and cap, is standing while the dog is pointing, indicating a hunting or tracking activity. The background is a grassy field extending to the horizon with trees faintly visible in the distance. The atmosphere appears calm, with an overcast sky providing soft lighting for the scene.",
    "Thoughts": "The manipulation intent is to change the scene by introducing a dog of Hungarian Vizsla breed but with darker brown fur, and to have it laying in the grass with a woven basket of flowers. This involves altering the dog's position and color, as well as adding a new element, the basket of flowers, to the scene.",
    "Reflections": "I reflect that the manipulation focuses on changing the dog's appearance and activity while adding a new object to the scene, which shifts the focus from the interaction between the person and the dog to a more serene and decorative setting.",
    "Target Image Description": "A Hungarian Vizsla dog with darker brown fur is laying in the grass next to a woven basket of flowers."
}

## Example 2
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "Focus on a single bird with open wings and get closer"
}
<Response>
{
    "Original Image Description": "The image shows a group of black vultures perched on a metal structure. The birds are gathered closely together, with two of them touching beaks, forming a heart shape. Several birds have their wings folded, while one is displaying open wings. The background features a blurred landscape, suggesting an elevated or open area.",
    "Thoughts": "The manipulation intent is to focus on a single bird with open wings and to get closer, which suggests a change in both the subject focus and the perspective. The original image features a group of vultures, but the manipulation requires isolating one bird, particularly one with open wings, and zooming in to provide a closer view. This would emphasize the details of the bird's wings and features, shifting the focus from the group to an individual bird.",
    "Reflections": "I reflect that the manipulation intent involves changing the focus from a group to a single bird, enhancing the details and features of the bird by getting closer, which alters the narrative from a collective scene to an individual focus.",
    "Target Image Description": "A single black vulture with open wings is shown up close."
}

## Example 3
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "Replace entire content with saloon spot with man and boy."
}
<Response>
{
    "Original Image Description": "The image features two miniature Schnauzer dogs standing on a mat near a glass door. One dog has a red collar, and they are positioned close to each other, facing in the same direction, with one appearing to sniff the other. The background includes a folding chair visible outside through the glass door, a curtain with decorative patterns, and part of an indoor cabinet with household items.",
    "Thoughts": "The manipulation intent was to replace the original content featuring two dogs with an entirely different setting involving a barbershop where a man and a young boy are present. This includes changing the focus from pets to human characters, transforming the context into a typical barbershop scene. The new background features elements such as barber tools, a window with blinds, and a more human-oriented environment that provides a sense of familiarity and warmth. This manipulation significantly shifts the original focus and dynamics, changing both the subjects and the setting entirely to create a human-centered narrative.",
    "Reflections": "I reflect that the manipulation intent involved creating an entirely different scene by introducing human subjects and a barbershop environment. This required me to focus on capturing the new setting details, including the interaction between the man and the boy, the objects they interact with, and the new atmosphere that evokes a sense of everyday life.",
    "Target Image Description": "A man giving a young boy a haircut in a barbershop."
}
'''

image_mllm_structural_predictor_prompt_CoT = '''
Your task is to modify the reference image based on the modification instructions and generate the updated image description. The description should be complete and can cover various semantic aspects, such as cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint.

To complete the task accurately, please follow these steps:
### Understand the Reference Image ###
1. Identify all the objects, attributes, and their relationships in the image.
2. Pay attention to the spatial relations, background, viewpoint in the image.
3. Please complete this task step by step.

### Analyze the Modification Instructions ###
1. Break down the modification instructions into separate modification steps.
2. Determine which objects or attributes need to be modified and how.
3. Pay attention to any additions, deletions, or changes to attributes.
4. Please complete this task step by step.

### Apply the Modifications###
1. Apply the modifications step by step to update the content of the reference image.

### Generate the Target Image Caption ###
1. Write a coherent and concise image caption.
2. Ensure the caption accurately reflects all the modifications.
3. The edited caption needs to be as simple as possible.
4. Do not mention the content that will not be present in the target image.

## On the input format <Input>
- Input consist of two parts: The reference image and the modification instructions.
{
    "Reference Image": <image_url>,
    "Modification Text": <modification_text>.
}
    - The reference image is a URL provided in the image_url field of the user content data type, which furnishes the content of the reference image.
    - The modification instructions is the text that describes the changes to be made to the reference image.

## Guidelines on determining the response <Response>
- Response is the Target Image Description.
{
    "Target Image Description": <target_image_description>
}

Here are some more examples for reference:

## Example 1
<Input>
{
    "Reference Image": <image_url>,
    "Modification Text": "add one more deer and add some sunlight."
}
<Response>
{
    "Target Image Description": "Two deer are standing in a sunlit grassy field."
}

## Example 2
<Input>
{
    "Reference Image": <image_url>,
    "Modification Text": "Focus on a single bird with open wings and get closer"
}
<Response>
{
    "Target Image Description": "A close-up of a single black vulture centered with wings spread against a softly blurred background."
}

## Example 3
<Input>
{
    "Reference Image": <image_url>,
    "Modification Text": "Replace entire content with saloon spot with man and boy in window wal background effect."
}
<Response>
{
    "Target Image Description": "A man giving a young boy a haircut in a barbershop."
}
'''

mllm_structural_predictor_prompt_CoT_multi = '''
You are an advanced reasoning expert tasked with generating potential target image descriptions for Composed Image Retrieval.

The relationship between reference image ($I_r$), modification text ($T_m$), and target image ($I_t$) is complex. The target image should be a coherent scene satisfying $T_m$, while selectively retaining relevant elements from $I_r$. Correctly determining *which* elements to retain, especially when $T_m$ involves implicit spatial, viewpoint, or scale changes, is challenging due to inherent ambiguity.

To systematically explore the ambiguity of user intent and maximize retrieval coverage, your task is to generate three distinct target query descriptions by following three complementary reasoning paths, representing different hypotheses about element retention. Each description must be concise, objective, and focused on retrievable visual content.

**Please complete this task STEP-BY-STEP.**
---------------------------------------------
### **Step 1: Analyze the Multi-Modal Queries**
Perform a comprehensive analysis of both inputs:
1.  **Understand the Reference Image ($I_r$)**:
    * Identify primary objects, their attributes, and spatial relationships.
    * Analyze scene context, background elements, and environmental setting.
    * Determine core visual elements and potential noise elements.
2.  **Analyze the Modification Text ($T_m$)**:
    * Extract explicit instructions: additions, removals, attribute changes.
    * Identify implicit requirements: perspective shifts (viewpoint, scale), theme changes, causal implications.
    * Detect potential conflicts between modification intent and reference elements.

---------------------------------------------
### **Step 2: Generate Query 1 - The Conservative Path ($Q_{conservative}$)**
This path follows a Minimal Core Hypothesis: It prioritizes the explicit modification ($T_m$) and core subject, assuming most other reference ($I_r$) context is irrelevant.
1.  **Reasoning Strategy (Minimal Core Hypothesis)**:
    * Start with the content **explicitly described in $T_m$**.
    * Identify and include *only* the **absolute essential subject(s)** from $I_r$ that are necessary for $T_m$ to be meaningful.
    * **Assume high potential for conflict or irrelevance**: Aggressively **exclude nearly all other contextual and attribute details** from $I_r$, particularly if any spatial/viewpoint/scale shift is detected in $T_m$.
    * This path represents a "safe" interpretation focusing purely on the modification and core entity.
2.  **Generate $Q_{conservative}$**: Create a concise description based on this **minimal core hypothesis** after thinking step-by-step, adhering to the Universal Principles in Step 5.

---------------------------------------------
### **Step 3: Generate Query 2 - The Balanced Path ($Q_{balanced}$)**
This path follows a **Standard Filtering Hypothesis**: It assumes the user intends a direct application of $T_m$ while removing all clear and implicit contradictions.
1.  **Reasoning Strategy (Standard Filtering Hypothesis)**:
    * Apply all explicit modifications from $T_m$ precisely.
    * Remove reference elements that **directly and obviously conflict** with $T_m$ (primarily thematic or clear attribute contradictions).
    * **Exclude reference elements whose necessity or compatibility is unclear** after applying explicit changes.
    * Focus on creating a **clean, conflict-free description** based on this direct compatibility check, without deep inference about implicit spatial/causal effects.
2.  **Generate $Q_{balanced}$**: Create a concise description based on this **standard filtering hypothesis** after thinking step-by-step, adhering to the Universal Principles in Step 5.

---------------------------------------------
### **Step 4: Generate Query 3 - The Reasoning-Enhanced Path ($Q_{reasoning}$)**
This path follows a **Coherent Inference Hypothesis**: It assumes the user desires a logically consistent and contextually plausible scene, requiring inference about the implicit consequences of $T_m$.
1.  **Reasoning Strategy (Coherent Inference Hypothesis)**:
    * Apply all explicit modifications from $T_m$.
    * **Actively infer plausible consequences**: Analyze $T_m$ for implicit triggers (spatial, causal, thematic). Reason about their likely effects on the scene's composition and element visibility (e.g., significant perspective changes often alter what background elements are visible).
    * **Modify based on inference**: Proactively remove $I_r$ elements that become logically implausible due to inferred consequences. Retain $I_r$ context *only if* it remains plausible within the inferred target scene. Consider if the inferred theme or action suggests plausible addition/modification of related elements grounded in the input.
    * This path explores a more deeply reasoned interpretation striving for logical and contextual coherence.
2.  **Generate $Q_{reasoning}$**: Create a concise description based on this **inference-driven hypothesis** after thinking step-by-step, adhering to the Universal Principles in Step 5.

---------------------------------------------
### **Step 5: Universal Generation Principles**
All three queries MUST strictly adhere to these principles:
1.  **Completeness**: The target image description should be complete and can cover various semantic aspects, such as cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint.
2.  **Simplicity**: The target image description only contain the target image content and needs to be as simple as possible. The instruction does not need to explicitly indicate which type it is.
3.  **Objectivity**: Minimize aesthetic descriptions as much as possible.
4.  **Conciseness**: For each target image, generate exactly three distinct descriptions (one from each reasoning path). Each description must be a single, concise sentence and kept as short as possible.
5.  **Factual Accuracy**: Ensure the description accurately reflects the intended modifications.
6.  **Conflict Avoidance**: Do not mention content that will not be present in the target image.
7. Specifically, all three descriptions must represent plausible, coherent interpretations of the combined multi-modal input ($I_r$ + $T_m$). They are designed to explore different hypotheses of user intent, but none must ever contradict the explicit instructions in $T_m$.

## On the input format <Input>
- Input consist of two parts: The reference image and the modification text.
{
    "Reference Image": <image_url>,
    "Modification Text": <modification_text>.
}

    - The reference image is a URL provided in the image_url field of the user content data type, which furnishes the content of the reference image.
    - The modification instructions is the text that describes the changes to be made to the reference image.

## Guidelines on determining the response <Response>
- Response is a JSON object containing the three generated queries with brief rationales.
{
    "Conservative Query": {
        "description": "<Target Image Description from Step 2>",
        "rationale": "Brief explanation of conservative approach"
    },
    "Balanced Query": {
        "description": "<Target Image Description from Step 3>", 
        "rationale": "Brief explanation of balanced approach"
    },
    "Reasoning Enhanced Query": {
        "description": "<Target Image Description from Step 4>",
        "rationale": "Brief explanation of reasoning-enhanced approach"
    }
}


Here are some more examples for reference:

## Example 1
<Input>
{
    "Reference Image": <image_url>,
    "Modification Text": "has only one person wearing the same outfit, the photo is zoomed in."
}
<Response>
{
    "Conservative Query": {
        "description": "A close-up of one man wearing a red tie.",
        "rationale": "This query focuses only on the explicit instructions ('one person', 'zoomed in') and the single retained element from the 'same outfit' (the red tie), ignoring all other context like shirt color or background."
    },
    "Balanced Query": {
        "description": "A zoomed-in photo of one man wearing a white shirt and a red tie.", 
        "rationale": "This query applies the 'one person' and 'zoomed in' instructions, assuming the original kitchen background is irrelevant. It retains the core components of the 'same outfit' (white shirt, red tie) from the reference image."
    },
    "Reasoning Enhanced Query": {
        "description": "A close-up of a man's face with a pouty expression, wearing a white shirt and red tie.",
        "rationale": "This query infers that 'zoomed in' implies a 'close-up' focusing on the person's 'face'. It follows 'one person', retains the 'same outfit' (white shirt, red tie), and hypothesizes a specific facial expression ('pouty expression') as a plausible detail for a close-up shot."
    }
}

## Example 2
<Input>
{
    "Reference Image": <image_url>,
    "Modification Text": "is damaged and has the same shape."
}
<Response>
{
    "Conservative Query": {
        "description": "A damaged octagonal stop sign.",
        "rationale": "This query only includes the essential subject (stop sign), the retained attribute (octagonal shape), and the explicit modification (damaged)."
    },
    "Balanced Query": {
        "description": "A rusted stop sign with the same octagonal shape, surrounded by bushes.", 
        "rationale": "This query interprets 'damaged' as a specific plausible type of damage ('rusted'). It retains the 'same shape' and removes the conflicting reference background (street signs, pole), hypothesizing a new plausible background ('surrounded by bushes')."
    },
    "Reasoning Enhanced Query": {
        "description": "An old, rusted octagonal stop sign with faded letters, set in bushes.",
        "rationale": "This query infers a coherent scene based on 'damaged'. 'Damaged' is interpreted as 'old' and 'rusted', which logically implies secondary effects like 'faded letters'. It replaces the reference background with a new, plausible setting ('set in bushes') consistent with a neglected sign."
    }
}

## Example 3
<Input>
{
    "Reference Image": <image_url>,
    "Modification Text": "A dog of a different breed shown with a jolly roger."
}
<Response>
{
    "Conservative Query": {
        "description": "A dog of a different breed shown with a jolly roger.",
        "rationale": "This query focuses only on the explicit instructions: replacing the dog's breed and adding the jolly roger symbol. It discards all other context from the reference image, such as the hat and the basket."
    },
    "Balanced Query": {
        "description": "A dog of a different breed wearing a hat with a jolly roger.", 
        "rationale": "This query applies the 'different breed' instruction and infers that the 'jolly roger' (pirate theme) directly conflicts with and replaces the reference image's 'cowboy hat'. It removes the non-essential basket context."
    },
    "Reasoning Enhanced Query": {
        "description": "A dog of a different breed in a pirate costume with a jolly roger hat, next to a jug.",
        "rationale": "This query infers that 'jolly roger' implies a full thematic change from 'cowboy' to 'pirate'. This justifies replacing all cowboy-related items (hat, garment) with a pirate costume and hat, and adding a thematically-related object like a jug."
    }
}

## Example 4
<Input>
{
    "Reference Image": <image_url>,
    "Modification Text": "shows the cork on a table and has a more everyday background."
}
<Response>
{
    "Conservative Query": {
        "description": "A glass of wine, a wine bottle, and a cork on a table.",
        "rationale": "This query retains only the essential subjects from the reference (glass, bottle) and adds the elements explicitly requested by the modification (cork, table). It aggressively discards all other context like brand, bottle color, and background."
    },
    "Balanced Query": {
        "description": "A glass of red wine, a wine bottle, and a cork on a table with an everyday background.", 
        "rationale": "This query applies the modifications (cork, table, everyday background) and filters out conflicting reference elements. The 'plain background' is replaced, and the specific 'Javelina Leap' brand is assumed to be non-essential context and is discarded."
    },
    "Reasoning Enhanced Query": {
        "description": "An etched glass of red wine, a matching branded bottle, and its cork on a table with an indoor background.",
        "rationale": "This query infers that 'shows the cork' and 'everyday background' imply a new, complete, and coherent scene. It reasons that the 'cork' must match the 'bottle', and that the 'glass' would also be part of this branded set (etched), thus replacing all specific brand context from the reference."
    }
}
'''


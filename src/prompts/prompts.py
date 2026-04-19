###################################
###### GENERAL PROMPTS ############
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

mllm_structural_predictor_prompt_CoT_v1 = '''
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
    "Manipulation text": "has a dog of a different breed and shows a jolly roger."
}
<Response>
{
    "Original Image Description": "The image shows a small long-haired dog resting in a cart outdoors, wearing a bright red cowboy hat with trim; close-up view with metal bars, a towel, and street elements in the background.",
    "Thoughts": "The manipulation intent is to show a dog of a different breed and include a clear jolly roger emblem, so I will replace the shaggy dog with another breed and swap the cowboy hat for pirate attire bearing a skull-and-crossbones while keeping a simple, front-facing single-dog composition.",
    "Reflections": "I reflect that the change targets breed and costume rather than scene; to satisfy the jolly roger requirement the emblem must be prominent on the hat, while simplifying the background and allowing an adjacent prop for context without distracting from the subject.",
    "Target Image Description": "A pug in a pirate costume wears a hat with a skull-and-crossbones jolly roger beside a large brown glass bottle against a light backdrop."
}

## Example 2
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "shows two people and has a more colorful background."
}
<Response>
{
    "Original Image Description": "The image shows a single performer in traditional dress holding a red paper parasol on a dark indoor stage under a warm spotlight; full-body view with minimal background detail.",
    "Thoughts": "The manipulation intent is to show two people and have a more colorful background, so add a second performer and replace the dark backdrop with a vivid multicolored one while preserving the indoor stage context and traditional attire with red parasols.",
    "Reflections": "I reflect that the changes shift cardinality from one to two and convert a plain background to a bright multicolored backdrop; to keep coherence, retain traditional clothing and parasols and ensure both performers are clearly visible in the same stage setting.",
    "Target Image Description": "Two performers in pink traditional dress hold red paper parasols against a bright multicolored backdrop."
}

## Example 3
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "has only one person wearing the same outfit, the photo is zoomed in."
}
<Response>
{
    "Original Image Description": "The image shows four men indoors wearing matching outfits—white dress shirts, red ties, and beige pants—each holding a green bottle in a bright home interior; wide group shot with all standing.",
    "Thoughts": "The manipulation intent is to keep only one person wearing the same outfit and to zoom in, which requires reducing the group to a single subject in a light shirt with a red tie and reframing to a tight close-up that minimizes background detail.",
    "Reflections": "I reflect that the manipulation changes cardinality from four to one and viewpoint from a wide group shot to a close-up; to maintain consistency, retain the shirt-and-red-tie outfit, remove the other people, and limit background distractions.",
    "Target Image Description": "A close-up of a single man wearing a light shirt and a red tie indoors."
}

'''

###################################
######### CoTMR PROMPTS ###########
###################################

# Object Scale
object_mllm_structural_predictor_prompt_CoT = '''
You are provided with two inputs:
    - Reference Image: The image that will be modified.
    - Modification Text: Instructions that specify changes to be applied to the reference image.

Your goal is to:
    1. Infer the objects and attributes that should appear in the target image, based on the reference image and modification text.
    2. Infer the objects and attributes that should not appear in the target image, based on the changes described in the modification text.
    3. Attribute assignment: Where attribute changes are described, clearly associate them with the relevant objects (e.g., color change of a shirt).

To complete the task accurately, please follow these steps:
### Describe the Reference Image ###
- List the objects and their attributes present in the reference image step-by-step .

### Understand the Modification Instructions ###
- Analyze modification instruction step-by-step to identify changes to objects and attributes, including additions, deletions, or modifications.

### Apply the Modifications ###
1. Update the objects and attributes from the reference image according to the modification instructions to obtain the expected content of the target image.
2. Please complete this task step by step.

### Determine the Content of the Target Image ###
- Existent Object (Objects and Attributes that Must Exist):
    1. List the objects and attributes that must be present in the target image.
    2. Be specific, especially if attributes are provided in the modification text.
- Nonexistent Object (Objects and Attributes that Must Not Exist):
    1. List the objects and attributes that must not be present in the target image.
    2. Include any objects or attributes explicitly removed or modified to no longer exist.

## On the input format <Input>
- Input consist of two parts: The reference image and the modification instructions.
{
    "Reference Image": <image_url>,
    "Modification Text": <modification_text>.
}
    - The reference image is a URL provided in the image_url field of the user content data type, which furnishes the content of the reference image.
    - The modification instructions is the text that describes the changes to be made to the reference image.

## Guidelines on determining the response <Response>
- Responses consist of two parts: Existent Object, and Nonexistent Object.
{
    "Existent Object": <existent_object>
    "Nonexistent Object": <nonexistent_object>
}

Here are some more examples for reference:

## Example 1
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "add one more deer and add some sunlight."
}

<Response>
{
    "Existent Object": ["Two deer", "Brown color", "Long and curved horns", "Green grass field", "Sunlight"],
    "Nonexistent Object": [],
}

## Example 2
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "Change the cart to a white, unmanned carriage in daylight with no horses."
}
<Response>
{
    "Existent Object": ["white carriage", "daylight"],
    "Nonexistent Object": ["horses", "people"],
}

## Example 3
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "Shows a smaller, similarly shaped dog with lighter brown fur standing on stone tile path."
}
<Response>
{
    "Existent Object": ["small dog", "lighter brown fur", "stone tile path"],
    "Nonexistent Object": ["white fur", "lying down", "brown patches"],
}

'''


# Image Scale
image_mllm_structural_predictor_prompt_CoT_v0 = '''
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
5. The target image description you generate should be complete and can cover various semantic aspects, such as cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint.
6. The target image description only contain the target image content and needs to be as simple as possible. The instruction does not need to explicitly indicate which type it is. Minimize aesthetic descriptions as much as possible. Each time generate one target image description only. Keep the target image description as short as possible.

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

# Image Scale
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


# 效果很一般，无论是 show the reasoning process，还是也生成中间推理过程，效果都不是很好。
image_mllm_structural_predictor_prompt_CoT_v1 = '''
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

### Apply the Modifications ###
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
- Responses consist of Four parts: Understand the Reference Image, Analyze the Modification Instructions, Apply the Modifications, and the Target Image Description.
{
    "Understand the Reference Image": <understand_the_reference_image>,
    "Analyze the Modification Text": <analyze_the_modification_text>,
    "Apply the Modifications": <apply_the_modifications>,
    "Target Image Description": <target_image_description>
}

Here are some more examples for reference:

## Example 1
<Input>
{
    "Reference Image": <image_url>,
    "Modification Text": "has a dog of a different breed and shows a jolly roger."
}
<Response>
{
    "Understand the Reference Image": "The image shows a small long-haired dog in a cart wearing a red cowboy hat; close-up outdoors.",
    "Analyze the Modification Text": "'has a dog of a different breed' means replace the dog. 'shows a jolly roger' means a skull-and-crossbones must be visible."
    "Apply the Modifications": "Switch to another breed; add pirate hat with a clear jolly roger; keep single-dog focus and simple background."
    "Target Image Description": "A different-breed dog wears a pirate hat with a skull-and-crossbones jolly roger."
}

## Example 2
<Input>
{
    "Reference Image": <image_url>,
    "Modification Text": "shows two people and has a more colorful background."
}
<Response>
{
    "Understand the Reference Image": "The image shows one performer in traditional dress holding a red parasol on a dark indoor stage; full-body view.",
    "Analyze the Modification Text": "'shows two people' means add a second performer. 'has a more colorful background' means replace the dark backdrop with a vivid multicolored one.",
    "Apply the Modifications": "Place a second performer beside the first; keep traditional attire and one red parasol; swap in a bright multicolored backdrop.",
    "Target Image Description": "Two performers in pink traditional dress hold red paper parasols against a bright multicolored backdrop."
}

## Example 3
<Input>
{
    "Reference Image": <image_url>,
    "Modification Text": "has only one person wearing the same outfit, the photo is zoomed in."
}
<Response>
{
    "Understand the Reference Image": "The image shows four men indoors in matching outfits—white shirts, red ties, beige pants—each holding a bottle; wide group shot.",
    "Analyze the Modification Text": "'has only one person wearing the same outfit' means keep one subject in white shirt and red tie. 'photo is zoomed in' means a tight close-up.",
    "Apply the Modifications": "Remove the others; frame head-and-shoulders; retain white shirt and red tie; minimize background detail.",
    "Target Image Description": "A close-up of a single man wearing a white shirt and a red tie indoors."
}

'''


# Image Scale 仿照 conservative_path 生成单查询，但效果并不好。所以强制 MLLM 生成多查询，也可以导致查询多样化，从而有助于提升检索性能
image_mllm_structural_predictor_prompt_CoT_new = '''
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
1. Apply **only** the explicit modifications identified in Step 2 precisely.
2. **Remove** any reference elements identified as directly conflicting in Step 2.
3. **Crucially, exclude** any reference elements whose necessity for the target scene is **not explicitly required** by modification instructions or whose compatibility after the modifications is **unclear**. Focus only on elements confirmed by modification instructions or absolutely essential context.
4. Aim to construct a **clean, conflict-free** mental representation of the target scene.

### Generate the Target Image Caption ###
1.  **Completeness**: The target image description should be complete and can cover various semantic aspects, such as cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint.
2.  **Simplicity**: The target image description only contain the target image content and needs to be as simple as possible. The instruction does not need to explicitly indicate which type it is.
3.  **Objectivity**: Minimize aesthetic descriptions as much as possible.
4.  **Conciseness**: For each target image, the target query description must be a single, concise sentence and kept as short as possible.
5.  **Factual Accuracy**: Ensure the description accurately reflects the intended modifications.
6.  **Conflict Avoidance**: Do not mention content that will not be present in the target image.

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


target_mllm_structural_predictor_prompt_CoT = '''
Your task is to generate the Target Image Description only.

To complete the task accurately, please follow these steps:
### Understand the Target Image ###
1. Identify all the objects, attributes, and their relationships in the image.
2. Pay attention to the spatial relations, background, viewpoint in the image.
3. Please complete this task step by step.

### Generate the Target Image Caption ###
1. Write a coherent and concise image caption, which needs to be as simple as possible.
2. Do not mention the content that will not be present in the target image.
3. The target image description you generate should be complete and can cover various semantic aspects, such as cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint.
4. The target image description only contain the target image content and needs to be as simple as possible. The instruction does not need to explicitly indicate which type it is. Minimize aesthetic descriptions as much as possible. Each time generate one target image description only. Keep the target image description as short as possible.

## On the input format <Input>
- Input is the Target Image.
{
    "Target Image": <image_url>
}
    - The target image is a URL provided in the image_url field of the user content data type, which furnishes the content of the target image.

## Guidelines on determining the response <Response>
- Response is the Target Image Description.
{
    "Target Image Description": <target_image_description>
}

Here are some more examples for reference:

## Example 1
<Input>
{
    "Target Image": <image_url>
}
<Response>
{
    "Target Image Description": "Two deer are standing in a sunlit grassy field."
}

## Example 2
<Input>
{
    "Target Image": <image_url>
}
<Response>
{
    "Target Image Description": "A close-up of a single black vulture centered with wings spread against a softly blurred background."
}

## Example 3
<Input>
{
    "Target Image": <image_url>
}
<Response>
{
    "Target Image Description": "A man giving a young boy a haircut in a barbershop."
}

'''


fiq_target_mllm_structural_predictor_prompt_CoT = '''
Your task is to generate the Target Image Description only.

To complete the task accurately, please follow these steps:
### Understand the Target Image ###
1. Identify the **main fashion product** (e.g., dress, shirt, toptee).
2. Pay attention to **category/type, style & fit, color, silhouette & length, neckline & sleeves/collar, hem, pattern/print, fabric/finish (e.g., denim/satin/ribbed), closures/embellishments (e.g., buttons/zipper/ruffles), and logo/graphic/text details (presence, placement, scale, style, color)**.
3. Please complete this task step by step.

### Generate the Target Image Caption ###
1. Write a coherent and concise image caption, which needs to be as simple as possible.
2. Do not mention the content that not be present in the target image.
3. The target image description you generate should be complete for fashion attributes when visible, such as: category/type, color, style & fit, silhouette & length, neckline & sleeves/collar, hem, pattern/print, fabric/finish, closures/embellishments, and logo/graphic/text details (presence, placement, scale, color). Mention viewpoint only when necessary (e.g., back view, close-up).
4. The target image description only contain the target image content and needs to be as simple as possible. The instruction does not need to explicitly indicate which type it is. Minimize aesthetic descriptions as much as possible. Each time generate one target image description only. Keep the target image description as short as possible.

## On the input format <Input>
- Input is the Target Image.
{
    "Target Image": <image_url>
}
    - The target image is a URL provided in the image_url field of the user content data type, which furnishes the content of the target image.

## Guidelines on determining the response <Response>
- Response is the Target Image Description.
{
    "Target Image Description": <target_image_description>
}

Here are some more examples for reference:

## Example 1
<Input>
{
    "Target Image": <image_url>
}
<Response>
{
    "Target Image Description": "The woman is wearing a denim-like dress with a ruffled neckline, and a ruffled skirt."
}

## Example 2
<Input>
{
    "Target Image": <image_url>
}
<Response>
{
    "Target Image Description": "The woman is wearing a long, shiny black dress with an Asian-inspired design and short sleeves."
}

## Example 3
<Input>
{
    "Target Image": <image_url>
}
<Response>
{
    "Target Image Description": "is white colored and is less religious and more humorous."
}

'''


# FashionIQ prompts
fiq_image_mllm_structural_predictor_prompt_CoT = '''
- Your task is to modify the reference image based on the modification instructions and generate the updated image description.
- Modification strictly within the products such as dress, shirt, and toptee, keep the product type unless explicitly told to change it.
- Focus only on the garment and ignore models, footwear, jewelry, and other accessories unless explicitly requested.
- Use concise fashion terms and mention viewpoint only when necessary, apply edits to key apparel attributes—color, style and fit, silhouette and length, neckline and sleeves or collar, hem, pattern or print, fabric and finish, closures and embellishments—plus logo/graphic/text details including presence or removal, placement (chest/front/back/sleeve), scale, style, and color, keep the result realistic for the chosen category.

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
5. The target image description you generate should be complete and can cover various semantic aspects, such as cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint.
6. The target image description only contain the target image content and needs to be as simple as possible. The instruction does not need to explicitly indicate which type it is. Minimize aesthetic descriptions as much as possible. Each time generate one target image description only. Keep the target image description as short as possible.

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

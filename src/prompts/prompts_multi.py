# Gemini Prompt
deepseek_mllm_structural_predictor_prompt_CoT = '''
You are an advanced reasoning expert for Composed Image Retrieval. 

The relationship between reference image, modification text, and target image is complex - The target image is a coherent scene that satisfies the modification constraints, selectively retains relevant elements from the reference image, and may include new elements inferred from the modification's causal implications.

To systematically explore the ambiguity of user intent and maximize retrieval coverage, your task is to generate three distinct target query descriptions by following three complementary reasoning paths. Each description must be concise, objective, and focused on retrievable visual content.

**Please complete this task STEP-BY-STEP.**
---------------------------------------------
### **Step 1: Analyze the Multi-Modal Queries**
Perform a comprehensive analysis of both inputs that will inform all three query paths:
1.  **Understand the Reference Image ($I_r$)**:
    * Identify primary objects, their attributes, and spatial relationships
    * Analyze scene context, background elements, and environmental setting
    * Determine core visual elements and potential noise elements
2.  **Analyze the Modification Text ($T_m$)**:
    * Extract explicit instructions: additions, removals, attribute changes
    * Identify implicit requirements: perspective shifts, theme changes, causal implications
    * Detect potential conflicts between modification intent and reference elements

---------------------------------------------
### **Step 2: Generate Query 1 - Conservative Path ($Q_{conservative}$)**
This path prioritizes strict adherence to modification text while eliminating potential conflicts:
1.  **Reasoning Strategy**: 
    * Apply all explicit modifications from $T_m$ precisely
    * Remove any reference elements that directly conflict with $T_m$
    * Exclude elements that are not explicitly required or clearly compatible
    * Focus on creating a clean, conflict-free description
2.  **Generate $Q_{conservative}$**: Create a concise description based on this strict filtering approach after think step-by-step.

---------------------------------------------
### **Step 3: Generate Query 2 - Balanced Path ($Q_{balanced}$)**
This path represents the baseline approach, balancing modification execution with reasonable retention:
1.  **Reasoning Strategy**:
    * Apply modifications while maintaining continuity with reference
    * Preserve elements that are compatible with modification intent
    * Ensure logical coherence in the resulting scene
    * Maintain core spatial relationships where feasible
2.  **Generate $Q_{balanced}$**: Create a description that serves as the performance baseline after think step-by-step.

---------------------------------------------
### **Step 4: Generate Query 3 - Reasoning-Enhanced Path ($Q_{reasoning}$)**
This path extends reasoning beyond literal instructions to capture implicit requirements:
1.  **Reasoning Strategy**:
    * Infer causal consequences and logical implications of modifications
    * Identify theme shifts and ensure overall scene consistency
    * Include elements logically required by the modification context
    * Remove elements that become implausible due to inferred changes
2.  **Generate $Q_{reasoning}$**: Create a description that captures both explicit and implicit requirements after think step-by-step.

---------------------------------------------
### **Step 5: Universal Generation Principles**
All three queries MUST strictly adhere to these principles:
1.  **Completeness**: The target image description should be complete and can cover various semantic aspects, such as cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint.
2.  **Simplicity**: The target image description only contain the target image content and needs to be as simple as possible. The instruction does not need to explicitly indicate which type it is.
3.  **Objectivity**: Minimize aesthetic descriptions as much as possible.
4.  **Conciseness**: For each target image, generate exactly three distinct descriptions (one from each reasoning path). Each description must be a single, concise sentence and kept as short as possible.
5.  **Factual Accuracy**: Ensure the description accurately reflects the intended modifications.
6.  **Conflict Avoidance**: Do not mention content that will not be present in the target image.

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
    "Modification Text": "shows two people and has a more colorful background."
}
<Response>
{
    "Conservative Query": {
        "description": "Two people pose on a stage in front of a bright multicolored background.",
        "rationale": "Applies only the modifications—add a second person and require a colorful background—while omitting optional details to avoid conflicts."
    },
    "Balanced Query": {
        "description": "Two performers in traditional dress, one holding a red parasol, stand on a stage against a bright multicolored backdrop.", 
        "rationale": "Retains compatible elements from the reference (stage, traditional attire, parasol) while satisfying the added person and colorful background."
    },
    "Reasoning Enhanced Query": {
        "description": "Two performers in traditional long robes, each holding a parasol, dance on an indoor stage with a vivid multicolored backdrop.",
        "rationale": "Infers a coordinated duet with mirrored props and indoor staging to maintain scene coherence and emphasize the required colorful background."
    }
}

## Example 2
<Input>
{
    "Reference Image": <image_url>,
    "Modification Text": "has a dog of a different breed and shows a jolly roger."
}
<Response>
{
    "Conservative Query": {
        "description": "A dog of a different breed with a jolly roger flag clearly visible.",
        "rationale": "Applies only the modifications—change to a different dog breed and show a jolly roger—while omitting basket, hat, and street details to avoid conflicts."
    },
    "Balanced Query": {
        "description": "A different-breed dog in a pirate costume wearing a bandana printed with a skull-and-crossbones jolly roger..", 
        "rationale": "Keeps compatible context from the reference (costume style) while executing the required breed change and adding the jolly roger."
    },
    "Reasoning Enhanced Query": {
        "description": "A different-breed dog wearing a pirate hat and costume stands next to a large brown glass bottle.",
        "rationale": "Infers a pirate theme from the jolly roger, making the emblem a central, visible element while ensuring the dog is a different breed."
    }
}

## Example 3
<Input>
{
    "Reference Image": <image_url>,
    "Modification Text": "has plants instead of plungers."
}
<Response>
{
    "Conservative Query": {
        "description": "An open toilet bowl contains green plants instead of plungers.",
        "rationale": "Applies the replacement exactly and omits background, writing, and other optional details to avoid conflicts."
    },
    "Balanced Query": {
        "description": "An outdoor toilet repurposed as a planter, with leafy plants growing from the bowl beneath the open lid.", 
        "rationale": "Keeps compatible context from the reference (outdoor setting) while replacing plungers with plants."
    },
    "Reasoning Enhanced Query": {
        "description": "A toilet with a tree growing out of it, surrounded by plants and a stone wall.",
        "rationale": "Infers a coherent planter theme from the replacement and includes logically consistent elements like the open lid and outdoor setting."
    }
}

'''


# Conservative -> Balanced
deepseek_mllm_structural_predictor_prompt_CoT_v1 = '''
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


deepseek_mllm_structural_predictor_prompt_CoT_v2 = '''
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
- Response is a JSON object containing the three generated queries.
{
    "Conservative Query": {
        "description": "<Target Image Description from Step 2>"
    },
    "Balanced Query": {
        "description": "<Target Image Description from Step 3>"
    },
    "Reasoning Enhanced Query": {
        "description": "<Target Image Description from Step 4>"
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
        "description": "A close-up of one man wearing a red tie."
    },
    "Balanced Query": {
        "description": "A zoomed-in photo of one man wearing a white shirt and a red tie."
    },
    "Reasoning Enhanced Query": {
        "description": "A close-up of a man's face with a pouty expression, wearing a white shirt and red tie."
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
        "description": "A damaged octagonal stop sign."
    },
    "Balanced Query": {
        "description": "A rusted stop sign with the same octagonal shape, surrounded by bushes."
    },
    "Reasoning Enhanced Query": {
        "description": "An old, rusted octagonal stop sign with faded letters, set in bushes."
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
        "description": "A dog of a different breed shown with a jolly roger."
    },
    "Balanced Query": {
        "description": "A dog of a different breed wearing a hat with a jolly roger."
    },
    "Reasoning Enhanced Query": {
        "description": "A dog of a different breed in a pirate costume with a jolly roger hat, next to a jug."
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
        "description": "A glass of wine, a wine bottle, and a cork on a table."
    },
    "Balanced Query": {
        "description": "A glass of red wine, a wine bottle, and a cork on a table with an everyday background."
    },
    "Reasoning Enhanced Query": {
        "description": "An etched glass of red wine, a matching branded bottle, and its cork on a table with an indoor background."
    }
}

'''


deepseek_mllm_structural_predictor_prompt_CoT_weight_all = '''
You are an advanced reasoning expert tasked with generating potential target image descriptions for Composed Image Retrieval and assessing their relative plausibility.

The relationship between reference image ($I_r$), modification text ($T_m$), and target image ($I_t$) is complex. The target image should be a coherent scene satisfying $T_m$, while selectively retaining relevant elements from $I_r$. Correctly determining *which* elements to retain, especially when $T_m$ involves implicit spatial, viewpoint, or scale changes, is challenging due to inherent ambiguity.

To systematically explore plausible interpretations of the ambiguous user intent, maximizing retrieval coverage, your task is two-fold:
1.  **Generate**: Create three distinct target query descriptions ($Q_{conservative}$, $Q_{balanced}$, $Q_{reasoning}$) by following three complementary reasoning paths, representing different hypotheses about element retention.
2.  **Assess**: Critically evaluate the plausibility of each generated query as an accurate reflection of the *most likely* user intent behind the original inputs ($I_r, T_m$), assigning a confidence score (0.0-1.0) to each.

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
1.  **Intent Plausibility**: All three descriptions must represent plausible, coherent interpretations of the combined multi-modal input ($I_r$ + $T_m$). They are designed to explore different hypotheses of user intent, but none must ever contradict the explicit instructions in $T_m$.
2.  **Completeness**: The target image description should be complete and can cover various semantic aspects, such as cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint.
3.  **Simplicity**: The target image description only contain the target image content and needs to be as simple as possible. The instruction does not need to explicitly indicate which type it is.
4.  **Objectivity**: Minimize aesthetic descriptions as much as possible.
5.  **Conciseness**: For each target image, generate exactly three distinct descriptions (one from each reasoning path). Each description must be a single, concise sentence and kept as short as possible.
6.  **Factual Accuracy**: Ensure the description accurately reflects the intended modifications.
7.  **Conflict Avoidance**: Do not mention content that will not be present in the target image.

---------------------------------------------
### **Step 6: Evaluate Query Plausibility (CRITICAL Assessment)**
Critically evaluate the **plausibility** of each of the three queries you just generated ($Q_{conservative}$, $Q_{balanced}$, $Q_{reasoning}$) as an interpretation of the original user intent ($I_r, T_m$).
1.  **Evaluation Criteria**: For each query, consider:
    * **Adherence to Instructions**: Does it accurately follow all explicit instructions in $T_m$?
    * **Context Handling**: Is its strategy for retaining/pruning/inferring $I_r$ context reasonable given the specific input and potential ambiguities (like spatial shifts)?
    * **Scene Plausibility**: Does the resulting description depict a coherent and logical scene?
2.  **Assign Plausibility Scores (0.0 - 1.0)**:
    * Assign a float score between **0.0 (implausible interpretation)** and **1.0 (highly plausible interpretation)** to each query.
    * The score should reflect the **relative likelihood** that the query's underlying reasoning hypothesis correctly captures the user's probable intent for this specific $I_r, T_m$ pair.
    * Scores are independent estimates and **do not** need to sum to 1.
    * Provide a brief `evaluation_rationale` explaining the score based on the criteria above.
    
---------------------------------------------
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
        "rationale": "Brief explanation of conservative approach",
        "confidence_score": <float between 0.0 and 1.0>,
        "evaluation_rationale": "Brief justification for the score based on Step 6 analysis."
    },
    "Balanced Query": {
        "description": "<Target Image Description from Step 3>", 
        "rationale": "Brief explanation of balanced approach",
        "confidence_score": <float between 0.0 and 1.0>,
        "evaluation_rationale": "Brief justification for the score based on Step 6 analysis."
    },
    "Reasoning Enhanced Query": {
        "description": "<Target Image Description from Step 4>",
        "rationale": "Brief explanation of reasoning-enhanced approach",
        "confidence_score": <float between 0.0 and 1.0>,
        "evaluation_rationale": "Brief justification for the score based on Step 6 analysis."
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
        "confidence_score": 1.0,
        "evaluation_rationale": "Highest retrieval quality. This is the safest, lowest-noise query. It's robust even if the target image's 'white shirt' is obscured or slightly different."
    },
    "Balanced Query": {
        "description": "A zoomed-in photo of one man wearing a white shirt and a red tie.", 
        "rationale": "This query applies the 'one person' and 'zoomed in' instructions, assuming the original kitchen background is irrelevant. It retains the core components of the 'same outfit' (white shirt, red tie) from the reference image."
        "confidence_score": 0.7,
        "evaluation_rationale": "High retrieval quality. This query fully matches the text instructions. It's slightly less robust than the conservative query because 'white shirt' adds another specific constraint that could cause a mismatch."
    },
    "Reasoning Enhanced Query": {
        "description": "A close-up of a man's face with a pouty expression, wearing a white shirt and red tie.",
        "rationale": "This query infers that 'zoomed in' implies a 'close-up' focusing on the person's 'face'. It follows 'one person', retains the 'same outfit' (white shirt, red tie), and hypothesizes a specific facial expression ('pouty expression') as a plausible detail for a close-up shot."
        "confidence_score": 0.5,
        "evaluation_rationale": "Low retrieval quality. This query is high-risk. It makes two significant, un-guaranteed inferences: 1) 'zoomed in' means 'close-up of face', and 2) the retained person is the one with the 'pouty expression'. This is very brittle."
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
        "confidence_score": 1.0,
        "evaluation_rationale": "Highest retrieval quality. This query is minimal-noise and perfectly translates the text instructions ('damaged', 'same shape' -> octagonal) while retaining only the core subject (stop sign). It's the most robust query."
    },
    "Balanced Query": {
        "description": "A rusted stop sign with the same octagonal shape, surrounded by bushes.", 
        "rationale": "This query interprets 'damaged' as a specific plausible type of damage ('rusted'). It retains the 'same shape' and removes the conflicting reference background (street signs, pole), hypothesizing a new plausible background ('surrounded by bushes')."
        "confidence_score": 0.2,
        "evaluation_rationale": "Low retrieval quality. This query introduces significant, high-risk noise. It makes two un-guaranteed assumptions: 1) 'damaged' specifically means 'rusted', and 2) the new background is 'bushes'. This is very brittle."
    },
    "Reasoning Enhanced Query": {
        "description": "An old, rusted octagonal stop sign with faded letters, set in bushes.",
        "rationale": "This query infers a coherent scene based on 'damaged'. 'Damaged' is interpreted as 'old' and 'rusted', which logically implies secondary effects like 'faded letters'. It replaces the reference background with a new, plausible setting ('set in bushes') consistent with a neglected sign."
        "confidence_score": 0.6,
        "evaluation_rationale": "This query is high-risk, as it's a cascade of inferences ('damaged' -> 'old' -> 'rusted' -> 'faded letters' -> 'bushes'). However, it fulfills the 'Reasoning-Enhanced' goal by creating a *plausible, coherent, alternative scene*. This specificity is useful for diversifying retrieval results."
    }
}

## Example 3
<Input>
{
    "Reference Image": <image_url>,
    "Modification Text": "Has a dog of a different breed shown with a jolly roger."
}
<Response>
{
    "Conservative Query": {
        "description": "A dog of a different breed shown with a jolly roger.",
        "rationale": "This query strictly adheres to the two explicit instructions: replacing the dog's breed and adding a jolly roger. It discards all other context from the reference image (cowboy hat, garment, basket) as irrelevant noise."
        "confidence_score": 1.0,
        "evaluation_rationale": "Highest retrieval quality. This query is the least noisy and makes zero assumptions about *how* or *where* the jolly roger is displayed, maximizing robustness. This is the safest anchor query."
    },
    "Balanced Query": {
        "description": "A dog of a different breed wearing a hat with a jolly roger.", 
        "rationale": "This query applies the 'different breed' instruction and infers that the 'jolly roger' (pirate theme) directly conflicts with and replaces the reference image's 'cowboy hat'. It removes the non-essential basket context."
        "confidence_score": 0.2,
        "evaluation_rationale": "Low retrieval quality. This query is a high-risk 'half-measure'. It introduces a significant, un-guaranteed inference (that the jolly roger is on a hat) without the benefit of creating a fully distinct, coherent alternative scene. It's brittle and lacks diversity."
    },
    "Reasoning Enhanced Query": {
        "description": "A dog of a different breed in a pirate costume with a jolly roger hat, next to a jug.",
        "rationale": "This query infers that 'jolly roger' implies a full thematic change from 'cowboy' to 'pirate'. This justifies replacing all cowboy-related items (hat, garment) with a pirate costume and hat, and adding a thematically-related object like a jug."
        "confidence_score": 0.5,
        "evaluation_rationale": "This query is high-risk, but provides high-value diversity. It builds a *plausible, coherent, alternative scene* (pirate theme) based on the 'jolly roger' prompt. While brittle, it's useful for query expansion in combination with the conservative query."
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
        "confidence_score": 1.0,
        "evaluation_rationale": "Highest retrieval quality. This query is minimal-noise, directly follows all instructions, and discards all non-essential context. It provides the most robust and safe baseline for retrieval."
    },
    "Balanced Query": {
        "description": "A glass of red wine, a wine bottle, and a cork on a table with an everyday background.", 
        "rationale": "This query applies the modifications (cork, table, everyday background) and filters out conflicting reference elements. The 'plain background' is replaced, and the specific 'Javelina Leap' brand is assumed to be non-essential context and is discarded."
        "confidence_score": 0.3,
        "evaluation_rationale": "Low retrieval quality. This query creates noise by adding the new, subjective, and hard-to-query term 'everyday background'. This mix of old and new context is less robust."
    },
    "Reasoning Enhanced Query": {
        "description": "An etched glass of red wine, a matching branded bottle, and its cork on a table with an indoor background.",
        "rationale": "This query infers that 'shows the cork' and 'everyday background' imply a new, complete, and coherent scene. It reasons that the 'cork' must match the 'bottle', and that the 'glass' would also be part of this branded set (etched), thus replacing all specific brand context from the reference."
        "confidence_score": 0.7,
        "evaluation_rationale": "This query is high-risk and full of assumptions (etched, matching brand). However, it represents a *plausible, coherent, alternative scene* rather than a simple edit. This specificity, while brittle, is useful for diversifying retrieval results."
    }
}

'''


deepseek_mllm_structural_predictor_prompt_CoT_weight = '''
You are a meticulous evaluator specializing in Composed Image Retrieval query interpretation. You are given the original user query, consisting of a Reference Image ($I_r$) and Modification Text ($T_m$), along with three potential interpretations generated by different reasoning paths: a Conservative Query ($Q_{conservative}$), a Balanced Query ($Q_{balanced}$), and a Reasoning Enhanced Query ($Q_{reasoning}$).

Your task is to critically assess the **plausibility** of each of the three generated queries ($Q_{conservative}$, $Q_{balanced}$, $Q_{reasoning}$) as an accurate reflection of the *most likely* user intent behind the original $I_r, T_m$ input. You must output a confidence score (a float between 0.0 and 1.0) for each query, where 1.0 represents high confidence and 0.0 represents low confidence.

**Acknowledge Ambiguity**: Remember that the original user intent is inherently ambiguous. Your scores should reflect the *relative likelihood* or *reasonableness* of each interpretation, not absolute certainty. The scores do not need to sum to 1.

**Please complete this task STEP-BY-STEP.**
---------------------------------------------
### **Step 1: Re-Analyze the Original Query ($I_r, T_m$)**
Carefully examine the inputs again to identify the core challenges:
1.  **Explicit Instructions ($T_m$)**: What MUST be changed/added/removed?
2.  **Implicit Requirements ($T_m$)**: Are there subtle spatial shifts (viewpoint, scale), thematic changes, or causal implications? How strong are these implications?
3.  **Reference Context ($I_r$)**: What are the key elements? Which are likely relevant vs. potentially irrelevant given $T_m$?
4.  **Potential Conflicts**: Identify any ambiguities or potential conflicts between $T_m$ and $I_r$ that require interpretation.

---------------------------------------------
### **Step 2: Evaluate Each Generated Query Individually**
For each of the three provided queries ($Q_{conservative}$, $Q_{balanced}$, $Q_{reasoning}$):
1.  **Assess Faithfulness to Explicit $T_m$**: Does the query correctly implement all *explicit* instructions from the Modification Text? (This is a minimum requirement).
2.  **Evaluate Risk and Assumption Level**:
    - **Minimalist ($Q_{conservative}$)**: Assess its **safety**. Is it 100% faithful to the explicit text, making **zero unsupported assumptions**? This path prioritizes avoiding error over completeness.
    - **Balanced ($Q_{balanced}$)**: Assess its **assumption level**. It makes a "common sense" assumption (e.g., retaining the 'white shirt'). How reasonable, but still *assumed*, is this leap?
    - **Contextual Inference ($Q_{reasoning}$)**: Assess its **speculation level**. How speculative are its inferences (e.g., the 'pouty expression')? Are these details *grounded* in the text, or are they **high-risk, unverified additions**?
3.  **Check Logical Coherence**: Does the query describe a logically consistent and plausible scene?

---------------------------------------------
### **Step 3: Assign Confidence Scores (0.0 - 1.0)**
Based on your evaluation in Step 2, assign a confidence score to each query ($Q_{conservative}$, $Q_{balanced}$, $Q_{reasoning}$). Consider:
1. **Higher Scores (e.g., 0.9-1.0)**: Assign to queries that are **maximally safe and faithful**. This path **avoids all speculation** and only includes what is explicitly stated. (This aligns with $Q_{conservative}$).
2. **Moderate Scores (e.g., 0.5-0.8)**: Assign to queries that make **reasonable, low-risk assumptions**. This path (like $Q_{balanced}$) makes a plausible, common-sense leap, but it is still an assumption.
3. **Lower Scores (e.g., 0.0-0.4)**: Assign to queries that are **highly speculative**. This path (like $Q_{reasoning}$) **introduces new, ungrounded details** (like 'pouty expression') that are not supported by the original query.

Provide a brief rationale explaining the key factors influencing each score.

---------------------------------------------
## On the input format <Input>
- Input consists of the original Reference Image, the original Modification Text, and the three generated query descriptions.
{
    "Reference Image": <image_url>,
    "Modification Text": "<original_modification_text>",
    "Generated Queries": {
        "Conservative Query": {
            "description": "<Q_minimalist description>",
            "rationale": "<Rationale from generation step>"
        },
        "Balanced Query": {
            "description": "<Q_balanced description>",
            "rationale": "<Rationale from generation step>"
        },
        "Reasoning Enhanced Query": {
            "description": "<Q_contextual_inference description>",
            "rationale": "<Rationale from generation step>"
        }
    }
}

## Guidelines on determining the response <Response>
- Response is a JSON object containing the confidence scores and evaluation rationales for each of the three input queries.
{
    "Query Confidence Scores": {
        "Conservative Query": {
            "confidence_score": <float between 0.0 and 1.0>,
            "evaluation_rationale": "Brief justification for why this score was assigned based on Step 2 & 3 analysis."
        },
        "Balanced Query": {
            "confidence_score": <float between 0.0 and 1.0>,
            "evaluation_rationale": "Brief justification for why this score was assigned based on Step 2 & 3 analysis."
        },
        "Reasoning Enhanced Query": {
            "confidence_score": <float between 0.0 and 1.0>,
            "evaluation_rationale": "Brief justification for why this score was assigned based on Step 2 & 3 analysis."
        }
    }
}

## Example 1
<Input>
{
    "Reference Image": <image_url>,
    "Modification Text": "has only one person wearing the same outfit, the photo is zoomed in.",
    "Generated Queries": {
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
}

<Response>
{
    "Query Confidence Scores": {
        "Conservative Query": {
            "confidence_score": 1.0,
            "evaluation_rationale": "This is the safest and most faithful query. It perfectly executes the explicit instructions ('one person', 'zoomed in'/'close-up') and only retains the 'red tie', making no assumptions about other unmentioned attributes like the shirt."
        },
        "Balanced Query": {
            "confidence_score": 0.8,
            "evaluation_rationale": "Highly plausible, but makes one small assumption: that 'same outfit' definitely includes retaining the 'white shirt'. This is reasonable, but slightly less safe than the conservative query."
        },
        "Reasoning Enhanced Query": {
            "confidence_score": 0.3,
            "evaluation_rationale": "This query is too speculative. It correctly handles the outfit, but introduces 'pouty expression', which is extra information not grounded in the modification text."
        }
    }
}

## Example 2
<Input>
{
    "Reference Image": <image_url>,
    "Modification Text": "is damaged and has the same shape.",
    "Generated Queries": {
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
}

<Response>
{
    "Query Confidence Scores": {
        "Conservative Query": {
            "confidence_score": 1.0,
            "evaluation_rationale": "This is the safest and most faithful query. It perfectly executes the explicit instructions ('damaged', 'same shape') and retains only the core subject ('stop sign'), making zero assumptions about the type of damage or the background."
        },
        "Balanced Query": {
            "confidence_score": 0.7,
            "evaluation_rationale": "This query makes reasonable, but still low-risk, assumptions. It correctly interprets 'damaged' as 'rusted' but also speculates on a new background ('surrounded by bushes') that is not grounded in the reference image or text."
        },
        "Reasoning Enhanced Query": {
            "confidence_score": 0.5,
            "evaluation_rationale": "This query is highly speculative. It not only assumes 'rusted' and 'bushes' but introduces 'old' and 'faded letters', which are unverified, high-risk details not supported by the modification text."
        }
    }
}

## Example 3
<Input>
{
    "Reference Image": <image_url>,
    "Modification Text": "has a dog of a different breed shown with a jolly roger.",
    "Generated Queries": {
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
}

<Response>
{
    "Query Confidence Scores": {
        "Conservative Query": {
            "confidence_score": 1.0,
            "evaluation_rationale": "This is the safest query. It is faithful to the explicit text ('different breed', 'jolly roger') and makes zero unsupported assumptions, correctly discarding the conflicting 'cowboy hat' and 'basket' context."
        },
        "Balanced Query": {
            "confidence_score": 0.8,
            "evaluation_rationale": "This query makes a very reasonable, low-risk assumption. It correctly infers the 'pirate theme' from 'jolly roger' and logically retains the 'hat' concept from the reference image, leading to 'hat with a jolly roger'. This is highly plausible, but still an assumption."
        },
        "Reasoning Enhanced Query": {
            "confidence_score": 0.2,
            "evaluation_rationale": "This query is highly speculative. While the 'pirate costume' is a plausible inference, the addition of 'next to a jug' is a high-risk, unverified detail that is not grounded in either the reference image or the modification text."
        }
    }
}

## Example 4
<Input>
{
    "Reference Image": <image_url>,
    "Modification Text": "shows the cork on a table and has a more everyday background.",
    "Generated Queries": {
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
}

<Response>
{
    "Query Confidence Scores": {
        "Conservative Query": {
            "confidence_score": 1.0,
            "evaluation_rationale": "."
        },
        "Balanced Query": {
            "confidence_score": 0.8,
            "evaluation_rationale": "."
        },
        "Reasoning Enhanced Query": {
            "confidence_score": 0.9,
            "evaluation_rationale": "."
        }
    }
}

'''


fiq_deepseek_mllm_structural_predictor_prompt_CoT = '''
You are an advanced reasoning expert in **fashion image analysis** tasked with generating potential target image descriptions for Composed Image Retrieval.
The relationship between a reference image ($I_r$) and modification text ($T_m$) is complex. The target image should satisfy $T_m$ while selectively retaining relevant **fashion attributes** from $I_r$. And the FashionIQ dataset presents a unique challenge: the modification text ($T_m$) is a synthesis of two parallel human descriptions ($C_a$, $C_b$).
To **systematically explore the ambiguity of user intent** and maximize retrieval coverage, your task is to generate three distinct target query descriptions by following three complementary reasoning paths. Each description must be concise, objective, and **strictly focused on the garment (e.g., shirt, dress, toptee)**.

**Please complete this task STEP-BY-STEP.**
---------------------------------------------
### **Step 1: Analyze the Multi-Modal Queries**
Perform a comprehensive analysis of both inputs:
1.  **Understand the Reference Image ($I_r$)**:
    - Identify the garment type, color, pattern, texture, and silhouette.
    - Analyze key fashion attributes: neckline, sleeves, length, fit, details (e.g., logos, graphics, or text elements).
    - Ignore models, background, and non-garment elements.
2.  **Analyze the Modification Texts ($T_m$)**:
    - Identify explicit edits to key apparel attributes (e.g., color, style and fit, pattern/print, or logo/graphic details) mentioned in $C_a$ or $C_b$.
    - Analyze implicit changes and define the ambiguity space, which $I_r$ attributes are not explicitly addressed by $T_m$ and are thus open to interpretation.
    - Analyze both $C_a$ and $C_b$ to find the core modification intent (their similarities) and key ambiguities (their differences or conflicts).

---------------------------------------------
### **Step 2: Generate Query 1 - The Conservative Path ($Q_{conservative}$)**
This path assumes the user *only* cares about the modification, and all unmentioned/ambiguous attributes are **irrelevant or should be discarded**.
1.  **Reasoning Strategy**: 
    - Start with the **Garment Category** identified in Step 1 (e.g., "shirt", "dress", "toptee").
    - Apply the **explicit attribute modifications** identified from $T_m$ (the synthesized core intent from $C_a$ and $C_b$).
    - **Assume high ambiguity**: Aggressively **exclude** other $I_r$ attributes (such as neckline, fit, or logos) that were *not* explicitly mentioned or confirmed in $T_m$.
    - This path represents the "safest," text-focused interpretation, minimizing assumptions about attribute retention.
2.  **Generate $Q_{conservative}$**: Create a concise description based on this **minimal core hypothesis** after thinking step-by-step, adhering to the Universal Principles in Step 5.

---------------------------------------------
### **Step 3: Generate Query 2 - The Balanced Path ($Q_{balanced}$)**
This path assumes the user wants to **keep** non-conflicting reference attributes.
1.  **Reasoning Strategy**:
    - Start with the Garment Category identified in Step 1 (e.g., "shirt", "dress", "toptee").
    - Apply all **explicit attribute modifications** identified from $T_m$.
    - **Proactively retain** all $I_r$ attributes (such as *neckline*, *fit*, or *logos*) that are *not* in direct, logical conflict with the $T_m$ instructions.
    - This path represents the most direct "blend" of $I_r$ and $T_m$, assuming all reasonable, non-conflicting elements are intended to be kept.
2.  **Generate $Q_{balanced}$**: Create a concise description based on this standard preservation hypothesis after thinking step-by-step, adhering to the Universal Principles in Step 5.

---------------------------------------------
### **Step 4: Generate Query 3 - The Reasoning-Enhanced Path ($Q_{reasoning}$)**
This path assumes the modification implies a **thematic change** or a **generalization** of ambiguous attributes.
1.  **Reasoning Strategy**:
    - Start with the Garment Category identified in Step 1 (e.g., "shirt", "dress", "toptee").
    - Apply all explicit attribute modifications identified from $T_m$.
    - **Infer** a plausible generalization for ambiguous $I_r$ attributes by considering their abstract **thematic role** or **general category** instead of their specific instance.
    - This path explores the most deeply reasoned interpretation of user intent.
2.  **Generate $Q_{reasoning}$**: Create a description based on this **inference-driven** hypothesisafter thinking step-by-step, adhering to the Universal Principles in Step 5.

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
- Input consists of three parts:
{
    "Reference Image": <image_url>,
    "Caption A": <modification_caption_a>,
    "Caption B": <modification_caption_b>
}

## Guidelines on determining the response <Response>
- Response contains three generated queries:
{
    "Conservative Query": {
        "description": "<Target Image Description from Step 2>",
        "rationale": "Brief explanation"
    },
    "Balanced Query": {
        "description": "<Target Image Description from Step 3>", 
        "rationale": "Brief explanation"
    },
    "Reasoning Enhanced Query": {
        "description": "<Target Image Description from Step 4>",
        "rationale": "Brief explanation of combination"
    }
}

Here are some more examples for reference:

## Example 1
<Input>
{
    "Reference Image": "<image_url>",
    "Caption A": "is gold and strapless",
    "Caption B": "button front longer sleeves"
}
<Response>
{
    "Conservative Query": {
        "description": "A black and gold strapless dress with a button front.",
        "rationale": "Focuses only on the garment. Applies the explicit text instructions 'gold,' 'strapless,' and 'button front.' The 'longer sleeves' instruction is discarded as it creates a logical contradiction with 'strapless'."
    },
    "Balanced Query": {
        "description": "A woman wearing a black and gold geometric-patterned strapless maxi dress with a button front.",
        "rationale": "Retains the wearer, the 'black and gold' pattern, and 'maxi length' from the reference image. It applies 'strapless' and 'button front' from the text. It ignores the text's 'longer sleeves' because it contradicts the 'strapless' command."
    },
    "Reasoning Enhanced Query": {
        "description": "A black and gold geometric-patterned long-sleeved maxi dress with a button front.",
        "rationale": "Arbitrates the contradictory text ('strapless' vs 'longer sleeves'). It distrusts 'strapless' as it conflicts with both the reference image ('long-sleeved') and another part of the text ('longer sleeves'). Therefore, it retains the reference image's core structure (long sleeves, maxi length, 'black and gold' pattern) and adds the non-conflicting 'button front' modification."
    }
}

## Example 2
<Input>
{
    "Reference Image": "<image_url>", 
    "Caption A": "is longer and less casual and sleeveless",
    "Caption B": "is a sleeveless black dress"
}
<Response>
{
    "Conservative Query": {
        "description": "A black sleeveless dress that is longer and less casual.",
        "rationale": "Focuses only on the garment. Applies the explicit text instructions 'longer,' 'less casual,' 'sleeveless,' and 'black,' ignoring all specific $I_r$ attributes like the skull graphic or neckline."
    },
    "Balanced Query": {
        "description": "A woman wearing a black sleeveless dress with a scoop neckline and a large skull graphic, that is longer and less casual.",
        "rationale": "Retains non-conflicting $I_r$ attributes (wearer context, scoop neck, skull graphic) while applying $T_m$ modifications (longer, less casual)."
    },
    "Reasoning Enhanced Query": {
        "description": "A long black sleeveless dress with a V-neckline and a decorative skull pattern.",
        "rationale": "Focuses on critical identifiers. Retains the essential 'skull pattern' as it's a key graphic. Infers 'less casual' to mean 'long' (maxi) and infers a 'V-neckline' as a plausible replacement for the more casual 'scoop neck' from the reference."
    }
}

## Example 3
<Input>
{
    "Reference Image": "<image_url>",
    "Caption A": "is more transparent",
    "Caption B": "is more solid colored"
}
<Response>
{
    "Conservative Query": {
        "description": "A solid-colored transparent dress.",
        "rationale": "Focuses only on the garment itself. Applies the contradictory text instructions 'transparent' and 'solid colored' literally, ignoring all reference image context."
    },
    "Balanced Query": {
        "description": "A woman wearing a solid black sleeveless V-neck dress with a transparent overlay.",
        "rationale": "Retains the wearer, the solid black color, and the V-neck style from the reference. Arbitrates the contradictory text by interpreting 'transparent' as an added element (an overlay) to the 'solid colored' dress."
    },
    "Reasoning Enhanced Query": {
        "description": "A solid black sleeveless dress with a fitted bodice and a transparent sheer overlay skirt.",
        "rationale": "Focuses on the garment's key features. This path interprets the 'transparent' and 'solid colored' contradiction as a style replacement. It discards the reference image's ruffles and trim, describing a new item that is solid black (bodice) but also transparent (the sheer skirt)."
    }
}

'''


fiq_deepseek_mllm_structural_predictor_prompt_CoT_shirt = '''
You are an advanced reasoning expert in fashion image analysis tasked with generating potential target image descriptions for Composed Image Retrieval. 
The relationship between a reference image ($I_r$) and modification text ($T_m$) is complex. The target image should satisfy $T_m$ while selectively retaining relevant attributes.
To systematically explore the ambiguity of user intent and maximize retrieval coverage, your task is to generate three distinct target query descriptions by following three complementary reasoning paths.

**Please complete this task STEP-BY-STEP.**
---------------------------------------------
### **Step 1: Analyze the Multi-Modal Queries**
Perform a comprehensive analysis of both inputs:
1.  **Understand the Reference Image ($I_r$)**:
    - Identify the primary garment (i.e., shirt and toptee) and its context.
    - Analyze key 'shirt'/'toptee' attributes:
        - Garment Properties: Identify the specific style, core attributes (color, pattern, fit, length), and component details (neckline, sleeves, collar, pockets, closures).
        - Critical Identifiers: Give special attention to describing any graphics, logos, or text (their content, placement, color, etc.).
2.  **Analyze the Modification Texts ($T_m$)**:
    - Break down modification text into individual modification instructions (e.g., color change, graphic addition/negation, style change).

---------------------------------------------
### **Step 2: Generate Query 1 - The Conservative Path ($Q_{conservative}$)**
This path assumes the user's intent is narrowly focused on the shirt or the toptee. 
1.  **Reasoning Strategy**: 
    - Start with the specific Garment Category identified in Step 1 (e.g., "shirt", "t-shirt", "polo shirt").
    - Apply all modification instructions from $T_m$ in Step 1.
    - Ignore models, background, and non-garment elements.
    - This path represents the "safest," text-focused interpretation, minimizing assumptions about retaining other contextual elements.
2.  **Generate $Q_{conservative}$**: Create a concise description based on this **minimal core hypothesis** after thinking step-by-step, adhering to the Universal Principles in Step 5.

---------------------------------------------
### **Step 3: Generate Query 2 - The Balanced Path ($Q_{balanced}$)**
This path assumes the user wants to keep relevant reference attributes and context.
1.  **Reasoning Strategy**:
    - Start with the full analysis of the Reference Image ($I_r$) from Step 1.
    - Apply the modification instruction ($T_m$) step-by-step.
    - Proactively retain all $I_r$ attributes (such as neckline, fit, logos, or other contexts) that are not in direct, logical conflict with the modification instruction ($T_m$).
    - This path represents the most direct "combination" of $I_r$ and $T_m$, assuming all reasonable, non-conflicting elements are intended to be kept.
2.  **Generate $Q_{balanced}$**: Create a concise description based on this standard preservation hypothesis after thinking step-by-step, adhering to the Universal Principles in Step 5.

---------------------------------------------
### **Step 4: Generate Query 3 - The Reasoning-Enhanced Path ($Q_{reasoning}$)**
This path assumes the user is focused on the most identifiable features that define a specific shirt or toptee.
1.  **Reasoning Strategy**:
    - Start with the full analysis of the Reference Image ($I_r$) from Step 1.
    - Enhance this description by ensuring the most critical shirt/ toptee identifiers (from Step 1) are explicitly stated.
    - Give special attention to describing any graphics, logos, or text (their content, placement, color, etc.), and the specific style (e.g., "Polo shirt", "V-neck t-shirt").
    - However, if $T_m$ explicitly defines a new garment style that replaces the reference style, then do not retain identifiers specific to the original style (like a t-shirt graphic).
    - This path generates the most detailed, feature-rich query, focusing on what makes the garment unique.
2.  **Generate $Q_{reasoning}$**: Create a description based on this **feature-driven** hypothesisafter thinking step-by-step, adhering to the Universal Principles in Step 5.

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
8. Only generate three target descriptions.

## On the input format <Input>
- Input consists of three parts:
{
    "Reference Image": <image_url>,
    "Modification Text": <modification_text>.
}

## Guidelines on determining the response <Response>
- Response is a JSON object containing the three generated queries with brief rationales.
{
    "Conservative Query": {
        "description": "<Target Image Description from Step 2>",
        "rationale": "Brief explanation"
    },
    "Balanced Query": {
        "description": "<Target Image Description from Step 3>", 
        "rationale": "Brief explanation"
    },
    "Reasoning Enhanced Query": {
        "description": "<Target Image Description from Step 4>",
        "rationale": "Brief explanation of combination"
    }
}

Here are some more examples for reference:

## Example 1
<Input>
{
    "Reference Image": "<image_url>",
    "Modification Text": "is black and white with checks and is lighter"
}
<Response>
{
    "Conservative Query": {
        "description": "A lighter black and white checked shirt.",
        "rationale": "Focuses only on the garment attributes (shirt, lighter, black and white, checked) from the modification text, discarding the wearer and original pattern."
    },
    "Balanced Query": {
        "description": "A man wearing a lighter black and white checked long-sleeved shirt with colorful cuffs.",
        "rationale": "Retains the wearer, long sleeves, and colorful cuffs from the reference, while applying the modification for a lighter, black and white checked pattern."
    },
    "Reasoning Enhanced Query": {
        "description": "A man wearing a lighter black and white checked button-up shirt with a collar and colorful patterned inner cuffs.",
        "rationale": "Enhances the balanced query by specifying the style (button-up shirt, collar) and adding detail to the retained cuffs (patterned inner)."
    }
}

## Example 2
<Input>
{
    "Reference Image": "<image_url>", 
    "Modification Text": "has short sleeves and has shorter sleeves"
}
<Response>
{
    "Conservative Query": {
        "description": "A short-sleeved t-shirt.",
        "rationale": "Focuses only on the garment category (t-shirt) and the modification text (short sleeves), discarding all other reference attributes like pattern, color, or neckline."
    },
    "Balanced Query": {
        "description": "A man wearing a multi-color striped t-shirt with short sleeves.",
        "rationale": "Retains the wearer (context) and the primary pattern (stripes) from the reference, as the modification 'short sleeves' is redundant and non-conflicting."
    },
    "Reasoning Enhanced Query": {
        "description": "A man wearing a red, pink, and navy horizontal striped V-neck t-shirt with short sleeves.",
        "rationale": "Enhances the balanced query by specifying the critical identifiers: the colors of the stripes (red, pink, navy) and the specific style (V-neck)."
    }
}

## Example 3
<Input>
{
    "Reference Image": "<image_url>",
    "Modification Text": "white tunic ruffled bodice short sleeves and has longer sleeves"
}
<Response>
{
    "Conservative Query": {
        "description": "A white tunic top with a ruffled bodice and short sleeves.",
        "rationale": "Focuses only on the garment attributes explicitly mentioned in the text (white, tunic, ruffled bodice, short sleeves). Discards all reference attributes (wearer, pattern, v-neck)."
    },
    "Balanced Query": {
        "description": "A woman wearing a white tunic top with a ruffled bodice and short sleeves.",
        "rationale": "Applies the modification text (white tunic, ruffled bodice, short sleeves) as a style replacement. The original pattern is discarded, but the non-conflicting wearer/context is retained."
    },
    "Reasoning Enhanced Query": {
        "description": "A woman wearing a white tunic top with a stand-up collar, button details, a ruffled bodice, and short sleeves.",
        "rationale": "Enhances the balanced query by adding specific component details (stand-up collar, button details) that are commonly associated with the new 'ruffled bodice' shirt style."
    }
}

'''

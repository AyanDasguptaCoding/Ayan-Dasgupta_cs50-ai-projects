import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id` in the
    `inputs` sequence of tokens. The returned index should be an integer.
    If the mask token isn't present in the inputs, return None.
    """
    input_ids = inputs["input_ids"][0]
    for i, token_id in enumerate(input_ids):
        if token_id == mask_token_id:
            return i
    return None


def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing an RGB color that
    corresponds to the given `attention_score`. The first integer in
    the tuple should represent the red value, the second integer should
    represent the green value, and the third integer should represent
    the blue value.
    
    Each value should be an integer between 0 and 255, inclusive. The
    higher the attention score, the lighter the color should be.
    """
    intensity = int(round(attention_score * 255))
    return (intensity, intensity, intensity)


def visualize_attentions(tokens, attentions):
    """
    Generate one attention diagram for each of the model's attention heads.
    Each diagram should visualize the attention scores for a particular
    attention head, with the rows representing the tokens doing the attending
    and the columns representing the tokens being attended to.
    """
    num_layers = len(attentions)
    num_heads = len(attentions[0][0])
    
    for layer in range(num_layers):
        for head in range(num_heads):
            # Get attention scores for this head
            attention = attentions[layer][0][head]
            
            # Generate diagram (1-indexed for display)
            generate_diagram(
                layer + 1,
                head + 1,
                tokens,
                attention
            )


def generate_diagram(layer_number, head_number, tokens, attention):
    """
    Generate a diagram visualizing the attention scores for a particular
    attention head. The diagram shows one row and column for each token
    in `tokens`, and cells are shaded based on `attention`, with darker
    cells corresponding to higher attention scores.
    """
    # Create new image
    cell_size = 100
    cell_padding = 10
    width = cell_size * len(tokens) + cell_padding * (len(tokens) - 1)
    height = cell_size * len(tokens) + cell_padding * (len(tokens) - 1)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Draw each cell
    for i, token in enumerate(tokens):
        for j in range(len(tokens)):
            # Get attention score and color
            score = attention[i, j].numpy()
            color = get_color_for_attention_score(score)

            # Draw cell
            x = j * (cell_size + cell_padding)
            y = i * (cell_size + cell_padding)
            draw.rectangle(
                ((x, y), (x + cell_size, y + cell_size)),
                fill=color,
                outline="black"
            )

            # Draw token labels in first row and column
            if i == 0:
                draw.text(
                    (x + cell_size / 2, y + cell_size + 5),
                    token,
                    fill="black",
                    anchor="mt"
                )
            if j == 0:
                draw.text(
                    (x - 5, y + cell_size / 2),
                    token,
                    fill="black",
                    anchor="rm"
                )

    # Save image
    output_filename = f"attention_layer_{layer_number}_head_{head_number}.png"
    img.save(output_filename)
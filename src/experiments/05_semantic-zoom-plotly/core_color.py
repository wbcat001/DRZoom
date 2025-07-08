
import random


def generate_colormap(df, attribute_name:str, default_colormap=None):
    """
    Generate a colormap for the given attribute name in the dataframe.
   
    default_colormap example
    {
        "Setup": "skyblue",
        "Inciting Incident": "green",
        "Turning Point": "orange",
        "Climax": "red",
        "Resolution": "purple",
        }
    """
    if default_colormap:

        colormap = {value: default_colormap[value] if value in default_colormap else f"#{random.randint(0, 0xFFFFFF):06x}" for value in df[attribute_name].unique() }
    else:
        colormap = {value: f"#{random.randint(0, 0xFFFFFF):06x}" for value in df[attribute_name].unique()}
    return colormap
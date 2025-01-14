import os


def save_string_to_file(content: str, file_path: str) -> None:
    """
    Saves the given string content to a .txt file at the specified file path.

    Args:
        content (str): The string content to be saved.
        file_path (str): The path where the .txt file should be saved.
    """
    try:
        with open(file_path, "w") as file:
            file.write(content)
        print(f"File saved successfully at {file_path}")
    except IOError as e:
        print(f"An error occurred while saving to file: {e}")


def read_string_from_file(file_path: str) -> str:
    """
    Reads the content of a .txt file from the specified file path and returns it as a string.

    Args:
        file_path (str): The path of the .txt file to be read.

    Returns:
        str: The content of the file as a string.
    """
    try:
        with open(file_path, "r") as file:
            content = file.read()
        print(f"File read successfully from {file_path}")
        return content
    except IOError as e:
        print(f"An error occurred while reading from file: {e}")
        return ""


def delete_json_file():
    # Define the path to the file
    file_path = "./chat_histories/input_test.json"
    try:
        # Remove the file
        os.remove(file_path)
        print(f"File {file_path} has been deleted successfully.")
    except FileNotFoundError:
        print(f"File {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred while deleting the file: {e}")

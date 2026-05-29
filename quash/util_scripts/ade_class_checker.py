import difflib
import ast
from tqdm import tqdm
import nltk
from nltk.corpus import words

def load_list_from_file(file_path):
    """Loads a list of strings from a text file, removing newlines."""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def find_matches(list_all, list_used, english_words, threshold=0.5):
    """Find exact matches and potential close matches."""
    exact_matches = []
    unmatched_items = []
    completely_unmatched = []
    potential_matches = {}

    # Find exact matches and keep track of unmatched items
    for item in tqdm(list_all):
        if item in list_used:
            exact_matches.append(item)
        else:
            unmatched_items.append(item)

    # For unmatched items, find potential close matches
    for item in tqdm(unmatched_items):
        # Get potential matches based on similarity ratio
        close_matches = difflib.get_close_matches(item, list_used, n=3, cutoff=threshold)

        # Filter out any potential match that is a valid English word
        filtered_matches = [match for match in close_matches if match.lower() not in english_words]

        if filtered_matches:
            potential_matches[item] = filtered_matches
        else:
            completely_unmatched.append(item)

    return exact_matches, potential_matches, completely_unmatched

def main(english_words):
    # Load the lists from files
    list_all = load_list_from_file('config/ade_classes.txt')
    with open("config/ade847.json", "r") as f:
        file_content = f.read()
        list_used = ast.literal_eval(file_content)

    # Find exact matches and potential matches
    exact_matches, potential_matches, completely_unmatched = find_matches(list_all, list_used, english_words)



    print("Potential Matches for Unmatched Items:")
    for unmatched, matches in potential_matches.items():
        print(f"{unmatched}: {', '.join(matches)}")

    print("")
    print("All classes", len(list_all))
    print("Exact Matches:", len(exact_matches))
    print("Potential matches:", len(potential_matches))
    print("Completely unmatched:", len(completely_unmatched))
    sum = len(exact_matches) + len(potential_matches) + len(completely_unmatched)
    print("Sum:", sum)



if __name__ == "__main__":
    nltk.download('words')
    english_words = set(words.words())
    main(english_words)

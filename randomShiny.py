from shiny import App, ui, render
import numpy as np
import random

# --- Function Definitions (from notebook) ---

def groupcountlist(string):
    """Return list of run length in a sequence of 1s and 0s

    Input is a series of 0s and 1s of positive length

    Output is a list that contains each run length L(1, 2, 3, etc) in order of when it appears in the sequence"""
    L = []
    if not string:
        return L
    pb = string[0]
    en = 1
    for b in string[1:]:
      if b == pb:
        en = en+1
      else:
        L.append(en)
        en = 1
        pb = b
    L.append(en)
    return(L)

def groupcount(string):
    """Return number runs of each length in a sequence of 1s and 0s

    Input is a series of 0s and 1s

    Output is a list that assigns each run length L(1, 2, 3, etc) to the number of times a consecutive series of 1s or 0s of length L appear in a sequence"""
    runs = groupcountlist(string)
    if not runs: # Handle empty string case for groupcount
        return [0] * (len(string) + 1)
    hist =  [0]*(len(string)+1)
    for  r in runs:
      # Ensure r is within bounds of hist
      if r < len(hist):
        hist[r] = hist[r] + 1
    return hist

def generate_random_binary_sequence(k):
  """Return a random binary sequence of length k."""
  return ''.join(random.choices(['0','1'],k=k))

def array_of_rand_groupcounts(length=100,n=20):
  """Returns an array, where each of 'n' rows is the group counts for a random string of length 'length'
  Written by oren"""
  r = []
  for i in range(n):
    r.append((groupcount(generate_random_binary_sequence(length))))
  # Pad histories to a consistent length if necessary. Max possible run length is `length`.
  # The `groupcount` function creates a list of `length + 1` size.
  max_hist_len = length + 1
  padded_r = []
  for hist in r:
      if len(hist) < max_hist_len:
          padded_hist = hist + [0] * (max_hist_len - len(hist))
          padded_r.append(padded_hist)
      else:
          padded_r.append(hist)
  return np.array(padded_r)

def array_of_groupcounts(d):
  """Returns an array of group counts for a list of strings of 0 and 1"""
  if not d: # Handle empty input list
      return np.array([[]])
  # Determine max length for padding to ensure all arrays in list have same shape
  max_len = 0
  for s in d:
      max_len = max(max_len, len(s) + 1) # groupcount returns list of length + 1
  
  results = []
  for x in d:
      gc = groupcount(x)
      # Pad with zeros to max_len
      padded_gc = gc + [0] * (max_len - len(gc))
      results.append(padded_gc)
  return np.array(results)

def filterbi(li):
  """:Given an input list of strings, returns a list only of strings containing only 01
  Written by oren"""
  liss = []
  for s in li:
    # Ensure s is a string before iterating
    if not isinstance(s, str):
        continue
    valid_sequence = True
    for c in s:
      if (c != '0' and c != '1'):
        valid_sequence = False
        break
    if valid_sequence:
      liss.append(s)
  return liss

def humscore(s):
  gs = groupcount(s)
  score = 0
  # Ensure gs has enough elements before accessing by index
  # Assuming s has length 100, gs should have length 101 (indices 0-100)
  # Adjust indices if s length varies, or handle IndexError
  # For a sequence of length 100, max run length is 100, so index 100 is valid.
  # The original code assumes gs is long enough, let's replicate that logic, 
  # but add checks for robustness, especially if sequence length varies.
  
  # Pad gs to ensure indices can be accessed safely for a sequence of length 100
  target_len = 101 # For sequence length 100, groupcount returns list of size 101
  if len(gs) < target_len:
      gs = gs + [0] * (target_len - len(gs))

  if gs[1] > 30:
    score +=(gs[1]-27)
  if gs[1]>35:
    score += (gs[1]-32)*2
  if gs[2] < 9:
    score += (9-gs[2])
  if gs[2] < 5:
    score += (6-gs[2])*3
  if gs[1] > 40:
    score +=(gs[1]-40)*5
  if gs[1] < 15:
    score +=(15-gs[1])
  if sum(gs[8:9]) > 1: # sum(gs[8:9]) is just gs[8] if it exists
    score+=gs[8]*2 if len(gs) > 8 else 0
  score+=sum(gs[10:12])*5 # sum of gs[10] and gs[11]
  score+=sum(gs[12:])*25
  return score

def ishuman(s):
  return humscore(s) > 7.5

# --- Pre-computation for comparison (as derived from the notebook) ---
# This will be computed once when the Shiny app starts.

# Set random seed for reproducibility of random sequence generation
random.seed(4765696532232)

# Not repeating the random sequence generation.
# Generate a large number of random sequences and their group counts
# This `ar` is used to calculate `avar` which serves as a baseline for random sequences.
# It's important to keep the `length` and `n` consistent with the notebook's analysis.
#RAND_SEQ_LENGTH = 100
#NUM_RAND_SAMPLES = 100000
#ar = array_of_rand_groupcounts(RAND_SEQ_LENGTH, NUM_RAND_SAMPLES)

## Calculate the average group counts for random sequences
#avar = np.average(ar, 0)

# --- Shiny UI ---
app_ui = ui.page_fluid(
    ui.h2("Human Randomness Classifier"),
    ui.hr(),
    ui.input_text_area(
        "sequence_input",
        "Enter a binary sequence (0s and 1s only, e.g., 010101...)",
        value="0101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101",
        rows=5,
        cols=100
    ),
    ui.output_text("validation_message"),
    ui.output_text("score_output"),
    ui.output_text("classification_output"),
    ui.hr(),
    ui.h3("How it works:"),
    ui.markdown("This app evaluates how 'human-like' a binary sequence is based on its run-length distribution compared to truly random sequences. The `humscore` function assigns a score based on various criteria such as frequencies of runs of length 1, 2, and longer runs. A sequence is classified as 'human-like' if its score exceeds a threshold of 7.5.")
)

# --- Shiny Server ---
def server(input, output, session):

    @render.text
    def validation_message():
        sequence = input.sequence_input()
        if not sequence:
            return "Please enter a sequence."
        if not all(c in '01' for c in sequence):
            return "Error: Sequence must contain only 0s and 1s."
        return ""

    @render.text
    def score_output():
        sequence = input.sequence_input()
        if not sequence or not all(c in '01' for c in sequence):
            return ""
        try:
            score = humscore(sequence)
            return f"Human Score: {score:.2f}"
        except Exception as e:
            return f"Error calculating score: {e}"

    @render.text
    def classification_output():
        sequence = input.sequence_input()
        if not sequence or not all(c in '01' for c in sequence):
            return ""
        try:
            is_human_like = ishuman(sequence)
            status = "Human-like" if is_human_like else "Random-like"
            return f"Classification: {status}"
        except Exception as e:
            return f"Error classifying: {e}"


app = App(app_ui, server)
